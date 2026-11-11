"""Training loop: bf16 decoder/discriminators, wall-clock budget, resumable checkpoints."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import contextlib
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import __version__
from .audio import mel_spectrogram, spec_to_mel, spectrogram
from .checkpoint import CheckpointSaver, discard, load_latest_checkpoint, to_cpu
from .config import load_config
from .data.dataset import Collate, SortishBatchSampler, TTSDataset, _worker_init
from .inference import synthesize
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .models import MultiPeriodDiscriminator, build_generator, slice_segments
from .monotonic_align import HAVE_NUMBA
from .text import Tokenizer
from .utils import fmt_time, log


class StopFlag:
    def __init__(self):
        self.flag, self.count = False, 0

    def __call__(self, signum, frame):
        self.count += 1
        self.flag = True
        if self.count == 1:
            log("Stop requested: finishing this step, saving a checkpoint, then exiting (repeat to force-quit).")
        else:
            raise KeyboardInterrupt


def run_eval(net_g, tok, hp, eval_items, spk_index, step, run_dir, writer, data_audio=None):
    import soundfile as sf
    was_training, amp = net_g.training, net_g.amp_dtype
    net_g.eval()
    net_g.amp_dtype = None  # full precision for listening samples
    out = Path(run_dir) / "samples" / f"step_{step:08d}"
    out.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(run_dir) / "samples" / "ground_truth"
    for e in eval_items:
        try:
            audio = synthesize(net_g, tok, hp, e["text"], spk_index[e["speaker"]], warn=False)
            sf.write(str(out / f"{e['speaker']}.wav"), audio, hp["sample_rate"])
            if writer is not None and len(audio):
                writer.add_audio(f"sample/{e['speaker']}", torch.from_numpy(audio).unsqueeze(0), step,
                                 sample_rate=hp["sample_rate"])
            if data_audio is not None and not (gt_dir / f"{e['speaker']}.wav").exists():
                gt_dir.mkdir(parents=True, exist_ok=True)
                gt = np.array(data_audio[e["offset"]:e["offset"] + e["length"]], dtype=np.float32) / 32768.0
                sf.write(str(gt_dir / f"{e['speaker']}.wav"), gt, hp["sample_rate"])
                (gt_dir / f"{e['speaker']}.txt").write_text(e["text"], encoding="utf-8")
        except Exception as ex:
            log(f"WARNING: eval sample for {e['speaker']} failed: {type(ex).__name__}: {ex}")
    net_g.amp_dtype = amp
    net_g.train(was_training)
    log(f"eval samples written to {out}")


def train(args):
    data_dir, run_dir = Path(args.data), Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for leftover in run_dir.glob(".*.tmp"):  # half-written checkpoint from a disconnected session
        discard(leftover)
    man = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    else:
        log("WARNING: no GPU found - training on CPU will be extremely slow.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    symbols, all_speakers = man["symbols"], man["speakers"]
    speakers = all_speakers
    if args.speakers.strip():
        speakers = []
        for w in [s.strip() for s in args.speakers.split(",") if s.strip()]:
            if w in all_speakers:
                speakers.append(w)
            elif w.isdigit() and int(w) < len(all_speakers):
                speakers.append(all_speakers[int(w)])
            else:
                raise SystemExit(f"Unknown speaker {w!r}. Available: {all_speakers}")
    hp = load_config(args.config)
    hp.update(man.get("hparams", {}))

    ckpt_path, state = load_latest_checkpoint(run_dir)
    if state is not None:
        if state["symbols"] != symbols or state["speakers"] != speakers:
            raise SystemExit(f"{ckpt_path} was trained with a different vocabulary or speaker list. "
                             f"Use a new --run_dir (or the same --speakers / data as before).")
        hp = state["hparams"]
        log(f"Resuming from {ckpt_path.name}: step {state['step']}, epoch {state['epoch']}, "
            f"{fmt_time(state['train_seconds'])} already trained")

    spk_index = {s: i for i, s in enumerate(speakers)}
    train_items = [e for e in man["train"] if e["speaker"] in spk_index]
    val_items = [e for e in man["val"] if e["speaker"] in spk_index]
    eval_items, seen = [], set()
    for e in val_items:  # one listening sample per speaker
        if e["speaker"] not in seen and len(eval_items) < args.n_eval:
            eval_items.append(e)
            seen.add(e["speaker"])

    tok = Tokenizer(symbols)
    audio_path = data_dir / man["audio_file"]
    ds = TTSDataset(train_items, audio_path, tok, spk_index, hp["add_blank"])
    sampler = SortishBatchSampler(ds.lengths, args.batch_size, seed=args.seed)
    if len(sampler) == 0:
        raise SystemExit("Not enough training utterances for one batch - lower --batch_size.")
    seg_frames = hp["segment_size"] // hp["hop"]
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=_worker_init,
                     collate_fn=Collate(hp["hop"], seg_frames))
    if args.workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(ds, batch_sampler=sampler, **loader_kw)

    multi = len(speakers) > 1
    net_g = build_generator(hp, len(symbols), len(speakers)).to(device)
    net_d = MultiPeriodDiscriminator().to(device)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None
    if amp_dtype is not None and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        log("bf16 not supported on this GPU -> using fp32/TF32")
        amp_dtype = None
    net_g.amp_dtype = amp_dtype
    opt_kw = dict(lr=hp["lr"], betas=tuple(hp["betas"]), eps=hp["eps"])
    if device.type == "cuda":
        opt_kw["fused"] = True
    optim_g = torch.optim.AdamW(net_g.parameters(), **opt_kw)
    optim_d = torch.optim.AdamW(net_d.parameters(), **opt_kw)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp["lr_decay"])
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp["lr_decay"])

    def restore(st):
        net_g.load_state_dict(st["G"])
        net_d.load_state_dict(st["D"])
        optim_g.load_state_dict(st["optim_g"])
        optim_d.load_state_dict(st["optim_d"])
        sched_g.load_state_dict(st["sched_g"])
        sched_d.load_state_dict(st["sched_d"])

    step, epoch, train_seconds = 0, 0, 0.0
    if state is not None:
        restore(state)
        step, epoch, train_seconds = state["step"], state["epoch"], float(state["train_seconds"])
        del state

    def snapshot():
        return to_cpu(dict(G=net_g.state_dict(), D=net_d.state_dict(), optim_g=optim_g.state_dict(),
                           optim_d=optim_d.state_dict(), sched_g=sched_g.state_dict(), sched_d=sched_d.state_dict(),
                           step=step, epoch=epoch, train_seconds=train_seconds, hparams=hp, symbols=symbols,
                           speakers=speakers, speaker_info=man.get("speaker_info", []), version=__version__))

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "logs"), flush_secs=60)
    except Exception:
        log("tensorboard not available - logging to stdout only")

    n_g = sum(p.numel() for p in net_g.parameters()) / 1e6
    n_d = sum(p.numel() for p in net_d.parameters()) / 1e6
    hours_data = ds.lengths.sum() / hp["sample_rate"] / 3600
    budget = args.hours * 3600.0
    log(f"data: {len(ds)} utterances ({hours_data:.1f} h), {len(speakers)} speaker(s), vocab {len(symbols)}, "
        f"{len(sampler)} steps/epoch at batch {args.batch_size}")
    log(f"model: G {n_g:.1f}M params, D {n_d:.1f}M params | precision: "
        f"{'bf16 decoder+discriminators, fp32/TF32 rest' if amp_dtype else 'fp32/TF32'} | "
        f"alignment search: {'numba' if HAVE_NUMBA else 'numpy (slow - pip install numba)'}")
    log(f"time budget: {fmt_time(budget)} total, {fmt_time(budget - train_seconds)} left in this run")
    if train_seconds >= budget:
        log("Time budget already used up. Increase --hours to keep training.")
        return

    stop = StopFlag()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    saver = CheckpointSaver(run_dir, keep=args.keep_ckpts)
    data_audio = np.memmap(audio_path, dtype=np.int16, mode="r")
    net_g.train()
    net_d.train()

    def autocast():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    def train_step(batch):
        text, text_len, wav, wav_len, sid = (t.to(device, non_blocking=True) for t in batch)
        wav = wav.float() / 32768.0
        with torch.no_grad():
            spec = spectrogram(wav, hp["n_fft"], hp["hop"], hp["win"])
        spec_len = wav_len // hp["hop"]
        y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
            text, text_len, spec, spec_len, sid if multi else None)
        with torch.no_grad():
            y_mel = spec_to_mel(slice_segments(spec, ids_slice, seg_frames), hp)
        y_hat_mel = mel_spectrogram(y_hat.squeeze(1).float(), hp)
        y = slice_segments(wav.unsqueeze(1), ids_slice * hp["hop"], hp["segment_size"])

        # --- discriminator ---
        net_d.requires_grad_(True)
        with autocast():
            d_real, d_fake, _, _ = net_d(y, y_hat.detach())
        loss_d = discriminator_loss(d_real, d_fake)
        optim_d.zero_grad(set_to_none=True)
        loss_d.backward()
        gn_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), args.grad_clip)
        ok_d = bool(torch.isfinite(gn_d))
        if ok_d:
            optim_d.step()

        # --- generator ---
        net_d.requires_grad_(False)  # no weight-gradients for D here
        with autocast():
            _, d_fake, f_real, f_fake = net_d(y, y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hp["c_mel"]
        loss_dur = torch.sum(l_length.float())
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hp["c_kl"]
        loss_fm = feature_loss(f_real, f_fake)
        loss_gen = generator_loss(d_fake)
        loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad(set_to_none=True)
        loss_g.backward()
        gn_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), args.grad_clip)
        ok_g = bool(torch.isfinite(gn_g))
        if ok_g:
            optim_g.step()
        vals = torch.stack([loss_d.detach(), loss_g.detach(), loss_gen.detach(), loss_fm.detach(),
                            loss_mel.detach(), loss_dur.detach(), loss_kl.detach(), gn_d.detach().float(),
                            gn_g.detach().float()]).float()
        return vals, ok_d and ok_g

    names = ["loss_d", "loss_g", "adv", "fm", "mel", "dur", "kl", "gnorm_d", "gnorm_g"]
    acc, acc_n = torch.zeros(len(names), device=device), 0
    t_prev = t_log = time.time()
    step_log = step
    last_save = last_eval = time.time()
    saved_step = evaled_step = -1
    bad_streak, n_skipped, n_oom = 0, 0, 0
    done, start_step = False, step
    log("training started" + (" (first steps are slower: cuDNN autotuning + numba compile)" if step == 0 else ""))
    while not done:
        sampler.set_epoch(epoch)
        for batch in loader:
            oom = False
            try:
                vals, ok = train_step(batch)
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                optim_g.zero_grad(set_to_none=True)
                optim_d.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                n_oom += 1
                log(f"WARNING: out of GPU memory on a batch (x{n_oom}) - skipped. "
                    f"If this repeats, lower --batch_size or prepare with a smaller --max_sec.")
                continue
            step += 1
            if ok and bool(torch.isfinite(vals).all()):
                acc += vals
                acc_n += 1
                bad_streak = 0
            else:
                bad_streak += 1
                n_skipped += 1
                if bad_streak in (1, 10) or bad_streak % 50 == 0:
                    log(f"WARNING: non-finite loss/gradients at step {step} - update skipped ({bad_streak} in a row)")
                if bad_streak >= args.max_bad_steps:
                    p, st = load_latest_checkpoint(run_dir)
                    if st is not None:
                        restore(st)
                        log(f"reloaded {p.name} after {bad_streak} bad steps")
                        del st
                    bad_streak = 0

            now = time.time()
            train_seconds += now - t_prev
            t_prev = now

            if step % args.log_every == 0 and acc_n > 0:
                m = (acc / acc_n).tolist()
                its = (step - step_log) / max(1e-6, now - t_log)
                left = budget - train_seconds
                log(f"step {step:>7} ep {epoch:>4} | {its:5.2f} it/s | {fmt_time(train_seconds)} / {fmt_time(budget)}"
                    f" (~{int(step + its * left):,} steps at end) | "
                    + " ".join(f"{k}={v:.3f}" for k, v in zip(names[:7], m[:7]))
                    + f" | gn_d={m[7]:.0f} gn_g={m[8]:.0f} | lr={optim_g.param_groups[0]['lr']:.2e}")
                if writer is not None:
                    for k, v in zip(names, m):
                        writer.add_scalar(f"train/{k}", v, step)
                    writer.add_scalar("train/lr", optim_g.param_groups[0]["lr"], step)
                    writer.add_scalar("train/steps_per_sec", its, step)
                acc.zero_()
                acc_n, t_log, step_log = 0, now, step

            if now - last_eval >= args.eval_every_min * 60:
                run_eval(net_g, tok, hp, eval_items, spk_index, step, run_dir, writer, data_audio)
                last_eval, evaled_step = time.time(), step
            if now - last_save >= args.save_every_min * 60:
                saver.save(snapshot, step)
                last_save, saved_step = time.time(), step
            if train_seconds >= budget or stop.flag or (args.max_steps and step - start_step >= args.max_steps):
                done = True
                break
        else:  # finished a full pass over the data
            epoch += 1
            sched_g.step()
            sched_d.step()

    reason = "stop requested" if stop.flag else ("time budget reached" if train_seconds >= budget else "max_steps reached")
    log(f"stopping ({reason}) at step {step}; skipped steps: {n_skipped} non-finite, {n_oom} OOM")
    if saved_step != step:
        saver.save(snapshot, step, blocking=True)
    saver.wait()
    if evaled_step != step:
        run_eval(net_g, tok, hp, eval_items, spk_index, step, run_dir, writer, data_audio)
    if writer is not None:
        writer.close()
    log(f"finished. Inference checkpoint: {run_dir / 'G_latest.pt'}")
