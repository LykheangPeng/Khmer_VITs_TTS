"""Command line: khmer-vits {download,prepare,train,synth}."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from pathlib import Path

from . import __version__
from .utils import log


def synth_cli(args):
    import soundfile as sf

    from .inference import load_generator, synthesize
    net_g, hp, tok, speakers = load_generator(args.ckpt)
    texts = [args.text] if args.text else [l.strip() for l in open(args.text_file, encoding="utf-8") if l.strip()]
    if args.speaker == "all":
        sids = list(range(len(speakers)))
    elif args.speaker in speakers:
        sids = [speakers.index(args.speaker)]
    else:
        sids = [int(args.speaker)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    multi_out = len(texts) * len(sids) > 1
    for ti, text in enumerate(texts):
        for sid in sids:
            audio = synthesize(net_g, tok, hp, text, sid, args.noise_scale, args.noise_scale_w, args.length_scale)
            path = out if not multi_out else out.with_name(f"{out.stem}_{ti:03d}_{speakers[sid]}{out.suffix}")
            sf.write(str(path), audio, hp["sample_rate"])
            log(f"wrote {path} ({len(audio) / hp['sample_rate']:.2f} s, speaker {speakers[sid]})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="khmer-vits", description="Khmer VITS text-to-speech: download, prepare, train, synthesize.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download + extract the Khmer ASR Cultural Dataset (needs $MDC_API_KEY)")
    d.add_argument("--out", default="data/raw", help="extraction folder")
    d.add_argument("--cache_dir", default="data/mdc", help="where the .tar.gz archive is stored")
    d.add_argument("--dataset_id", default="cmkcy8in2004umo0775mye43g")
    d.add_argument("--archive", default="", help="use an already downloaded .tar.gz instead of the API")

    p = sub.add_parser("prepare", help="raw dataset -> preprocessed training cache")
    p.add_argument("--src", required=True, help="folder that contains metadata.csv and data/**/*.wav")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None, help="JSON config (default: built-in VITS base)")
    p.add_argument("--workers", type=int, default=0, help="processes (default: all CPU cores)")
    p.add_argument("--min_sec", type=float, default=1.0)
    p.add_argument("--max_sec", type=float, default=20.0)
    p.add_argument("--trim_db", type=float, default=40.0, help="silence threshold below the loudest frame")
    p.add_argument("--trim_margin", type=float, default=0.1, help="seconds of silence kept at each end")
    p.add_argument("--target_db", type=float, default=-20.0, help="loudness target (RMS dBFS)")
    p.add_argument("--min_char_count", type=int, default=3)
    p.add_argument("--rate_z", type=float, default=4.0, help="drop speaking-rate outliers beyond this robust z")
    p.add_argument("--val_per_speaker", type=int, default=2)

    t = sub.add_parser("train", help="train / resume")
    t.add_argument("--data", required=True)
    t.add_argument("--run_dir", required=True, help="checkpoints, samples, TensorBoard logs")
    t.add_argument("--config", default=None, help="JSON config (default: built-in VITS base); ignored when resuming")
    t.add_argument("--hours", type=float, default=11.0, help="total training wall-clock budget, across resumes")
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--speakers", default="", help="comma-separated speaker IDs or indices (default: all)")
    t.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--grad_clip", type=float, default=1000.0)
    t.add_argument("--save_every_min", type=float, default=30.0)
    t.add_argument("--eval_every_min", type=float, default=30.0)
    t.add_argument("--log_every", type=int, default=100)
    t.add_argument("--keep_ckpts", type=int, default=2)
    t.add_argument("--n_eval", type=int, default=8)
    t.add_argument("--max_steps", type=int, default=0, help="stop after this many steps in this session (0 = no limit)")
    t.add_argument("--max_bad_steps", type=int, default=50)
    t.add_argument("--seed", type=int, default=1234)

    s = sub.add_parser("synth", help="text -> wav")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--text", default="")
    s.add_argument("--text_file", default="")
    s.add_argument("--speaker", default="0", help="index, speaker ID, or 'all'")
    s.add_argument("--out", default="out.wav")
    s.add_argument("--noise_scale", type=float, default=0.667)
    s.add_argument("--noise_scale_w", type=float, default=0.8)
    s.add_argument("--length_scale", type=float, default=1.0, help=">1 slower, <1 faster")

    args = ap.parse_args(argv)
    if args.cmd == "download":
        from .data.download import download
        download(args)
    elif args.cmd == "prepare":
        from .data.prepare import prepare
        prepare(args)
    elif args.cmd == "train":
        from .train import train
        train(args)
    else:
        synth_cli(args)


if __name__ == "__main__":
    main()
