"""Background checkpoint writing with atomic renames and trash-safe deletion."""
import os
import threading
from pathlib import Path

import torch

from .utils import log


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def discard(path):
    """Delete a file without leaving a full-size copy behind on synced/FUSE storage (e.g. a mounted
    Google Drive moves deletions to a trash that still counts against quota): truncate first, then delete."""
    try:
        with open(path, "wb"):
            pass
        os.remove(path)
    except FileNotFoundError:
        pass


class CheckpointSaver:
    """Writes checkpoints in a background thread (the GPU keeps training). Atomic rename on completion."""

    def __init__(self, run_dir, keep=2):
        self.run_dir, self.keep, self.thread = Path(run_dir), keep, None

    def save(self, make_state, step, blocking=False):
        self.wait()  # finish the previous write first -> at most one CPU snapshot in memory
        full_state = make_state()
        g_state = {k: full_state[k] for k in ("G", "hparams", "symbols", "speakers", "speaker_info", "step")}

        def work():
            try:
                for name, obj in ((f"ckpt_{step:08d}.pt", full_state), ("G_latest.pt", g_state)):
                    tmp, final = self.run_dir / f".{name}.tmp", self.run_dir / name
                    torch.save(obj, tmp)
                    if final.exists():
                        discard(final)
                    os.replace(tmp, final)
                for old in sorted(self.run_dir.glob("ckpt_*.pt"))[:-self.keep]:
                    discard(old)
                log(f"checkpoint saved (step {step}) -> {self.run_dir}")
            except Exception as e:  # never kill training because storage hiccupped
                log(f"WARNING: checkpoint save failed: {type(e).__name__}: {e}")

        self.thread = threading.Thread(target=work, daemon=False)
        self.thread.start()
        if blocking:
            self.wait()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None


def load_latest_checkpoint(run_dir):
    for p in sorted(Path(run_dir).glob("ckpt_*.pt"), reverse=True):
        try:
            return p, torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            log(f"WARNING: could not read {p.name} ({type(e).__name__}); trying an older checkpoint")
    return None, None
