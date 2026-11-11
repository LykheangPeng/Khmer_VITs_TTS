"""Load a trained generator and synthesize Khmer text."""
import re
from pathlib import Path

import numpy as np
import torch

from .models import build_generator, remove_weight_norms
from .text import Tokenizer, normalize_khmer
from .utils import log


def load_generator(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu", fold_weight_norm=True):
    """ckpt_path: G_latest.pt, a full ckpt_*.pt, or a run folder (uses G_latest.pt, else the newest ckpt)."""
    p = Path(ckpt_path)
    candidates = [p / "G_latest.pt"] + sorted(p.glob("ckpt_*.pt"), reverse=True) if p.is_dir() else [p]
    ck = None
    for c in candidates:
        try:
            ck = torch.load(c, map_location="cpu", weights_only=False)
            log(f"loaded {c} (step {ck.get('step', '?')})")
            break
        except Exception as e:
            log(f"could not load {c}: {type(e).__name__}")
    if ck is None:
        raise FileNotFoundError(f"No usable checkpoint in {ckpt_path} yet - the first one is saved "
                                f"--save_every_min (default 30) minutes after training starts.")
    hp, symbols, speakers = ck["hparams"], ck["symbols"], ck["speakers"]
    net_g = build_generator(hp, len(symbols), len(speakers))
    net_g.load_state_dict(ck["G"])
    net_g.eval()
    if fold_weight_norm:
        remove_weight_norms(net_g)
    return net_g.to(device), hp, Tokenizer(symbols), speakers


def split_sentences(text, max_chars=300):
    parts = [p.strip() for p in re.split(r"(?<=[។៕?!])\s*", text) if p.strip()]
    out = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 3 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


@torch.no_grad()
def synthesize(net_g, tokenizer, hp, text, speaker=0, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0,
               pause=0.25, warn=True):
    """Khmer text (any length) -> float32 waveform at hp['sample_rate']."""
    device = next(net_g.parameters()).device
    pieces = []
    for sent in split_sentences(normalize_khmer(text)):
        ids, unknown = tokenizer.encode(sent, hp["add_blank"], return_unknown=True)
        if unknown and warn:
            log(f"note: characters not in the training vocabulary were skipped: {sorted(set(unknown))}")
        if len(ids) < 3:
            continue
        x = torch.LongTensor(ids).unsqueeze(0).to(device)
        x_len = torch.LongTensor([x.size(1)]).to(device)
        sid = torch.LongTensor([int(speaker)]).to(device) if net_g.n_speakers > 1 else None
        audio = net_g.infer(x, x_len, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                            length_scale=length_scale)[0][0, 0].float().cpu().numpy()
        pieces += [audio, np.zeros(int(pause * hp["sample_rate"]), np.float32)]
    return np.concatenate(pieces[:-1]) if pieces else np.zeros(0, np.float32)
