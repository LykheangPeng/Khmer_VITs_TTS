"""Memory-mapped dataset, padding collate, length-bucketed batch sampler."""
import signal

import numpy as np
import torch
from torch.utils.data import Dataset


class TTSDataset(Dataset):
    def __init__(self, items, audio_path, tokenizer, spk_index, add_blank=True):
        self.audio_path = str(audio_path)
        self.offsets = np.array([e["offset"] for e in items], dtype=np.int64)
        self.lengths = np.array([e["length"] for e in items], dtype=np.int64)
        self.sids = np.array([spk_index[e["speaker"]] for e in items], dtype=np.int64)
        self.tokens = [np.asarray(tokenizer.encode(e["text"], add_blank), dtype=np.int64) for e in items]
        self._mm = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        if self._mm is None:  # opened lazily inside each worker
            self._mm = np.memmap(self.audio_path, dtype=np.int16, mode="r")
        o, n = self.offsets[i], self.lengths[i]
        return self.tokens[i], np.array(self._mm[o:o + n]), int(self.sids[i])


class Collate:
    """Pads to fixed multiples so cuDNN sees few distinct shapes. Audio stays int16 until the GPU."""

    def __init__(self, hop, min_frames, frame_multiple=32, text_multiple=16):
        self.hop, self.min_frames = hop, min_frames
        self.frame_multiple, self.text_multiple = frame_multiple, text_multiple

    def __call__(self, batch):
        def up(n, m):
            return ((n + m - 1) // m) * m
        n = len(batch)
        tl = [len(b[0]) for b in batch]
        wl = [len(b[1]) for b in batch]
        t_text = up(max(tl), self.text_multiple)
        t_frames = max(up(max(wl) // self.hop, self.frame_multiple), self.min_frames)
        text = torch.zeros(n, t_text, dtype=torch.long)
        wav = torch.zeros(n, t_frames * self.hop, dtype=torch.int16)
        for i, (tok, w, _) in enumerate(batch):
            text[i, :len(tok)] = torch.from_numpy(tok)
            wav[i, :len(w)] = torch.from_numpy(w)
        return (text, torch.tensor(tl, dtype=torch.long), wav, torch.tensor(wl, dtype=torch.long),
                torch.tensor([b[2] for b in batch], dtype=torch.long))


class SortishBatchSampler:
    """Shuffle, sort by length inside large chunks, cut into batches, shuffle the batches."""

    def __init__(self, lengths, batch_size, seed=0, bucket_mult=50):
        self.lengths, self.batch_size = np.asarray(lengths), batch_size
        self.seed, self.bucket_mult, self.epoch = seed, bucket_mult, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        idx = rng.permutation(len(self.lengths))
        chunk = self.batch_size * self.bucket_mult
        batches = []
        for s in range(0, len(idx), chunk):
            c = idx[s:s + chunk]
            c = c[np.argsort(self.lengths[c], kind="stable")]
            for b in range(0, len(c) - self.batch_size + 1, self.batch_size):
                batches.append(c[b:b + self.batch_size].tolist())
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        chunk = self.batch_size * self.bucket_mult
        full, rem = divmod(len(self.lengths), chunk)
        return full * (chunk // self.batch_size) + rem // self.batch_size


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the main process
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)
