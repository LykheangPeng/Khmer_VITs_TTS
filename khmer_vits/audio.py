"""Spectrograms computed on the GPU (same definitions as the official VITS code)."""
import numpy as np
import torch
import torch.nn.functional as F


_WINDOWS, _MEL_BASIS = {}, {}


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Slaney-style mel filterbank, identical to librosa.filters.mel defaults."""
    fmax = float(fmax) if fmax else sr / 2.0
    fftfreqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


def spectrogram(y, n_fft, hop, win):
    """y: [B, T] float32 -> linear magnitude [B, n_fft//2+1, T//hop]."""
    key = (str(y.device), win)
    if key not in _WINDOWS:
        _WINDOWS[key] = torch.hann_window(win, device=y.device)
    p = (n_fft - hop) // 2
    y = F.pad(y.unsqueeze(1), (p, p), mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=_WINDOWS[key], center=False,
                      normalized=False, onesided=True, return_complex=True)
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel(spec, hp):
    key = (str(spec.device), hp["sample_rate"], hp["n_fft"], hp["n_mels"], hp["fmin"], hp["fmax"])
    if key not in _MEL_BASIS:
        _MEL_BASIS[key] = torch.from_numpy(mel_filterbank(hp["sample_rate"], hp["n_fft"], hp["n_mels"],
                                                          hp["fmin"], hp["fmax"])).to(spec.device)
    return torch.log(torch.clamp(torch.matmul(_MEL_BASIS[key], spec), min=1e-5))


def mel_spectrogram(y, hp):
    return spec_to_mel(spectrogram(y, hp["n_fft"], hp["hop"], hp["win"]), hp)
