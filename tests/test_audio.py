import numpy as np
import pytest
import torch

from khmer_vits.audio import mel_filterbank, spectrogram


def test_mel_filterbank_matches_librosa():
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0.0, fmax=None)
    assert np.allclose(ref, mel_filterbank(22050, 1024, 80, 0.0, None), atol=1e-6)


def test_spectrogram_frames_equal_samples_over_hop():
    y = torch.randn(2, 96 * 256)
    assert spectrogram(y, 1024, 256, 1024).shape == (2, 513, 96)
