import numpy as np
import pytest
import torch

from khmer_vits import monotonic_align as ma


def _random_problem(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((4, 60, 25)).astype(np.float32), np.array([60, 50, 40, 30]), np.array([25, 20, 25, 12])


@pytest.mark.skipif(not ma.HAVE_NUMBA, reason="numba not installed")
def test_numba_equals_numpy():
    vals, t_ys, t_xs = _random_problem()
    p1 = np.zeros(vals.shape, np.uint8)
    p2 = np.zeros(vals.shape, np.uint8)
    ma._mas_batch(p1, vals.copy(), t_ys, t_xs)
    for i in range(len(vals)):
        ma._mas_one_numpy(p2[i], vals[i].copy(), int(t_ys[i]), int(t_xs[i]))
    assert (p1 == p2).all()


def test_paths_are_valid_monotonic_alignments():
    vals, t_ys, t_xs = _random_problem(1)
    mask = torch.zeros(vals.shape)
    for i, (ty, tx) in enumerate(zip(t_ys, t_xs)):
        mask[i, :ty, :tx] = 1
    path = ma.maximum_path(torch.from_numpy(vals), mask).numpy()
    for i, (ty, tx) in enumerate(zip(t_ys, t_xs)):
        p = path[i, :ty, :tx]
        assert (p.sum(1) == 1).all() and (p.sum(0) >= 1).all()   # every frame -> one token, every token used
        idx = p.argmax(1)
        assert ((np.diff(idx) >= 0) & (np.diff(idx) <= 1)).all()  # monotonic, no skips
        assert path[i].sum() == ty                                 # nothing outside the mask
