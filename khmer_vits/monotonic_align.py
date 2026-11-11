"""Monotonic alignment search (Glow-TTS / VITS). numba-parallel, with a numpy fallback."""
import numpy as np
import torch


try:
    import numba

    @numba.njit(nogil=True)
    def _mas_one(path, value, t_y, t_x):
        max_neg_val = -1e9
        index = t_x - 1
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                index = index - 1

    @numba.njit(nogil=True, parallel=True)
    def _mas_batch(paths, values, t_ys, t_xs):
        for i in numba.prange(paths.shape[0]):
            _mas_one(paths[i], values[i], t_ys[i], t_xs[i])

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


def _mas_one_numpy(path, value, t_y, t_x):
    neg = -1e9
    q = np.full((t_y, t_x), neg)
    v = value[:t_y, :t_x].astype(np.float64)
    xs = np.arange(t_x)
    for y in range(t_y):
        lo, hi = max(0, t_x + y - t_y), min(t_x, y + 1)
        if y == 0:
            stay = np.full(t_x, neg)
            move = np.full(t_x, neg)
            move[0] = 0.0
        else:
            stay = np.where(xs == y, neg, q[y - 1])
            move = np.concatenate([[neg], q[y - 1, :-1]])
        row = v[y] + np.maximum(stay, move)
        q[y, lo:hi] = row[lo:hi]
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or q[y - 1, index] < q[y - 1, index - 1]):
            index -= 1


def maximum_path(neg_cent, mask):
    """neg_cent, mask: [B, T_spec, T_text] -> hard monotonic alignment of the same shape."""
    device, dtype = neg_cent.device, neg_cent.dtype
    values = np.ascontiguousarray(neg_cent.detach().float().cpu().numpy())
    paths = np.zeros(values.shape, dtype=np.uint8)
    t_ys = mask.sum(1)[:, 0].long().cpu().numpy().astype(np.int64)
    t_xs = mask.sum(2)[:, 0].long().cpu().numpy().astype(np.int64)
    if HAVE_NUMBA:
        _mas_batch(paths, values, t_ys, t_xs)
    else:
        for i in range(values.shape[0]):
            _mas_one_numpy(paths[i], values[i], int(t_ys[i]), int(t_xs[i]))
    return torch.from_numpy(paths).to(device).to(dtype)
