"""Shared helpers (ported from VITS commons.py) and weight-norm compatibility."""
import torch
import torch.nn.functional as F


try:  # torch >= 2.1
    from torch.nn.utils.parametrizations import weight_norm
    from torch.nn.utils import parametrize as _parametrize
    _NEW_WEIGHT_NORM = True
except ImportError:  # older torch
    from torch.nn.utils import weight_norm
    _NEW_WEIGHT_NORM = False


__all__ = ["LRELU_SLOPE", "fused_add_tanh_sigmoid_multiply", "generate_path", "get_padding", "rand_slice_segments",
           "remove_weight_norms", "sequence_mask", "slice_segments", "weight_norm"]

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def slice_segments(x, ids_str, segment_size):
    idx = ids_str.view(-1, 1, 1) + torch.arange(segment_size, device=x.device).view(1, 1, -1)
    return torch.gather(x, 2, idx.expand(x.size(0), x.size(1), segment_size))


def rand_slice_segments(x, x_lengths, segment_size):
    ids_str_max = (x_lengths - segment_size + 1).clamp(min=1)
    ids_str = (torch.rand(x.size(0), device=x.device) * ids_str_max).long()
    ids_str = ids_str.clamp(min=0, max=max(0, x.size(2) - segment_size))
    return slice_segments(x, ids_str, segment_size), ids_str


def generate_path(duration, mask):
    b, _, t_y, t_x = mask.shape
    cum = torch.cumsum(duration, -1).view(b * t_x)
    path = sequence_mask(cum, t_y).to(mask.dtype).view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def fused_add_tanh_sigmoid_multiply(a, b, n_channels):
    in_act = a + b
    return torch.tanh(in_act[:, :n_channels]) * torch.sigmoid(in_act[:, n_channels:])


def remove_weight_norms(model):
    """Fold weight-norm into plain weights (faster inference)."""
    for m in list(model.modules()):
        if _NEW_WEIGHT_NORM:
            if _parametrize.is_parametrized(m, "weight"):
                _parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        else:
            try:
                torch.nn.utils.remove_weight_norm(m)
            except (ValueError, AttributeError):
                pass
    return model
