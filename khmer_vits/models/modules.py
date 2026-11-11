"""Building blocks: WaveNet, DDS-conv, HiFi-GAN ResBlock, normalizing-flow layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d

from .commons import LRELU_SLOPE, fused_add_tanh_sigmoid_multiply, get_padding, weight_norm
from .transforms import piecewise_rational_quadratic


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels, self.eps = channels, eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class DDSConv(nn.Module):
    """Dilated and depth-separable convolution."""

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep, self.convs_1x1 = nn.ModuleList(), nn.ModuleList()
        self.norms_1, self.norms_2 = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, groups=channels,
                                            dilation=dilation, padding=padding))
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = F.gelu(self.norms_1[i](y))
            y = self.convs_1x1[i](y)
            y = F.gelu(self.norms_2[i](y))
            x = x + self.drop(y)
        return x * x_mask


class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0.0):
        super().__init__()
        self.hidden_channels, self.n_layers = hidden_channels, n_layers
        self.in_layers, self.res_skip_layers = nn.ModuleList(), nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)) if gin_channels else None
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                                        dilation=dilation, padding=padding)))
            res_skip = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden_channels, res_skip, 1)))

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        h = self.hidden_channels
        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g[:, i * 2 * h:(i + 1) * 2 * h] if (g is not None and self.cond_layer is not None) else 0.0
            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, h))
            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip[:, :h]) * x_mask
                output = output + res_skip[:, h:]
            else:
                output = output + res_skip
        return output * x_mask


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                                        padding=get_padding(kernel_size, d))) for d in dilation])
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                                        padding=get_padding(kernel_size, 1))) for _ in dilation])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            return y, torch.sum(-y, [1, 2])
        return torch.exp(x) * x_mask


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = (self.m + torch.exp(self.logs) * x) * x_mask
            return y, torch.sum(self.logs * x_mask, [1, 2])
        return (x - self.m) * torch.exp(-self.logs) * x_mask


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0,
                 gin_channels=0, mean_only=False):
        super().__init__()
        self.half_channels, self.mean_only = channels // 2, mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m, logs = stats, torch.zeros_like(stats)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            return torch.cat([x0, x1], 1), torch.sum(logs, [1, 2])
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class ConvFlow(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
        super().__init__()
        self.filter_channels, self.num_bins, self.tail_bound = filter_channels, num_bins, tail_bound
        self.half_channels = in_channels // 2
        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.convs(self.pre(x0), x_mask, g=g)
        h = self.proj(h) * x_mask
        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)
        w = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
        hh = h[..., self.num_bins:2 * self.num_bins] / math.sqrt(self.filter_channels)
        d = h[..., 2 * self.num_bins:]
        x1, logabsdet = piecewise_rational_quadratic(x1, w, hh, d, inverse=reverse, tail_bound=self.tail_bound)
        x = torch.cat([x0, x1], 1) * x_mask
        if not reverse:
            return x, torch.sum(logabsdet * x_mask, [1, 2])
        return x
