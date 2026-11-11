"""VITS generator: text encoder, posterior encoder, flow, duration predictor, HiFi-GAN decoder."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d

from ..monotonic_align import maximum_path
from .attentions import Encoder
from .commons import LRELU_SLOPE, generate_path, rand_slice_segments, sequence_mask, weight_norm
from .modules import (WN, ConvFlow, DDSConv, ElementwiseAffine, Flip, Log, ResBlock1,
                      ResidualCouplingLayer)


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # as in the official code
        self.log_flow = Log()
        self.flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.cond = nn.Conv1d(gin_channels, filter_channels, 1) if gin_channels else None

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = self.pre(torch.detach(x))
        if g is not None and self.cond is not None:
            x = x + self.cond(torch.detach(g))
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask
        if not reverse:
            logdet_tot_q = 0
            h_w = self.post_proj(self.post_convs(self.post_pre(w), x_mask)) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2), device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q = logdet_tot_q + logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q = logdet_tot_q + torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q
            z0, logdet_tot = self.log_flow(z0, x_mask)
            z = torch.cat([z0, z1], 1)
            for flow in self.flows:
                z, logdet = flow(z, x_mask, g=x, reverse=False)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
        z = torch.randn(x.size(0), 2, x.size(2), device=x.device, dtype=x.dtype) * noise_scale
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)
        z0, _ = torch.split(z, [1, 1], 1)
        return z0  # log-duration


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.out_channels, self.hidden_channels = out_channels, hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = x.transpose(1, -1)
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                                    gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=False)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.enc(self.pre(x) * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(nn.Module):
    """HiFi-GAN V1 decoder."""

    def __init__(self, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels, self.num_upsamples = len(resblock_kernel_sizes), len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                                        upsample_initial_channel // (2 ** (i + 1)),
                                                        k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1) if gin_channels else None

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


class SynthesizerTrn(nn.Module):
    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, gin_channels=0):
        super().__init__()
        self.segment_size, self.n_speakers = segment_size, n_speakers
        self.amp_dtype = None  # set to torch.bfloat16 during training to run the decoder in bf16
        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers,
                                 kernel_size, p_dropout)
        self.dec = Generator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                             upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
        self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def _speaker(self, sid):
        return self.emb_g(sid).unsqueeze(-1) if (self.n_speakers > 1 and sid is not None) else None

    def _decode(self, z, g):
        if self.amp_dtype is None:
            return self.dec(z, g=g)
        with torch.autocast(device_type=z.device.type, dtype=self.amp_dtype):
            return self.dec(z, g=g)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._speaker(sid)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        with torch.no_grad():  # negative cross-entropy for monotonic alignment search
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
        w = attn.sum(2)
        l_length = self.dp(x, x_mask, w, g=g) / torch.sum(x_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self._decode(z_slice, g)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.no_grad()
    def infer(self, x, x_lengths, sid=None, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._speaker(sid)
        logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask * length_scale)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self._decode((z * y_mask)[:, :, :max_len], g)
        return o, attn, y_mask


def build_generator(hp, n_vocab, n_speakers):
    return SynthesizerTrn(
        n_vocab=n_vocab, spec_channels=hp["n_fft"] // 2 + 1, segment_size=hp["segment_size"] // hp["hop"],
        inter_channels=hp["inter_channels"], hidden_channels=hp["hidden_channels"],
        filter_channels=hp["filter_channels"], n_heads=hp["n_heads"], n_layers=hp["n_layers"],
        kernel_size=hp["kernel_size"], p_dropout=hp["p_dropout"],
        resblock_kernel_sizes=hp["resblock_kernel_sizes"], resblock_dilation_sizes=hp["resblock_dilation_sizes"],
        upsample_rates=hp["upsample_rates"], upsample_initial_channel=hp["upsample_initial_channel"],
        upsample_kernel_sizes=hp["upsample_kernel_sizes"], n_speakers=n_speakers,
        gin_channels=hp["gin_channels"] if n_speakers > 1 else 0)
