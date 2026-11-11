"""Transformer text encoder with relative positional multi-head attention."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LayerNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads, self.window_size = n_heads, window_size
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        rel_stddev = self.k_channels ** -0.5
        self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev)
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        return self.conv_o(self.attention(q, k, v, mask=attn_mask))

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        query = query / math.sqrt(self.k_channels)
        scores = torch.matmul(query, key.transpose(-2, -1))
        key_rel = self._get_relative_embeddings(self.emb_rel_k, t_s)
        rel_logits = torch.matmul(query, key_rel.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._relative_position_to_absolute_position(rel_logits)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)
        value_rel = self._get_relative_embeddings(self.emb_rel_v, t_s)
        output = output + torch.matmul(self._absolute_position_to_relative_position(p_attn), value_rel.unsqueeze(0))
        return output.transpose(2, 3).contiguous().view(b, d, t_t)

    def _get_relative_embeddings(self, rel, length):
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            rel = F.pad(rel, (0, 0, pad_length, pad_length))
        return rel[:, start:end]

    @staticmethod
    def _relative_position_to_absolute_position(x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, 1))
        x_flat = F.pad(x.view(batch, heads, length * 2 * length), (0, length - 1))
        return x_flat.view(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1:]

    @staticmethod
    def _absolute_position_to_relative_position(x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, length - 1))
        x_flat = F.pad(x.view(batch, heads, length ** 2 + length * (length - 1)), (length, 0))
        return x_flat.view(batch, heads, length, 2 * length)[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def _pad(self, x):
        if self.kernel_size == 1:
            return x
        return F.pad(x, ((self.kernel_size - 1) // 2, self.kernel_size // 2))

    def forward(self, x, x_mask):
        x = torch.relu(self.conv_1(self._pad(x * x_mask)))
        x = self.conv_2(self._pad(self.drop(x) * x_mask))
        return x * x_mask


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=4):
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers, self.norm_layers_1 = nn.ModuleList(), nn.ModuleList()
        self.ffn_layers, self.norm_layers_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout, window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.drop(self.attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_1[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)
        return x * x_mask
