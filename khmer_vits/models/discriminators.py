"""HiFi-GAN multi-period discriminator + MelGAN multi-scale discriminator."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d

from .commons import LRELU_SLOPE, get_padding, weight_norm


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        p = (get_padding(kernel_size, 1), 0)
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=p)),
            weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=p)),
            weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=p)),
            weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=p)),
            weight_norm(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=p)),
        ])
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(1, 16, 15, 1, padding=7)),
            weight_norm(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            weight_norm(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            weight_norm(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            weight_norm(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            weight_norm(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = weight_norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS()] + [DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        """Real and generated audio go through each discriminator in ONE batched pass."""
        b = y.size(0)
        x = torch.cat([y, y_hat.to(y.dtype)], 0)
        d_real, d_fake, f_real, f_fake = [], [], [], []
        for d in self.discriminators:
            out, fmap = d(x)
            d_real.append(out[:b])
            d_fake.append(out[b:])
            f_real.append([f[:b] for f in fmap])
            f_fake.append([f[b:] for f in fmap])
        return d_real, d_fake, f_real, f_fake
