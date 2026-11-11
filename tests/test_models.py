import torch
import torch.nn.functional as F

from khmer_vits.audio import mel_spectrogram, spec_to_mel, spectrogram
from khmer_vits.config import DEFAULT_CONFIG
from khmer_vits.losses import feature_loss, generator_loss, kl_loss
from khmer_vits.models import MultiPeriodDiscriminator, build_generator, remove_weight_norms, slice_segments
from khmer_vits.models.attentions import MultiHeadAttention
from khmer_vits.models.modules import ConvFlow, ResidualCouplingLayer
from khmer_vits.models.synthesizer import ResidualCouplingBlock

HP = dict(DEFAULT_CONFIG)


def test_relative_position_helpers_match_naive_reference():
    l = 7
    x = torch.randn(2, 2, l, 2 * l - 1)
    ref = torch.zeros(2, 2, l, l)
    for i in range(l):
        for j in range(l):
            ref[:, :, i, j] = x[:, :, i, j - i + l - 1]
    assert torch.allclose(MultiHeadAttention._relative_position_to_absolute_position(x), ref)
    y = torch.randn(2, 2, l, l)
    ref2 = torch.zeros(2, 2, l, 2 * l - 1)
    for i in range(l):
        for k in range(2 * l - 1):
            j = k - (l - 1) + i
            if 0 <= j < l:
                ref2[:, :, i, k] = y[:, :, i, j]
    assert torch.allclose(MultiHeadAttention._absolute_position_to_relative_position(y), ref2)


def test_flows_are_invertible():
    mask = torch.ones(2, 1, 50)
    mask[1, :, 40:] = 0
    flow = ResidualCouplingBlock(192, 192, 5, 1, 4, gin_channels=256)
    for m in flow.modules():
        if isinstance(m, ResidualCouplingLayer):
            torch.nn.init.normal_(m.post.weight, std=0.02)
    z, g = torch.randn(2, 192, 50) * mask, torch.randn(2, 256, 1)
    assert torch.allclose(flow(flow(z, mask, g=g), mask, g=g, reverse=True), z, atol=1e-4)
    cf = ConvFlow(2, 192, 3, n_layers=3).double()
    torch.nn.init.normal_(cf.proj.weight, std=0.5)
    m64 = mask.double()
    x, h = torch.randn(2, 2, 50).double() * 2 * m64, torch.randn(2, 192, 50).double()
    y, _ = cf(x, m64, g=h)
    assert torch.allclose(cf(y, m64, g=h, reverse=True), x, atol=1e-9)


def test_parameter_counts_match_vits_base():
    net_g = build_generator(HP, n_vocab=82, n_speakers=10)
    net_d = MultiPeriodDiscriminator()
    n_g = sum(p.numel() for p in net_g.parameters()) / 1e6
    n_d = sum(p.numel() for p in net_d.parameters()) / 1e6
    assert 39.5 < n_g < 39.9 and 46.6 < n_d < 46.9


def _forward_backward(amp):
    net_g = build_generator(HP, n_vocab=90, n_speakers=8)
    net_d = MultiPeriodDiscriminator()
    net_g.amp_dtype = amp
    text, text_len = torch.randint(1, 90, (2, 48)), torch.tensor([48, 30])
    wav, wav_len = torch.randn(2, 96 * 256) * 0.1, torch.tensor([96 * 256, 64 * 256])
    spec = spectrogram(wav, 1024, 256, 1024)
    y_hat, l_length, attn, ids, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(
        text, text_len, spec, wav_len // 256, torch.tensor([0, 5]))
    assert y_hat.shape == (2, 1, 8192) and attn.shape == (2, 1, 96, 48)
    y = slice_segments(wav.unsqueeze(1), ids * 256, 8192)
    ctx = torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp is not None)
    with ctx:
        _, d_g, f_r, f_g = net_d(y, y_hat)
    loss = (generator_loss(d_g) + feature_loss(f_r, f_g) + l_length.sum()
            + F.l1_loss(spec_to_mel(slice_segments(spec, ids, 32), HP), mel_spectrogram(y_hat.squeeze(1).float(), HP)) * 45
            + kl_loss(z_p, logs_q, m_p, logs_p, z_mask))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net_g.parameters())


def test_forward_backward_fp32():
    _forward_backward(None)


def test_forward_backward_bf16_decoder():
    _forward_backward(torch.bfloat16)


def test_inference_and_weight_norm_folding():
    net_g = build_generator(HP, n_vocab=90, n_speakers=8).eval()
    x, xl, sid = torch.randint(1, 90, (1, 31)), torch.tensor([31]), torch.tensor([3])
    torch.manual_seed(1)
    o1 = net_g.infer(x, xl, sid=sid)[0]
    remove_weight_norms(net_g)
    torch.manual_seed(1)
    o2 = net_g.infer(x, xl, sid=sid)[0]
    assert o1.shape == o2.shape and o1.shape[-1] % 256 == 0
    assert torch.allclose(o1, o2, atol=1e-4)
