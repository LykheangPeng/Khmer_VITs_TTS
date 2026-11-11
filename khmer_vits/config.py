"""Model / audio / optimisation hyper-parameters.

DEFAULT_CONFIG is the VITS "base" configuration (configs/vctk_base.json of the official repo).
configs/base.json mirrors it; pass --config to override any subset of keys.
"""
import copy
import json

DEFAULT_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop": 256,
    "win": 1024,
    "n_mels": 80,
    "fmin": 0.0,
    "fmax": None,
    "add_blank": True,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock_kernel_sizes": [
        3,
        7,
        11
    ],
    "resblock_dilation_sizes": [
        [
            1,
            3,
            5
        ],
        [
            1,
            3,
            5
        ],
        [
            1,
            3,
            5
        ]
    ],
    "upsample_rates": [
        8,
        8,
        2,
        2
    ],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [
        16,
        16,
        4,
        4
    ],
    "gin_channels": 256,
    "segment_size": 8192,
    "lr": 0.0002,
    "betas": [
        0.8,
        0.99
    ],
    "eps": 1e-09,
    "lr_decay": 0.999875,
    "c_mel": 45.0,
    "c_kl": 1.0
}


def load_config(path=None):
    """Defaults, overridden by the keys present in a JSON file (if given)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        unknown = set(user) - set(cfg)
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        cfg.update(user)
    assert cfg["hop"] == _prod(cfg["upsample_rates"]), "product of upsample_rates must equal hop"
    return cfg


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out
