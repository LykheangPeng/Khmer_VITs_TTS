import json
from pathlib import Path

import pytest

from khmer_vits.config import DEFAULT_CONFIG, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_base_json_equals_builtin_defaults():
    assert load_config(ROOT / "configs" / "base.json") == DEFAULT_CONFIG


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"learning_rate_typo": 1}))
    with pytest.raises(ValueError):
        load_config(p)
