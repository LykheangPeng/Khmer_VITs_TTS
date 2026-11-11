"""End-to-end: fake dataset -> prepare -> train -> resume -> synthesize, through the CLI (CPU, ~1-2 min)."""
import json

import pytest
import soundfile as sf

from fake_data import make_fake_dataset
from khmer_vits.cli import main


@pytest.mark.slow
def test_full_pipeline(tmp_path):
    raw, prep, run = tmp_path / "raw", tmp_path / "prep", tmp_path / "run"
    make_fake_dataset(raw)

    main(["prepare", "--src", str(raw), "--out", str(prep), "--workers", "2"])
    man = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
    assert len(man["speakers"]) == 3 and (prep / "DONE").exists()
    stats = man["stats"]
    assert stats["drop_latin_letters"] == 3 and stats["drop_ambiguous_number(decimal/time/date)"] == 3
    assert stats["drop_audio_silent"] == 3 and stats["drop_speaking_rate_outlier"] == 3

    common = ["--data", str(prep), "--run_dir", str(run), "--batch_size", "4", "--workers", "0",
              "--eval_every_min", "999", "--save_every_min", "999", "--hours", "1"]
    main(["train", *common, "--max_steps", "2"])
    assert (run / "G_latest.pt").exists() and (run / "ckpt_00000002.pt").exists()
    main(["train", *common, "--max_steps", "1"])                 # resumes from step 2
    assert (run / "ckpt_00000003.pt").exists()

    out = tmp_path / "out.wav"
    main(["synth", "--ckpt", str(run / "G_latest.pt"), "--text", "មុខម្ហូបខ្មែរ ឆ្ងាញ់ណាស់។",
          "--speaker", "m-adt1-0002", "--out", str(out)])
    audio, sr = sf.read(out)
    assert sr == 22050 and len(audio) > 0 and abs(audio).max() <= 1.0
