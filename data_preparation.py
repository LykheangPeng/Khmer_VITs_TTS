

import os
import json
import argparse
import shutil
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Audio, DatasetDict



# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path",    required=True,  help="Path to Khmer_travel_dataset.csv")
    p.add_argument("--audio_dir",   required=True,  help="Directory containing .wav audio files")
    p.add_argument("--output_dir",  default="./dataset", help="Where to save the HF dataset")
    p.add_argument("--speaker",     default=None,   help="Filter to a single speaker ID (optional)")
    p.add_argument("--min_duration",type=float, default=0.5,  help="Min clip duration in seconds")
    p.add_argument("--max_duration",type=float, default=15.0, help="Max clip duration in seconds")
    p.add_argument("--target_sr",   type=int,   default=16000, help="Target sample rate (MMS = 16000)")
    p.add_argument("--val_ratio",   type=float, default=0.1,  help="Fraction of data for validation")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

# Helpers
AUDIO_EXTENSIONS = [".wav", ".WAV", ".mp3", ".flac", ".ogg"]

def find_audio_file(audio_dir: str, file_name: str) -> str | None:
    """Search for the audio file by name (without extension) in audio_dir."""
    for ext in AUDIO_EXTENSIONS:
        path = os.path.join(audio_dir, file_name + ext)
        if os.path.exists(path):
            return path
        # Also try subdirectories one level deep
        for subdir in Path(audio_dir).iterdir():
            if subdir.is_dir():
                path = os.path.join(str(subdir), file_name + ext)
                if os.path.exists(path):
                    return path
    return None


def get_duration(path: str) -> float:
    """Return audio duration in seconds."""
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        try:
            y, sr = librosa.load(path, sr=None, duration=1)
            duration = librosa.get_duration(filename=path)
            return duration
        except Exception:
            return 0.0


def resample_if_needed(path: str, target_sr: int, out_dir: str) -> str:
    """Resample audio to target_sr if different. Returns path to (possibly new) file."""
    info = sf.info(path)
    if info.samplerate == target_sr:
        return path

    # Save resampled version to out_dir
    os.makedirs(out_dir, exist_ok=True)
    new_path = os.path.join(out_dir, os.path.basename(path))
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    sf.write(new_path, y, target_sr)
    return new_path

def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"\n{'='*55}")
    print(f"  Khmer VITS Dataset Preparation")
    print(f"{'='*55}")
    print(f"  CSV:        {args.csv_path}")
    print(f"  Audio dir:  {args.audio_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Speaker:    {args.speaker or 'ALL'}")
    print(f"  Duration:   {args.min_duration}s – {args.max_duration}s")
    print(f"  Target SR:  {args.target_sr} Hz")
    print()

    # Load CSV 
    df = pd.read_csv(args.csv_path, encoding="utf-8-sig")
    # Strip column name whitespace
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Columns: {list(df.columns)}")

    # Rename to standard names
    df = df.rename(columns={
        "Speaker ID": "speaker_id",
        "File Name":  "file_name",
        "Sentences":  "text",
    })

    # Optional speaker filter 
    if args.speaker:
        before = len(df)
        df = df[df["speaker_id"] == args.speaker].reset_index(drop=True)
        print(f"Filtered to speaker '{args.speaker}': {before} → {len(df)} rows")

    # Build speaker map 
    speakers = sorted(df["speaker_id"].unique().tolist())
    speaker_map = {s: i for i, s in enumerate(speakers)}
    print(f"\nSpeakers ({len(speakers)}): {speaker_map}")

    # Resample directory 
    resampled_dir = os.path.join(args.output_dir, "resampled_audio")

    # Process each row 
    records = []
    skipped_missing  = 0
    skipped_duration = 0

    print(f"\nSearching for audio files in: {args.audio_dir}")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        file_name = str(row["file_name"]).strip()
        text      = str(row["text"]).strip()
        speaker   = str(row["speaker_id"]).strip()

        # Skip empty text
        if not text or text == "nan":
            skipped_missing += 1
            continue

        # Find audio
        audio_path = find_audio_file(args.audio_dir, file_name)
        if audio_path is None:
            skipped_missing += 1
            continue

        # Check duration
        duration = get_duration(audio_path)
        if duration < args.min_duration or duration > args.max_duration:
            skipped_duration += 1
            continue

        # Resample if needed
        final_path = resample_if_needed(audio_path, args.target_sr, resampled_dir)

        records.append({
            "audio":      final_path,
            "text":       text,
            "speaker_id": speaker,
            "speaker_idx": speaker_map[speaker],
            "duration":   round(duration, 3),
            "file_name":  file_name,
        })

    print(f"\n{'─'*40}")
    print(f"  Total kept:           {len(records)}")
    print(f"  Skipped (missing):    {skipped_missing}")
    print(f"  Skipped (duration):   {skipped_duration}")

    if len(records) == 0:
        raise RuntimeError(
            "No valid records found! Check that --audio_dir contains files "
            "matching the 'File Name' column in your CSV."
        )

    # Per-speaker stats 
    print(f"\nPer-speaker breakdown:")
    per_speaker = {}
    for r in records:
        s = r["speaker_id"]
        per_speaker.setdefault(s, {"count": 0, "duration": 0.0})
        per_speaker[s]["count"]    += 1
        per_speaker[s]["duration"] += r["duration"]

    total_duration = sum(v["duration"] for v in per_speaker.values())
    for s, v in sorted(per_speaker.items()):
        mins = v["duration"] / 60
        pct  = 100 * v["duration"] / total_duration
        print(f"  {s:20s}: {v['count']:4d} clips | {mins:6.1f} min ({pct:.1f}%)")
    print(f"  {'TOTAL':20s}: {len(records):4d} clips | {total_duration/60:.1f} min")

    # Warn about small speakers 
    for s, v in per_speaker.items():
        mins = v["duration"] / 60
        if mins < 20:
            print(f"\n    Speaker '{s}' has only {mins:.1f} min — consider dropping them")

    # Train / val split 
    import random
    random.seed(args.seed)
    random.shuffle(records)

    n_val   = max(1, int(len(records) * args.val_ratio))
    n_train = len(records) - n_val
    train_records = records[:n_train]
    val_records   = records[n_train:]
    print(f"\nSplit: {n_train} train / {n_val} val")

    # Save as HuggingFace dataset 
    os.makedirs(args.output_dir, exist_ok=True)

    def make_hf_dataset(recs):
        ds = Dataset.from_dict({
            "audio":       [r["audio"]       for r in recs],
            "text":        [r["text"]        for r in recs],
            "speaker_id":  [r["speaker_id"]  for r in recs],
            "speaker_idx": [r["speaker_idx"] for r in recs],
            "duration":    [r["duration"]    for r in recs],
        })
        ds = ds.cast_column("audio", Audio(sampling_rate=args.target_sr))
        return ds

    dataset_dict = DatasetDict({
        "train":      make_hf_dataset(train_records),
        "validation": make_hf_dataset(val_records),
    })

    save_path = os.path.join(args.output_dir, "hf_dataset")
    dataset_dict.save_to_disk(save_path)
    print(f"\n HuggingFace dataset saved to: {save_path}")

    # ── Save speaker map ─
    speaker_map_path = os.path.join(args.output_dir, "speaker_map.json")
    with open(speaker_map_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, ensure_ascii=False, indent=2)
    print(f" Speaker map saved to: {speaker_map_path}")

    # ── Save metadata CSV 
    meta_df = pd.DataFrame(records)
    meta_path = os.path.join(args.output_dir, "metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f" Metadata saved to: {meta_path}")

    print(f"\n{'='*55}")
    print(f"  Dataset preparation complete!")
    print(f"  Next step: python 02_convert_checkpoint.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
