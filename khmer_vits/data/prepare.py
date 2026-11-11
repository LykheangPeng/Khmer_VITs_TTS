"""Raw dataset (metadata.csv + data/**/*.wav) -> audio_int16.bin + manifest.json."""
import csv
import json
import math
import multiprocessing as mp
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import load_config
from ..text import has_ambiguous_number, normalize_khmer
from ..utils import fmt_time, log


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}


TEXT_COLS = ("sentences", "sentence", "text", "transcript", "transcription", "normalizedtext",
             "utterance", "khmer", "label")


ID_COLS = ("sentenceid", "utteranceid", "uttid", "audioid", "fileid", "filename", "file",
           "path", "audio", "wav", "id")


SPK_COLS = ("speakerid", "speaker", "spkid", "spk", "clientid", "voiceid", "speakername")


def _norm_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _pick_col(cols, names):
    norm = {_norm_key(c): c for c in cols if c is not None}
    for n in names:
        if n in norm:
            return norm[n]
    return None


def _read_table(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        delim = max([",", "\t", ";", "|"], key=head.count)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _find_metadata(root):
    best, best_score = None, -1
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in (".csv", ".tsv") or not p.is_file():
            continue
        try:
            cols, rows = _read_table(p)
        except Exception:
            continue
        if not rows or _pick_col(cols, TEXT_COLS) is None:
            continue
        score = len(rows) + (10 ** 9 if p.name.lower() == "metadata.csv" else 0)
        if score > best_score:
            best, best_score = (p, cols, rows), score
    return best


def _speaker_genders(root):
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in (".csv", ".tsv") and "speaker" in p.name.lower():
            try:
                cols, rows = _read_table(p)
            except Exception:
                continue
            sc = _pick_col(cols, SPK_COLS)
            gc = next((c for c in cols if c and ("gender" in c.lower() or c.lower() == "sex")), None)
            if sc and gc:
                return {r[sc].strip(): r[gc].strip() for r in rows if r.get(sc)}
    return {}


def collect_items(src):
    found = _find_metadata(src)
    if found is None:
        raise SystemExit(f"No metadata CSV with a text column found under {src}")
    meta_path, cols, rows = found
    tcol, icol, scol = _pick_col(cols, TEXT_COLS), _pick_col(cols, ID_COLS), _pick_col(cols, SPK_COLS)
    if icol is None:  # e.g. 'Sentence ID' but with unusual spelling
        icol = next((c for c in cols if c and "id" in c.lower() and "speaker" not in c.lower()
                     and "paragraph" not in c.lower()), None)
    log(f"metadata: {meta_path}  ({len(rows)} rows) | text column = {tcol!r}, "
        f"id column = {icol!r}, speaker column = {scol!r}")

    by_stem, by_sent = {}, defaultdict(list)
    for p in Path(src).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            by_stem.setdefault(p.stem, p)
            if "_khm_" in p.stem:
                spk, sent = p.stem.split("_khm_", 1)
                by_sent[sent].append((spk, p))
    log(f"found {len(by_stem)} audio files")

    items, missing, seen = [], [], set()
    for r in rows:
        text = (r.get(tcol) or "").strip()
        sent = (r.get(icol) or "").strip() if icol else ""
        spk = (r.get(scol) or "").strip() if scol else ""
        p = None
        if spk and sent:  # documented naming: {speaker_id}_khm_{sentence_id}.wav
            p = by_stem.get(f"{spk}_khm_{sent}")
        if p is None and sent:
            p = by_stem.get(Path(sent).stem)
        if p is None and sent:
            cands = by_sent.get(sent, [])
            same = [c for c in cands if spk and c[0].lower() == spk.lower()]
            if len(same) == 1:
                p = same[0][1]
            elif len(cands) == 1:
                p = cands[0][1]
        if p is None:
            missing.append(sent or text[:40])
            continue
        if str(p) in seen:
            continue
        seen.add(str(p))
        if not spk:
            spk = p.stem.split("_khm_")[0] if "_khm_" in p.stem else "speaker"
        items.append(dict(path=str(p), speaker=spk, raw=text, id=p.stem))
    return items, missing


def _resample(x, sr_in, sr_out):
    try:
        import soxr
        return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _trim_silence(x, sr, top_db=40.0, margin=0.1, frame=1024, hop=256):
    if len(x) <= frame:
        return x
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    db = 10.0 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    active = np.flatnonzero(db > db.max() - top_db)
    m = int(margin * sr)
    start = max(0, int(active[0]) * hop - m)
    end = min(len(x), int(active[-1]) * hop + frame + m)
    return x[start:end]


def _process_audio(job):
    path, sr_out, hop, top_db, margin, target_db = job
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception as e:
        try:
            import librosa
            x, sr = librosa.load(path, sr=None, mono=True)
        except Exception:
            return None, f"unreadable: {type(e).__name__}"
    if sr != sr_out:
        x = _resample(np.ascontiguousarray(x, dtype=np.float32), sr, sr_out)
    x = x.astype(np.float32)
    if len(x) < hop * 8:
        return None, "too_short"
    x -= x.mean()
    x = _trim_silence(x, sr_out, top_db, margin)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak < 1e-3:
        return None, "silent"
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    gain = min(10 ** (target_db / 20.0) / max(rms, 1e-8), 0.95 / peak)
    x = x * gain
    n = (len(x) // hop) * hop
    return np.clip(np.round(x[:n] * 32767.0), -32768, 32767).astype(np.int16), None


def prepare(args):
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)
    cfg = load_config(getattr(args, "config", None))
    sr, hop = cfg["sample_rate"], cfg["hop"]
    t0 = time.time()
    items, missing = collect_items(src)
    log(f"matched {len(items)} rows to audio files; {len(missing)} rows without audio"
        + (f" (e.g. {missing[:3]})" if missing else ""))
    if not items:
        raise SystemExit("Nothing matched - check the folder layout / metadata columns printed above.")

    stats = Counter()
    kept = []
    for it in items:
        raw = it["raw"]
        if has_ambiguous_number(raw):
            stats["drop_ambiguous_number(decimal/time/date)"] += 1
            continue
        t = normalize_khmer(raw)
        if not t:
            stats["drop_empty_text"] += 1
            continue
        if re.search(r"[A-Za-z]", t):
            stats["drop_latin_letters"] += 1
            continue
        if re.search(r"[0-9០-៩]", raw):
            stats["kept_numbers_expanded_to_words"] += 1
        it["text"] = t
        kept.append(it)

    counts = Counter(c for it in kept for c in it["text"])
    rare = {c for c, n in counts.items() if n < args.min_char_count}
    if rare:
        before = len(kept)
        kept = [it for it in kept if not any(c in rare for c in it["text"])]
        stats["drop_rare_characters"] += before - len(kept)
        log(f"rare characters removed from vocabulary: {sorted(rare)}")
    symbols = ["_"] + sorted(c for c in counts if c not in rare)
    log(f"vocabulary: {len(symbols)} symbols: {''.join(symbols[1:])}")

    jobs = [(it["path"], sr, hop, args.trim_db, args.trim_margin, args.target_db) for it in kept]
    audio_tmp = out / "audio_int16.bin.tmp"
    entries, offset = [], 0
    workers = max(1, args.workers or os.cpu_count() or 1)
    log(f"processing {len(jobs)} audio files with {workers} processes -> {sr} Hz mono int16 ...")
    with open(audio_tmp, "wb") as f, mp.get_context("spawn").Pool(workers) as pool:
        for i, (wav, err) in enumerate(pool.imap(_process_audio, jobs, chunksize=8)):
            it = kept[i]
            if wav is None:
                stats[f"drop_audio_{err.split(':')[0]}"] += 1
                continue
            dur = len(wav) / sr
            if dur < args.min_sec:
                stats["drop_shorter_than_min_sec"] += 1
                continue
            if dur > args.max_sec:
                stats["drop_longer_than_max_sec"] += 1
                continue
            f.write(wav.tobytes())
            entries.append(dict(id=it["id"], speaker=it["speaker"], text=it["text"],
                                offset=offset, length=int(len(wav))))
            offset += len(wav)
            if (i + 1) % 2000 == 0:
                log(f"  {i + 1}/{len(jobs)} files ({offset / sr / 3600:.2f} h kept so far)")
    os.replace(audio_tmp, out / "audio_int16.bin")

    # transcripts that do not fit their audio (speaking-rate outliers, per speaker)
    by_spk = defaultdict(list)
    for e in entries:
        by_spk[e["speaker"]].append(e)
    final = []
    for spk, es in by_spk.items():
        rates = np.array([len(e["text"].replace(" ", "")) / (e["length"] / sr) for e in es])
        med = float(np.median(rates))
        mad = float(np.median(np.abs(rates - med))) * 1.4826 + 1e-6
        for e, r in zip(es, rates):
            if len(es) >= 20 and abs(r - med) / mad > args.rate_z:
                stats["drop_speaking_rate_outlier"] += 1
                continue
            if 2 * len(e["text"]) + 1 > e["length"] // hop:
                stats["drop_text_longer_than_audio"] += 1
                continue
            final.append(e)

    speakers = sorted({e["speaker"] for e in final})
    rng = random.Random(1234)
    train, val = [], []
    for spk in speakers:
        es = [e for e in final if e["speaker"] == spk]
        rng.shuffle(es)
        k = min(args.val_per_speaker, max(0, len(es) - 1))
        val += es[:k]
        train += es[k:]
    genders = _speaker_genders(src)
    spk_info = []
    for s in speakers:
        es = [e for e in final if e["speaker"] == s]
        spk_info.append(dict(speaker=s, gender=genders.get(s, "?"), utterances=len(es),
                             hours=round(sum(e["length"] for e in es) / sr / 3600, 3)))
    manifest = dict(version=1, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    hparams=dict(sample_rate=sr, hop=hop), audio_file="audio_int16.bin",
                    total_samples=offset, symbols=symbols, speakers=speakers,
                    speaker_info=spk_info, stats=dict(stats), train=train, val=val)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

    hours = sum(e["length"] for e in final) / sr / 3600
    log("-" * 70)
    log(f"done in {fmt_time(time.time() - t0)}: {len(final)} utterances, {hours:.2f} h, "
        f"{len(speakers)} speakers, train {len(train)} / val {len(val)}")
    for s in spk_info:
        log(f"  speaker {s['speaker']:<16} gender {s['gender']:<8} {s['utterances']:>6} utts  {s['hours']:.2f} h")
    for k, v in sorted(stats.items()):
        log(f"  {k}: {v}")
