"""Tiny fake dataset with the documented layout of the Khmer ASR Cultural Dataset:

    <root>/metadata.csv          Topic, Subtopic, Speaker ID, Paragraph ID, Sentence ID, Sentences
    <root>/speaker_metadata.csv
    <root>/data/{recipes,giving_gift}/{speaker_id}_khm_{sentence_id}.wav   (48 kHz, 16-bit)

plus deliberately bad rows that `prepare` must filter: Latin text, a decimal number, a missing
file, a silent file and a transcript far too long for its audio.

    python tests/fake_data.py /tmp/fake/raw
"""
import csv
import os
import random
import sys

import numpy as np
import soundfile as sf

WORDS = ["មុខម្ហូប", "តាមដងផ្លូវ", "គឺជា", "មួយ", "មានភាព", "សម្បូរបែប", "និង", "ងាយស្រួល", "ដែល", "គេ",
         "ពេញនិយម", "ក្នុង", "ការ", "បរិភោគ", "ថែមទាំង", "តម្លៃ", "សមរម្យ", "អំណោយ", "ជូន", "មិត្តភក្តិ",
         "ក្រុមគ្រួសារ", "ពិធី", "បុណ្យ", "ចូលឆ្នាំ", "ខ្មែរ", "ស្ករ", "ស្លាបព្រា", "ទឹកត្រី", "អង្ករ",
         "សាច់មាន់", "បន្លែ", "ឆ្ងាញ់", "ណាស់", "យើង", "ត្រូវ", "ចម្អិន", "រយៈពេល", "នាទី", "ក្រាម"]
SPEAKERS = ["f-adt1-0001", "m-adt1-0002", "f-adt2-0003"]
TOPICS = [("Recipes", "Street food dishes", "recipes"), ("Giving Gift", "New year gifts", "giving_gift")]


def _fake_speech(rng, n_chars, sr=48000, f0=140.0):
    t = np.arange(int(n_chars / 13.0 * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 8)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)
    x = 0.2 * x + 0.01 * rng.standard_normal(len(t))
    sil = 0.002 * rng.standard_normal(int(0.5 * sr))
    return np.concatenate([sil, x, sil]).astype(np.float32)


def make_fake_dataset(root, per_speaker=30, seed=0):
    rnd, rng = random.Random(seed), np.random.default_rng(seed)
    rows = []
    for si, spk in enumerate(SPEAKERS):
        for k in range(per_speaker):
            topic, sub, folder = TOPICS[k % 2]
            sent_id = f"{folder}_{k // 10 + 1:02d}_{k:04d}_0001"
            text = "".join(rnd.choice(WORDS) for _ in range(rnd.randint(4, 9)))
            text = text[:len(text) // 2] + "​ " + text[len(text) // 2:] + "។"
            if k == 3:
                text += " ស្ករ ២ ស្លាបព្រា"        # number -> words
            if k == 4:
                text = "Hello " + text              # Latin -> dropped
            if k == 5:
                text += " តម្លៃ 1.5 ដុល្លារ"        # decimal -> dropped
            n_chars = len(text.replace(" ", ""))
            if k == 6:
                n_chars //= 4                       # audio far too short -> speaking-rate outlier
            os.makedirs(f"{root}/data/{folder}", exist_ok=True)
            path = f"{root}/data/{folder}/{spk}_khm_{sent_id}.wav"
            if k == 8:
                sf.write(path, np.zeros(48000 * 2, np.float32), 48000, subtype="PCM_16")  # silent
            elif k != 7:                            # k == 7: metadata row without audio
                sf.write(path, _fake_speech(rng, n_chars, f0=120 + 40 * si), 48000, subtype="PCM_16")
            rows.append([topic, sub, spk, k // 10 + 1, sent_id, text])
    with open(f"{root}/metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Topic", "Subtopic", "Speaker ID", "Paragraph ID", "Sentence ID", "Sentences"])
        w.writerows(rows)
    with open(f"{root}/speaker_metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Speaker ID", "Gender", "Age Group", "Origin City"])
        for s in SPEAKERS:
            w.writerow([s, "Female" if s.startswith("f") else "Male", "adult", "x"])
    return root


if __name__ == "__main__":
    make_fake_dataset(sys.argv[1])
    print("fake dataset written to", sys.argv[1])
