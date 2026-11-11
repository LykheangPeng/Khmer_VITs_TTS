"""Minimal Python API example.

    python examples/synthesize.py runs/khmer_multispeaker/G_latest.pt
"""
import sys

import soundfile as sf

from khmer_vits.inference import load_generator, synthesize

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/khmer_multispeaker/G_latest.pt"
net_g, hp, tokenizer, speakers = load_generator(ckpt)   # a run folder also works
print("speakers:", speakers)

text = "សួស្តី! ថ្ងៃនេះ យើងនឹងរៀនធ្វើម្ហូបខ្មែរ។"
for name in speakers:
    audio = synthesize(net_g, tokenizer, hp, text, speaker=speakers.index(name),
                       noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)
    sf.write(f"{name}.wav", audio, hp["sample_rate"])
    print(f"{name}.wav  {len(audio) / hp['sample_rate']:.2f} s")
