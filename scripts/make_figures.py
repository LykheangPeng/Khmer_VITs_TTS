import csv, glob, os, numpy as np, soundfile as sf, librosa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
SURF, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
FIG = "docs/figures"
# 1. loss curves
rows = list(csv.DictReader(open("results/train_log.csv")))
step = np.array([int(r["step"]) for r in rows]) / 1000
panels = [("mel", "Mel L1 (×45)"), ("kl", "KL divergence"), ("dur", "Duration NLL"),
          ("loss_d", "Discriminator loss"), ("adv", "Generator adversarial loss"), ("fm", "Feature-matching loss")]
fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
for ax, (k, title) in zip(axes.flat, panels):
    y = np.array([float(r[k]) for r in rows])
    ax.plot(step, y, color=BLUE, lw=2)
    ax.set_title(f"{title}  ·  final {y[-1]:.2f}", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("training steps (thousands)")
fig.suptitle("Training losses, 107,336 steps in 11 h on one A100-80GB (200-step averages)", x=0.01, ha="left", fontsize=12, color=INK)
fig.savefig(f"{FIG}/loss_curves.png", dpi=150); plt.close(fig)
# 2. data per speaker (from the prepare log)
spk = [("f-adt1-0001","F",16,0.04),("f-adt1-0002","F",3891,8.34),("f-adt2-0001","F",16,0.04),("f-adt2-0002","F",4187,8.51),
       ("m-adt1-0001","M",16,0.04),("m-adt1-0002","M",2473,7.01),("m-adt1-0003","?",817,1.82),("m-adt1-0004","?",566,1.21),
       ("m-adt2-0001","M",16,0.03),("m-adt2-0002","M",4167,9.64)]
fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
names = [f"{s}  ({g})" for s, g, _, _ in spk][::-1]; hrs = [h for *_, h in spk][::-1]; n = [u for _, _, u, _ in spk][::-1]
bars = ax.barh(names, hrs, color=BLUE, height=0.62)
for b, h, u in zip(bars, hrs, n):
    ax.text(h + 0.12, b.get_y() + b.get_height() / 2, f"{h:.2f} h · {u:,} clips", va="center", fontsize=9, color=INK2)
ax.set_xlim(0, 12); ax.set_xlabel("hours of audio after filtering"); ax.grid(axis="y", visible=False)
ax.set_title("Training data per speaker ID (36.67 h total)", loc="left", fontsize=12, color=INK)
fig.savefig(f"{FIG}/data_per_speaker.png", dpi=150); plt.close(fig)
# 3. spectrograms
cmap = LinearSegmentedColormap.from_list("blue_ramp", RAMP)
files = sorted(glob.glob("samples/*.wav"))
fig, axes = plt.subplots(5, 2, figsize=(12, 11), constrained_layout=True)
for ax, f in zip(axes.flat, files):
    x, sr = sf.read(f)
    m = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, power=1.0)
    db = 20 * np.log10(np.maximum(m, 1e-5))
    ax.imshow(db, origin="lower", aspect="auto", cmap=cmap, vmin=db.max() - 80, vmax=db.max(),
              extent=[0, len(x) / sr, 0, 80])
    ax.set_title(os.path.basename(f)[:-4], loc="left", fontsize=10, color=INK); ax.grid(False)
    ax.set_ylabel("mel bin"); ax.set_xlabel("seconds")
fig.suptitle("Mel spectrograms of the synthesized samples (step 107,336)", x=0.01, ha="left", fontsize=12, color=INK)
fig.savefig(f"{FIG}/sample_spectrograms.png", dpi=100); plt.close(fig)
print("ok")
