import csv, json, os, random, sys
import numpy as np
from windows import build_clip

DATA = r"C:\dev\data\avcaffe"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rows = [r for r in csv.DictReader(open(os.path.join(DATA, "manifest.csv")))
        if r["split"] == "train" and r["has_crops"] == "True"]
random.seed(0)
sample = random.sample(rows, min(N, len(rows)))

a_n = a_s = a_ss = 0.0
na_n = na_s = 0.0
v_n = 0
v_s = np.zeros(3)
v_ss = np.zeros(3)
mismatch = 0
for i, r in enumerate(sample, 1):
    d = build_clip(r, DATA, "keep")
    if d is None:
        continue
    mismatch += d["n_video"] != d["n_audio"]
    a = d["audio"].astype(np.float64)
    if d["has_audio"]:
        a_n += a.size; a_s += a.sum(); a_ss += (a ** 2).sum()
    else:
        na_n += a.size; na_s += a.sum()
    px = (d["video"][:, ::4].astype(np.float64) / 255.0).reshape(-1, 3)
    v_n += len(px); v_s += px.sum(0); v_ss += (px ** 2).sum(0)
    if i % 100 == 0:
        print("processed", i, "of", len(sample))

a_mean = a_s / a_n
a_std = float(np.sqrt(a_ss / a_n - a_mean ** 2))
v_mean = v_s / v_n
v_std = np.sqrt(v_ss / v_n - v_mean ** 2)
out = {
    "note": "computed on a random sample of TRAIN clips with crops; audio stats use has_audio clips only",
    "sample_clips": len(sample), "seed": 0,
    "audio_logmel_mean": float(a_mean), "audio_logmel_std": a_std,
    "no_audio_logmel_mean_for_reference": float(na_s / na_n) if na_n else None,
    "video_rgb_mean_0to1": [float(x) for x in v_mean], "video_rgb_std_0to1": [float(x) for x in v_std],
}
print(json.dumps(out, indent=2))
print("clips where n_video != n_audio:", mismatch, "of", len(sample))
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "norm_stats.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", os.path.normpath(dst))
