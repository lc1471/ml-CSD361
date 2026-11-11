import csv, os, random
from collections import Counter
import numpy as np
from audio_features import load_audio

DATA = r"C:\dev\data\avcaffe"
rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.csv"))))
random.seed(0)

tot, na = Counter(r["task"] for r in rows), Counter(r["task"] for r in rows if r["has_audio"] == "False")
print("no-audio share by task:", {t: round(na[t] / tot[t], 2) for t in sorted(tot, key=int)})

for name, flag in (("no_audio", "False"), ("has_audio", "True")):
    pool = [r for r in rows if r["has_audio"] == flag]
    rms = []
    for r in random.sample(pool, 150):
        w = load_audio(os.path.join(DATA, r["video_path"]))
        rms.append(float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0)
    rms = np.array(rms)
    p = np.percentile(rms, [10, 50, 90])
    print("%-9s rms p10/p50/p90: %.5f / %.5f / %.5f | exactly silent (<1e-4): %d of 150" % (name, *p, (rms < 1e-4).sum()))
