import os, random, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ClipReader

DATA = r"C:\dev\data\avcaffe"
r = ClipReader(DATA, "val", mode="random")
print("val clips with crops:", len(r))
for i in range(3):
    it = r[i]
    print(it["clip_id"], it["video"].shape, it["video"].dtype, it["audio"].shape, it["labels"].tolist(),
          "has_audio", it["has_audio"], "window", it["window_idx"].tolist(), "of", it["n_windows"])

random.seed(0)
idx = random.sample(range(len(r)), 30)
fallback = 0
vm, vs, am, asd = [], [], [], []
t = time.time()
for i in idx:
    it = r[i]
    fallback += it["clip_id"] != r.rows[i]["clip_id"]
    vm.append(it["video"].mean()); vs.append(it["video"].std())
    if it["has_audio"]:
        am.append(it["audio"].mean()); asd.append(it["audio"].std())
ms = (time.time() - t) / len(idx) * 1000
print("30 random items: %.0f ms/item | fallbacks to next clip: %d" % (ms, fallback))
print("normalised video mean/std (avg over items): %.2f / %.2f" % (np.mean(vm), np.mean(vs)))
print("normalised audio mean/std (clips with audio): %.2f / %.2f" % (np.mean(am), np.mean(asd)))
a, b = r[5], r[5]
print("deterministic within an epoch:", np.array_equal(a["video"], b["video"]) and np.array_equal(a["audio"], b["audio"]))
ra = ClipReader(DATA, "val", mode="all")
it = ra[5]
print("all-mode:", it["clip_id"], it["video"].shape, it["audio"].shape, "windows", it["window_idx"].tolist())
rm = ClipReader(DATA, "val", mode="all", no_audio="mask")
na = next(i for i, row in enumerate(rm.rows) if row["has_audio"] != "True")
print("masked no-audio clip, unique audio values:", np.unique(rm[na]["audio"]).round(3)[:3])
