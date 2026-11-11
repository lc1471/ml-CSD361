import csv, os, random
from windows import build_clip

DATA = r"C:\dev\data\avcaffe"
rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.csv"))))
by_id = {r["clip_id"]: r for r in rows}
for cid in ("aiim043_task_2_clip_040", "aiim082_task_6_clip_018", "aiim060_task_9_clip_053",
            "aiim076_task_3_clip_049", "aiim001_task_1_clip_001"):
    d = build_clip(by_id[cid], DATA, "keep", min_valid=0.0)
    print(cid, "->", None if d is None else ("windows total:", d["n_audio"], "valid frac per window:",
                                              [round(float(x), 2) for x in d["video_valid"]]))

pool = [r for r in rows if r["has_crops"] == "True"]
random.seed(1)
sample = random.sample(pool, 300)
fracs, none_clips = [], 0
for r in sample:
    d = build_clip(r, DATA, "keep", min_valid=0.0)
    if d is None:
        none_clips += 1
        continue
    fracs.extend(float(x) for x in d["video_valid"])
n = len(fracs)
print("sample windows:", n, "| clips with no windows at all:", none_clips)
print("windows with any black frame (valid < 1.0):", sum(f < 1.0 for f in fracs))
print("windows dropped by default (valid < 0.75):", sum(f < 0.75 for f in fracs))
