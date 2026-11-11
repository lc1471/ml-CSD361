import csv, os
from windows import build_clip

DATA = r"C:\dev\data\avcaffe"
rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.csv"))))
val = [r for r in rows if r["split"] == "val" and r["has_crops"] == "True"]
pick = [r for r in val if r["has_audio"] == "True"][:3] + [r for r in val if r["has_audio"] == "False"][:3]
for r in pick:
    for mode in ("keep", "mask"):
        d = build_clip(r, DATA, mode)
        print(r["clip_id"], "| has_audio:", r["has_audio"], "| mode:", mode, "|",
              None if d is None else (d["video"].shape, d["audio"].shape, "n_video/n_audio:", d["n_video"], d["n_audio"],
                                      "audio mean: %.2f" % d["audio"].mean()))
