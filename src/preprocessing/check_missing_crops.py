import csv, os
from collections import Counter
DATA = r"C:\dev\data\avcaffe"
have = set()
for root, _, files in os.walk(os.path.join(DATA, "face_crops")):
    for f in files:
        if f.endswith(".zip"):
            have.add(f[:-4])
rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.csv"))))
miss = [r for r in rows if r["clip_id"] not in have]
print("missing:", len(miss), "of", len(rows), "= %.3f%%" % (100 * len(miss) / len(rows)))
print("by split:", dict(Counter(r["split"] for r in miss)))
print("by participant:", dict(Counter(r["participant"] for r in miss)))
print("by task:", dict(sorted(Counter(int(r["task"]) for r in miss).items())))
print("also no-audio:", sum(r["has_audio"] == "False" for r in miss))
g = {}
for r in miss:
    g.setdefault((r["participant"], r["task"]), []).append(int(r["clip_idx"]))
for k, v in sorted(g.items()):
    v.sort(); rng = []; s = p = v[0]
    for x in v[1:]:
        if x != p + 1:
            rng.append((s, p)); s = x
        p = x
    rng.append((s, p))
    print(k, ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in rng))
