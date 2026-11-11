import csv, os, sys

DATA = sys.argv[1] if len(sys.argv) > 1 else r"C:\dev\data\avcaffe"
crops = {}
for root, _, files in os.walk(os.path.join(DATA, "face_crops")):
    for f in files:
        if f.endswith(".zip"):
            crops[f[:-4]] = os.path.relpath(os.path.join(root, f), DATA)

path = os.path.join(DATA, "manifest.csv")
with open(path, newline="") as f:
    rdr = csv.DictReader(f)
    fields = [c for c in rdr.fieldnames if c not in ("crop_path", "has_crops")]
    rows = list(rdr)

for r in rows:
    r["crop_path"] = crops.get(r["clip_id"], "")
    r["has_crops"] = bool(r["crop_path"])
fields += ["crop_path", "has_crops"]

tmp = path + ".tmp"
with open(tmp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
os.replace(tmp, path)

n = len(rows)
c = [r["has_crops"] for r in rows]
a = [r["has_audio"] == "True" for r in rows]
print("clips:", n)
print("video + audio usable:", sum(x and y for x, y in zip(c, a)))
print("video only (no audio):", sum(x and not y for x, y in zip(c, a)))
print("audio only (no crops):", sum((not x) and y for x, y in zip(c, a)))
print("neither:", sum((not x) and (not y) for x, y in zip(c, a)))
for s in ("train", "val"):
    sub = [r for r in rows if r["split"] == s]
    print(s, "clips without crops:", sum(not r["has_crops"] for r in sub), "of", len(sub))
print("wrote", path)
