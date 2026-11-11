import csv, os
DATA = r"C:\dev\data\avcaffe"
have = set()
for root, _, files in os.walk(os.path.join(DATA, "face_crops")):
    for f in files:
        if f.endswith(".zip"):
            have.add(f[:-4])
ids = [r["clip_id"] for r in csv.DictReader(open(os.path.join(DATA, "manifest.csv")))]
miss = [i for i in ids if i not in have]
extra = have - set(ids)
print("manifest clips:", len(ids), "| crop zips:", len(have))
print("clips without crops:", len(miss), miss[:5])
print("crop zips not in manifest:", len(extra), sorted(extra)[:5])
