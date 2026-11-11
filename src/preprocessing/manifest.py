import csv, os, re, sys
from collections import Counter, defaultdict

DATA = sys.argv[1] if len(sys.argv) > 1 else r"C:\dev\data\avcaffe"
CL = ["mental_demand", "effort", "temporal_demand"]
AFFECT = {
    "arousal": ["excited", "wide-awake", "neutral", "dull", "calm"],
    "valence": ["pleasant", "pleased", "neutral", "unsatisfied", "unpleasant"],
}
CLIP_RE = re.compile(r"^(aiim\d+)_task_(\d+)_clip_(\d+)\.\w+$")


def read_labels(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                key, val = line.rsplit(",", 1)
                out[key.strip()] = val.strip()
    return out


def read_ids(path):
    with open(path) as f:
        return {l.strip() for l in f if l.strip()}


def main():
    gt = os.path.join(DATA, "ground_truths")
    info = os.path.join(DATA, "info")
    seg = os.path.join(DATA, "videos", "shorter_segments")

    train = read_ids(os.path.join(info, "train.txt"))
    val = read_ids(os.path.join(info, "val.txt"))
    assert not (train & val), "train/val overlap"
    no_audio = read_ids(os.path.join(info, "no_audio_files.txt"))

    rows = []
    for root, _, files in os.walk(seg):
        for fn in files:
            m = CLIP_RE.match(fn)
            if not m:
                continue
            rows.append(dict(
                clip_id=os.path.splitext(fn)[0],
                participant=m.group(1),
                task=int(m.group(2)),
                clip_idx=int(m.group(3)),
                video_path=os.path.relpath(os.path.join(root, fn), DATA),
                has_audio=fn not in no_audio,
            ))
    if not rows:
        sys.exit("No clips matched. Paste: Get-ChildItem <a task_1 folder> | Select -First 3 Name")
    rows.sort(key=lambda r: (r["participant"], r["task"], r["clip_idx"]))

    cl_lab = {n: read_labels(os.path.join(gt, n + ".txt")) for n in CL}
    af_lab = {n: read_labels(os.path.join(gt, n + ".txt")) for n in AFFECT}
    missing = Counter()
    for r in rows:
        key = "%s_task_%d" % (r["participant"], r["task"])
        r["split"] = "train" if r["participant"] in train else "val" if r["participant"] in val else "none"
        for n in CL:
            try:
                x = float(cl_lab[n].get(key))
            except (TypeError, ValueError):
                x = None
                missing[n] += 1
            r[n] = "" if x is None else x
            r[n + "_bin"] = "" if x is None else int(x > 10)
        for n, classes in AFFECT.items():
            v = af_lab[n].get(key)
            if v is None:
                missing[n] += 1
                r[n] = ""
                continue
            v = v.strip().lower()
            if v not in classes:
                sys.exit("%s: unexpected label value %r" % (n, v))
            r[n] = classes.index(v)

    print("clips:", len(rows), "| participants:", len({r["participant"] for r in rows}), "(paper: 58,112 clips)")
    for s in ("train", "val", "none"):
        sub = [r for r in rows if r["split"] == s]
        print("  %-5s clips=%d participants=%d" % (s, len(sub), len({r["participant"] for r in sub})))
    print("clips per task:", dict(sorted(Counter(r["task"] for r in rows).items())))
    print("no-audio clips:", sum(not r["has_audio"] for r in rows))
    print("missing labels:", dict(missing) or "none")
    for n in CL:
        for s in ("train", "val"):
            v = [r[n + "_bin"] for r in rows if r["split"] == s and r[n + "_bin"] != ""]
            print("%s High-rate (%s): %.3f" % (n, s, sum(v) / len(v) if v else float("nan")))

    out = os.path.join(DATA, "manifest.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
