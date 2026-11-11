import csv, os, random
from collections import Counter
import numpy as np
import av
from video_features import load_crops, video_windows, SRC_FPS
from audio_features import load_audio, log_mel, windows as audio_windows

DATA = r"C:\dev\data\avcaffe"
rows = [r for r in csv.DictReader(open(os.path.join(DATA, "manifest.csv"))) if r["has_crops"] == "True"]
random.seed(1)
sample = random.sample(rows, 300)

diffs, bad = Counter(), []
non_contig = start_nonzero = 0
ratios = []
for r in sample:
    idxs, arr = load_crops(os.path.join(DATA, r["crop_path"]))
    vp = os.path.join(DATA, r["video_path"])
    with av.open(vp) as c:
        dur = c.duration / 1e6
    n_v = len(video_windows(arr))
    n_a = len(audio_windows(log_mel(load_audio(vp))))
    contig = bool(idxs) and idxs == list(range(idxs[0], idxs[0] + len(idxs)))
    non_contig += not contig
    start_nonzero += bool(idxs) and idxs[0] != 0
    ratios.append(len(idxs) / (dur * SRC_FPS))
    diffs[n_v - n_a] += 1
    if n_v != n_a and len(bad) < 6:
        bad.append((r["clip_id"], len(idxs), (idxs[0], idxs[-1]) if idxs else None, contig, round(dur, 2), n_v, n_a))

print("n_video - n_audio:", dict(sorted(diffs.items())))
print("crop sequences with gaps:", non_contig, "of", len(sample))
print("crop sequences not starting at 0:", start_nonzero, "of", len(sample))
q = np.percentile(ratios, [1, 10, 50, 90, 99])
print("crops / (duration * %d fps) p1/p10/p50/p90/p99: %s" % (SRC_FPS, " / ".join("%.3f" % x for x in q)))
print("first mismatched clips (id, n_crops, idx range, contiguous, duration, n_video, n_audio):")
for b in bad:
    print(" ", b)
