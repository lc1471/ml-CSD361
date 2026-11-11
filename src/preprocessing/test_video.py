import os, sys
import numpy as np
import cv2
import av
from video_features import load_crops, video_windows

DATA = r"C:\dev\data\avcaffe"
for clip in sys.argv[1:]:
    p, rest = clip.split("_task_")
    t = rest.split("_clip_")[0]
    zp = os.path.join(DATA, "face_crops", p + "_task_" + t, clip + ".zip")
    vp = os.path.join(DATA, "videos", "shorter_segments", p, "task_" + t, clip + ".avi")
    idxs, arr = load_crops(zp)
    with av.open(vp) as c:
        dur = c.duration / 1e6
    contiguous = idxs == list(range(idxs[0], idxs[0] + len(idxs))) if idxs else False
    print(clip)
    print("  crops:", len(idxs), "| index range:", (idxs[0], idxs[-1]) if idxs else None, "| contiguous:", contiguous)
    print("  clip duration: %.2f s | implied crop fps: %.2f" % (dur, len(idxs) / dur))
    w = video_windows(arr)
    print("  frames array:", arr.shape, arr.dtype, "| windows:", w.shape)
    if len(w):
        sheet = np.vstack([np.hstack(list(w[0][:8])), np.hstack(list(w[0][8:16]))])
        out = os.path.join(DATA, "preview_" + clip + ".png")
        cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print("  preview:", out)
