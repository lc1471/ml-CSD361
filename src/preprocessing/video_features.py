import io, re, zipfile
import numpy as np
import cv2

SRC_FPS = 16   # crop frame rate (inferred; verify with test_video.py)
OUT_FPS = 8    # paper: 8 fps, 16 frames per 2 s window
SIZE = 112
WIN_FRAMES = 16
IDX_RE = re.compile(r"img_(\d+)\.jpg$")


def load_crops(zip_path):
    """Read all face crops of a clip (zip inside zip). Returns (indices, uint8 RGB frames), sorted by index."""
    with zipfile.ZipFile(zip_path) as outer:
        inner = outer.read(outer.namelist()[0])
    items = []
    with zipfile.ZipFile(io.BytesIO(inner)) as z:
        for n in z.namelist():
            m = IDX_RE.search(n)
            if m:
                items.append((int(m.group(1)), n))
        items.sort()
        idxs, frames = [], []
        for i, n in items:
            im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            idxs.append(i)
    arr = np.stack(frames) if frames else np.zeros((0, SIZE, SIZE, 3), np.uint8)
    return idxs, arr


def video_windows(frames, src_fps=SRC_FPS, out_fps=OUT_FPS, win=WIN_FRAMES):
    """(T,112,112,3) at src_fps -> (n,16,112,112,3) non-overlapping 2 s windows at out_fps."""
    sub = frames[:: max(1, round(src_fps / out_fps))]
    n = len(sub) // win
    return sub[: n * win].reshape(n, win, SIZE, SIZE, 3)
