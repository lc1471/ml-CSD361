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

TOL = 4  # max distance (source frames, 0.25 s at 16 fps) to accept a crop for a target frame


def video_windows_by_time(idxs, frames, n_windows, src_fps=SRC_FPS, out_fps=OUT_FPS, win=WIN_FRAMES, tol=TOL):
    """Pick crops by TIME so gaps in the crop sequence cannot shift later frames.
    Window k covers k*2 s .. (k+1)*2 s; its j-th frame is the crop nearest to source index
    (k*win + j) * (src_fps / out_fps). A target frame with no crop within `tol` source frames
    is left black and counted as invalid.
    Returns video (n,win,112,112,3) uint8 and valid_frac (n,) float."""
    step = src_fps / out_fps
    video = np.zeros((n_windows, win, SIZE, SIZE, 3), np.uint8)
    valid = np.zeros((n_windows, win), bool)
    if len(idxs) == 0:
        return video, valid.mean(axis=1)
    idx = np.asarray(idxs)
    for k in range(n_windows):
        for j in range(win):
            target = (k * win + j) * step
            pos = np.searchsorted(idx, target)
            cands = [p for p in (pos - 1, pos) if 0 <= p < len(idx)]
            best = min(cands, key=lambda p: abs(idx[p] - target))
            if abs(idx[best] - target) <= tol:
                video[k, j] = frames[best]
                valid[k, j] = True
    return video, valid.mean(axis=1)
