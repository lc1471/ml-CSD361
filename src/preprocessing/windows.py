import os
import numpy as np
from audio_features import load_audio, log_mel, windows as audio_windows
from video_features import load_crops, video_windows_by_time

SILENCE = float(np.log(1e-6))


def build_clip(row, root, no_audio="keep", min_valid=0.75):
    """row: a manifest row (dict). Returns aligned 2 s windows, or None if nothing is usable.
    video (n,16,112,112,3) uint8 | audio (n,80,200) float32 | window_idx (n,) = which 2 s slot each is.
    no_audio: 'keep' (as decoded), 'mask' (replace with silence), 'drop' (skip no-audio clips).
    min_valid: drop windows where less than this fraction of the 16 frames has a real face crop.
    n_video / n_audio are KEPT / TOTAL windows (names kept so the test scripts still work)."""
    if row["has_crops"] != "True":
        return None
    has_audio = row["has_audio"] == "True"
    if not has_audio and no_audio == "drop":
        return None
    idxs, frames = load_crops(os.path.join(root, row["crop_path"]))
    aw = audio_windows(log_mel(load_audio(os.path.join(root, row["video_path"]))))
    n_total = len(aw)
    if n_total == 0:
        return None
    vw, valid = video_windows_by_time(idxs, frames, n_total)
    keep = np.where(valid >= min_valid)[0]
    if len(keep) == 0:
        return None
    audio = np.stack(aw)[keep]
    if not has_audio and no_audio == "mask":
        audio = np.full_like(audio, SILENCE)
    return dict(video=vw[keep], audio=audio, window_idx=keep, video_valid=valid[keep],
                has_audio=has_audio, n=len(keep), n_video=len(keep), n_audio=n_total)
