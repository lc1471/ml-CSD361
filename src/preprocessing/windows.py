import os
import numpy as np
from audio_features import load_audio, log_mel, windows as audio_windows
from video_features import load_crops, video_windows

SILENCE = float(np.log(1e-6))


def build_clip(row, root, no_audio="keep"):
    """row: a manifest row (dict). Returns aligned 2 s windows or None.
    video (n,16,112,112,3) uint8 | audio (n,80,200) float32.
    no_audio: 'keep' (as decoded), 'mask' (replace with silence), 'drop' (skip no-audio clips)."""
    if row["has_crops"] != "True":
        return None
    has_audio = row["has_audio"] == "True"
    if not has_audio and no_audio == "drop":
        return None
    _, frames = load_crops(os.path.join(root, row["crop_path"]))
    vw = video_windows(frames)
    aw = audio_windows(log_mel(load_audio(os.path.join(root, row["video_path"]))))
    n = min(len(vw), len(aw))
    if n == 0:
        return None
    audio = np.stack(aw[:n])
    if not has_audio and no_audio == "mask":
        audio = np.full_like(audio, SILENCE)
    return dict(video=vw[:n], audio=audio, has_audio=has_audio, n=n, n_video=len(vw), n_audio=len(aw))
