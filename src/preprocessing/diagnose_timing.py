import os
import sys
from collections import Counter

import av
import numpy as np

from video_features import load_crops, SRC_FPS, OUT_FPS, WIN_FRAMES as WIN, TOL
from audio_features import load_audio, log_mel, windows as audio_windows

DATA = r"C:\dev\data\avcaffe"

CLIPS = [
    "aiim043_task_2_clip_040",
    "aiim082_task_6_clip_018",
    "aiim060_task_9_clip_053",
    "aiim076_task_3_clip_049",
    "aiim001_task_1_clip_001",
]

for clip in CLIPS:
    participant, rest = clip.split("_task_")
    task = rest.split("_clip_")[0]

    video_path = os.path.join(
        DATA, "videos", "shorter_segments",
        participant, "task_" + task, clip + ".avi"
    )

    crop_path = os.path.join(
        DATA, "face_crops",
        participant + "_task_" + task,
        clip + ".zip"
    )

    print("\n" + "=" * 65)
    print(clip)

    with av.open(video_path) as container:
        stream = container.streams.video[0]

        print("Video average rate:", stream.average_rate)
        print("Video base rate:", stream.base_rate)
        print("Video frames metadata:", stream.frames)
        print("Video time base:", stream.time_base)
        print("Video stream duration:", stream.duration)
        print("Container duration:", container.duration / 1e6)

    idxs, frames = load_crops(crop_path)

    print("Crop count:", len(idxs))
    print("Crop index range:",
          (idxs[0], idxs[-1]) if idxs else None)

    if len(idxs) > 1:
        gaps = [
            (idxs[i], idxs[i + 1])
            for i in range(len(idxs) - 1)
            if idxs[i + 1] - idxs[i] > 1
        ]
        print("Number of crop gaps:", len(gaps))
        print("First crop gaps:", gaps[:10])

    mel = log_mel(load_audio(video_path))
    n_windows = len(audio_windows(mel))

    print("Audio windows:", n_windows)
    print("Configured source FPS:", SRC_FPS)
    print("Configured output FPS:", OUT_FPS)
    print("Crop tolerance:", TOL)

    selected = []
    distances = []
    valid_counts = []

    idx = np.asarray(idxs)

    for k in range(n_windows):
        window_selected = []
        valid = 0

        for j in range(WIN):
            target = (k * WIN + j) * (SRC_FPS / OUT_FPS)

            if len(idx) == 0:
                continue

            pos = np.searchsorted(idx, target)
            candidates = [
                p for p in (pos - 1, pos)
                if 0 <= p < len(idx)
            ]

            best = min(
                candidates,
                key=lambda p: abs(idx[p] - target)
            )

            distance = abs(idx[best] - target)

            if distance <= TOL:
                valid += 1
                window_selected.append(int(idx[best]))
                distances.append(float(distance))

        selected.extend(window_selected)
        valid_counts.append(valid)

    counts = Counter(selected)

    print("Valid frames per window:", valid_counts)
    print("Total selected crops:", len(selected))
    print("Unique selected crops:", len(counts))
    print("Repeated crop selections:",
          sum(v - 1 for v in counts.values() if v > 1))
    print("Maximum selection distance:",
          max(distances) if distances else None)

    print("Window validity:",
          [round(x / WIN, 3) for x in valid_counts])

print("\nTiming diagnostic complete.")
