import csv
import json
import random
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from windows import build_clip
from video_features import (
    load_crops, SRC_FPS, OUT_FPS, WIN_FRAMES, TOL
)

DATA = r"C:\dev\data\avcaffe"
MANIFEST = os.path.join(DATA, "manifest.csv")
MIN_VALID = 0.75


def count_crop_reuse(idxs, window_indices):
    """Count repeated crop assignments in retained windows."""
    if not idxs:
        return 0, 0

    idx = np.asarray(idxs)
    selected = []

    for k in window_indices:
        for j in range(WIN_FRAMES):
            target = (int(k) * WIN_FRAMES + j) * (
                SRC_FPS / OUT_FPS
            )

            pos = np.searchsorted(idx, target)
            candidates = [
                p for p in (pos - 1, pos)
                if 0 <= p < len(idx)
            ]

            if not candidates:
                continue

            best = min(
                candidates,
                key=lambda p: abs(idx[p] - target)
            )

            if abs(idx[best] - target) <= TOL:
                selected.append(int(idx[best]))

    counts = Counter(selected)
    repeated = sum(v - 1 for v in counts.values() if v > 1)

    return len(selected), repeated


def new_stats():
    return Counter()


def update(stats, key, value=1):
    stats[key] += value


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if limit > 0:
        rng = random.Random(0)
        splits = sorted(set(row["split"] for row in rows))
        per_split = limit // len(splits)

        sampled = []
        for split in splits:
            group = [row for row in rows if row["split"] == split]
            sampled.extend(
                rng.sample(group, min(per_split, len(group)))
            )

        rows = sampled

    overall = new_stats()
    by_split = defaultdict(new_stats)
    errors = []
    per_clip = []

    start = time.perf_counter()

    for number, row in enumerate(rows, 1):
        clip = row["clip_id"]
        split = row["split"]

        stats = by_split[split]

        update(overall, "clips")
        update(stats, "clips")

        has_audio = row["has_audio"] == "True"

        if not has_audio:
            update(overall, "no_audio_clips")
            update(stats, "no_audio_clips")

        if row["has_crops"] != "True":
            update(overall, "missing_crop_clips")
            update(stats, "missing_crop_clips")
            per_clip.append({
                "clip_id": clip,
                "split": split,
                "has_audio": has_audio,
                "has_crops": False,
                "status": "missing_crops"
            })
            continue

        try:
            result = build_clip(row, DATA, min_valid=MIN_VALID)

            if result is None:
                update(overall, "clips_rejected_all_windows")
                update(stats, "clips_rejected_all_windows")
                per_clip.append({
                    "clip_id": clip,
                    "split": split,
                    "has_audio": has_audio,
                    "has_crops": True,
                    "status": "rejected_all_windows"
                })
                continue

            n_kept = int(result["n"])
            n_audio = int(result["n_audio"])

            update(overall, "clips_kept")
            update(stats, "clips_kept")

            update(overall, "audio_windows_total", n_audio)
            update(stats, "audio_windows_total", n_audio)

            update(overall, "video_windows_kept", n_kept)
            update(stats, "video_windows_kept", n_kept)

            dropped = n_audio - n_kept

            update(overall, "windows_dropped", dropped)
            update(stats, "windows_dropped", dropped)

            valid = np.asarray(result["video_valid"])

            partial = int(np.sum(valid < 1.0))

            update(overall, "partial_kept_windows", partial)
            update(stats, "partial_kept_windows", partial)

            if not has_audio:
                update(overall, "no_audio_clips_kept")
                update(stats, "no_audio_clips_kept")

            # Count repeated crop assignments in retained windows.
            idxs, _ = load_crops(
                os.path.join(DATA, row["crop_path"])
            )

            assignments, repeated = count_crop_reuse(
                idxs, result["window_idx"]
            )

            update(overall, "crop_assignments", assignments)
            update(stats, "crop_assignments", assignments)

            update(overall, "repeated_crop_assignments", repeated)
            update(stats, "repeated_crop_assignments", repeated)

            if repeated > 0:
                update(overall, "clips_with_crop_reuse")
                update(stats, "clips_with_crop_reuse")

            window_indices = [int(x) for x in result["window_idx"]]
            valid_values = [round(float(x), 4) for x in valid]

            per_clip.append({
                "clip_id": clip,
                "split": split,
                "has_audio": has_audio,
                "has_crops": True,
                "status": "kept",
                "audio_windows_total": n_audio,
                "video_windows_kept": n_kept,
                "dropped_windows": dropped,
                "dropped_window_indices": [
                    i for i in range(n_audio)
                    if i not in window_indices
                ],
                "partial_kept_windows": partial,
                "kept_window_indices": window_indices,
                "video_valid_per_kept_window": valid_values,
                "crop_assignments": assignments,
                "repeated_crop_assignments": repeated
            })

        except Exception as exc:
            update(overall, "processing_errors")
            update(stats, "processing_errors")

            per_clip.append({
                "clip_id": clip,
                "split": split,
                "has_audio": has_audio,
                "has_crops": True,
                "status": "processing_error",
                "error": repr(exc)
            })

            if len(errors) < 30:
                errors.append({
                    "clip_id": clip,
                    "error": repr(exc)
                })

        if number % 250 == 0 or number == len(rows):
            elapsed = time.perf_counter() - start
            print(
                f"Processed {number}/{len(rows)} clips "
                f"({elapsed:.1f}s)",
                flush=True
            )

    elapsed = time.perf_counter() - start

    report = {
        "clips_requested": len(rows),
        "min_valid": MIN_VALID,
        "source_fps": SRC_FPS,
        "output_fps": OUT_FPS,
        "window_frames": WIN_FRAMES,
        "crop_tolerance": TOL,
        "elapsed_seconds": round(elapsed, 2),
        "overall": dict(overall),
        "by_split": {
            split: dict(stats)
            for split, stats in by_split.items()
        },
        "per_clip": per_clip,
        "errors": errors
    }

    os.makedirs("reports", exist_ok=True)

    suffix = str(limit) if limit > 0 else "full"
    output = os.path.join(
        "reports", f"alignment_audit_{suffix}.json"
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== AUDIT RESULTS ==========")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
