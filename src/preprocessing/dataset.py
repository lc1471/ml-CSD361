"""Torch-free reader for AVCAffe preprocessing (testable on any machine with numpy, cv2, PyAV).
One item = one clip. See torch_dataset.py for the thin PyTorch wrapper."""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windows import build_clip  # noqa: E402

DEFAULT_TARGETS = ("mental_demand_bin", "effort_bin", "temporal_demand_bin")
DEFAULT_STATS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "norm_stats.json"))


class ClipReader:
    """mode='random': one random valid 2 s window per clip -> video (3,16,112,112), audio (1,80,200)
    mode='all':    every valid window                    -> video (W,3,16,112,112), audio (W,1,80,200)
    labels: int64 vector in the order of `targets` (-1 if a label is missing).
    Clips with no usable window are skipped by moving to the next clip (at most 10 tries)."""

    def __init__(self, root, split, targets=DEFAULT_TARGETS, mode="random", no_audio="keep",
                 min_valid=0.75, stats_path=DEFAULT_STATS, seed=0, normalize=True):
        assert split in ("train", "val"), split
        assert mode in ("random", "all"), mode
        assert no_audio in ("keep", "mask", "drop"), no_audio
        self.root, self.targets, self.mode = root, tuple(targets), mode
        self.no_audio, self.min_valid, self.seed, self.epoch = no_audio, min_valid, seed, 0
        self.normalize = normalize
        with open(os.path.join(root, "manifest.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        for t in self.targets:
            if t not in rows[0]:
                raise KeyError("target %r is not a manifest column" % t)
        self.rows = [r for r in rows if r["split"] == split and r["has_crops"] == "True"
                     and not (no_audio == "drop" and r["has_audio"] != "True")]
        if normalize:
            with open(stats_path) as f:
                s = json.load(f)
            self.a_mean = float(s["audio_logmel_mean"])
            self.a_std = float(s["audio_logmel_std"])
            self.v_mean = np.array(s["video_rgb_mean_0to1"], np.float32).reshape(1, 1, 1, 1, 3)
            self.v_std = np.array(s["video_rgb_std_0to1"], np.float32).reshape(1, 1, 1, 1, 3)

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _label(self, row, t):
        v = row[t]
        return -1 if v == "" else int(float(v))

    def __getitem__(self, i):
        n = len(self.rows)
        for off in range(min(n, 10)):
            row = self.rows[(i + off) % n]
            d = build_clip(row, self.root, self.no_audio, self.min_valid)
            if d is not None:
                return self._pack(row, d, i)
        raise RuntimeError("no usable clip found starting at index %d" % i)

    def _pack(self, row, d, i):
        if self.mode == "random":
            rng = np.random.default_rng([self.seed, self.epoch, i])
            sel = np.array([int(rng.integers(d["n"]))])
        else:
            sel = np.arange(d["n"])
        video = d["video"][sel].astype(np.float32) / 255.0            # (m,16,112,112,3)
        audio = d["audio"][sel].astype(np.float32)                    # (m,80,200)
        if self.normalize:
            video = (video - self.v_mean) / self.v_std
            audio = (audio - self.a_mean) / self.a_std
        video = np.ascontiguousarray(np.transpose(video, (0, 4, 1, 2, 3)))   # (m,3,16,112,112)
        audio = audio[:, None]                                               # (m,1,80,200)
        if self.mode == "random":
            video, audio = video[0], audio[0]
        return dict(video=video, audio=audio,
                    labels=np.array([self._label(row, t) for t in self.targets], np.int64),
                    has_audio=bool(d["has_audio"]), clip_id=row["clip_id"],
                    window_idx=d["window_idx"][sel].astype(np.int64), n_windows=int(d["n_audio"]))
