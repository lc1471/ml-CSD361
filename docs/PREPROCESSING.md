# AVCAffe Preprocessing Pipeline

## 1. Dataset overview

Dataset: AVCAffe

Dataset root:
`C:\dev\data\avcaffe`

Manifest:
`C:\dev\data\avcaffe\manifest.csv`

Total manifest clips: 58,112

| Split | Clips |
|---|---:|
| Train | 47,516 |
| Validation | 10,596 |
| Test | Not provided in the current manifest |

The current dataset uses participant-based train/validation assignments.

There are 74 manifest clips without corresponding crop ZIP files:
- Train: 69
- Validation: 5

These are excluded by the dataset loader.

## 2. Video preprocessing

Implementation: `src/preprocessing/video_features.py`

| Parameter | Value |
|---|---:|
| Source crop FPS | 16 |
| Output FPS | 8 |
| Frame dimensions | 112 x 112 |
| Frames per window | 16 |
| Window duration | 2 seconds |
| Crop timestamp tolerance | 4 source frames |

Video frames are aligned using timestamps.

For each output timestamp, the nearest available crop is selected within the configured tolerance.

Missing frames are represented by black frames.

Windows are retained when at least 75% of their frames are valid.

## 3. Audio preprocessing

Implementation: `src/preprocessing/audio_features.py`

| Parameter | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Feature representation | Log-Mel spectrogram |
| Mel bins | 80 |
| Hop length | 10 ms |
| FFT size | 1024 |
| Window duration | 2 seconds |
| Frames per window | 200 |

Audio is decoded and resampled before log-Mel feature extraction.

Only complete two-second audio windows are retained.

## 4. Audio-video alignment

Implementation: `src/preprocessing/windows.py`

Audio and video are divided into corresponding two-second windows.

A video window is retained when its valid-frame fraction is at least 0.75.

The loader supports three no-audio policies:

- `keep`: retain decoded audio features.
- `mask`: replace audio features with the configured silence value.
- `drop`: exclude clips marked as having no audio.

The default policy is `keep`.

The dataset also returns a `has_audio` flag so the model can account for missing audio.

## 5. Labels

Default prediction targets:

- `mental_demand_bin`
- `effort_bin`
- `temporal_demand_bin`

All three targets are binary.

Label validation found zero missing values across the manifest.

## 6. Normalization

Statistics are stored in:

`configs/norm_stats.json`

They were computed using a reproducible random sample of 1,000 training clips (seed 0).

Audio statistics use clips marked as having audio only.

Video statistics use RGB values scaled to [0, 1].

Validation data is not used to calculate normalization statistics.

## 7. Dataset loader

Implementation:
- `src/preprocessing/dataset.py`
- `src/preprocessing/torch_dataset.py`

The NumPy-based `ClipReader` performs preprocessing on demand rather than saving every processed window to disk.

Supported modes:

### Random mode

Returns one randomly selected valid two-second window per clip.

Shapes:

- Video: `(3, 16, 112, 112)`
- Audio: `(1, 80, 200)`

### All mode

Returns all valid windows for a clip.

Shapes:

- Video: `(W, 3, 16, 112, 112)`
- Audio: `(W, 1, 80, 200)`

`W` is the number of valid windows.

The PyTorch wrapper converts NumPy arrays into tensors.

## 8. Loading the dataset

Example:

```python
from torch_dataset import AVCAffeDataset

train_dataset = AVCAffeDataset(
    root=r"C:\dev\data\avcaffe",
    split="train",
    mode="random",
    no_audio="keep"
)

val_dataset = AVCAffeDataset(
    root=r"C:\dev\data\avcaffe",
    split="val",
    mode="random",
    no_audio="keep"
)

sample = train_dataset[0]

print(sample["video"].shape)
print(sample["audio"].shape)
print(sample["labels"])
print(sample["has_audio"])
```

For training, use `mode="random"` as the initial configuration.

For evaluation, consider `mode="all"` and aggregate predictions at the clip level.

## 9. Validation status

Completed checks:

- Manifest split validation.
- Missing crop identification.
- Video preprocessing smoke tests.
- Audio preprocessing smoke tests.
- Audio-video window alignment checks.
- No-audio handling checks.
- Label distribution and missing-label checks.
- Normalization statistics inspection.
- NumPy dataset loader smoke tests.

PyTorch DataLoader and GPU tests remain pending because Windows Application Control currently blocks the installed PyTorch DLL.

The installed PyTorch build is CPU-only (`2.15.0+cpu`), so a compatible CUDA-enabled build is also required.

## 10. Important limitations

- The current manifest contains train and validation splits only.
- No test split has been established.
- Preprocessing happens on demand; processed features are not stored as a separate dataset.
- Some clips have missing crop ZIP files and are excluded.
- The current 75% video-validity threshold is used to filter incomplete windows.
- Audio availability is explicitly returned to the model.

## 11. Main preprocessing files

| File | Responsibility |
|---|---|
| `video_features.py` | Video crop loading and temporal alignment |
| `audio_features.py` | Audio decoding and log-Mel extraction |
| `windows.py` | Audio-video window construction |
| `dataset.py` | NumPy dataset reader |
| `torch_dataset.py` | PyTorch Dataset wrapper |
| `compute_stats.py` | Training-data normalization statistics |
| `audit_dataset.py` | Dataset alignment audit |
