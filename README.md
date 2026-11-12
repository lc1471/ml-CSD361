# Affect-Aware Dynamic UI
### AVCAffe Dataset – Multimodal Preprocessing Pipeline
-Note: this repo is currently only for pre-processing , the model will be worked upon in the following weeks from 11th oct 2026

## Overview

Affect-Aware Dynamic UI is a project that aims to develop an adaptive user interface capable of responding to a user's cognitive and affective state in real time.

The system uses multimodal information, including facial video and audio, to estimate cognitive workload. Based on these predictions, the interface can dynamically adapt to reduce cognitive overload and improve the user experience.

For example, when a user experiences high cognitive load, the interface may activate a **Focus Mode** that:
- Hides unnecessary sidebars and interface elements.
- Mutes non-essential notifications.
- Reduces visual clutter.
- Prioritizes the user's primary task.

This repository focuses specifically on the **data preprocessing component** of the project.

## Project Objectives

The primary objective of this repository is to transform the raw AVCAffe dataset into clean, aligned and model-ready multimodal inputs.

The preprocessing pipeline is designed to:

- Organize raw video clips and ground-truth labels.
- Generate a structured dataset manifest.
- Extract audio features using log-Mel spectrograms.
- Extract facial video features from released face crops.
- Align audio and video into corresponding temporal windows.
- Calculate normalization statistics.
- Provide a reproducible data-loading interface for downstream model training and inference.

## Dataset

**AVCAffe: A Large Scale Audio-Visual Dataset of Cognitive Load and Affect for Remote Work**

Paper: Sarkar, Posen and Etemad, AAAI 2023.

Dataset source: [Borealis Dataverse](https://borealisdata.ca/)

DOI: `10.5683/SP3/PSWY62`

The dataset contains:

| Attribute | Details |
|---|---|
| Participants | 106 |
| Total clips | 58,112 |
| Modalities | Audio and video |
| Video format | AVI |
| Video resolution | 456 × 256 |
| Frame rate | 25 FPS |
| Audio | AAC, 44.1 kHz, stereo |
| Clip duration | Approximately 6 seconds |

### Ground-truth labels

The dataset contains cognitive workload and affect-related annotations.

**Cognitive workload:**
- Mental demand
- Temporal demand
- Effort
- Performance
- Physical demand

**Affective labels:**
- Arousal
- Valence
- Dominance

The current preprocessing pipeline primarily targets mental demand, temporal demand and effort.

Binary workload labels are generated using a threshold of 10:

- High workload: score greater than 10.
- Low workload: score less than or equal to 10.

Arousal and valence are represented as categorical labels.

## Dataset Splits

The official participant-level split information is used to prevent participant overlap between training and validation.

| Split | Participants | Clips |
|---|---:|---:|
| Training | 86 | 47,516 |
| Validation | 20 | 10,596 |
| Total | 106 | 58,112 |

The current release metadata provides training and validation splits. An independent test split has not been established in this preprocessing workflow.

## Preprocessing Architecture

The pipeline is organized into four main components.

### 1. Dataset Loading

- Clip reader
- Dataset adapter for downstream PyTorch integration

### 2. Dataset Preparation

- Manifest generation
- Face-crop preparation
- Clip and label organization

### 3. Statistics and Validation

- Audio availability checks
- Audio/video alignment checks
- Normalization statistics computation

### 4. Feature Extraction

- Temporal window generation
- Video feature extraction
- Audio feature extraction

## Audio Preprocessing

Audio features are extracted using a pure NumPy implementation and PyAV for decoding.

### Processing steps

1. Decode audio from the original video clips.
2. Downmix stereo audio to mono.
3. Resample audio to 16 kHz.
4. Compute the Short-Time Fourier Transform.
5. Apply a Mel filterbank.
6. Convert Mel power values into the logarithmic domain.
7. Divide the resulting spectrogram into temporal windows.

### Audio configuration

| Parameter | Value |
|---|---|
| Sampling rate | 16,000 Hz |
| Mel filters | 80 |
| FFT size | 1024 |
| Hop length | 160 samples |
| Hop duration | 10 ms |
| Window function | Hann |
| Mel scale | HTK |
| Spectrogram | Log-Mel |
| Window duration | 2 seconds |

Each approximately 6.04-second clip produces a log-Mel spectrogram of shape:

`(80, 604)`

The spectrogram is divided into three non-overlapping 2-second windows:

`(3, 80, 200)`

**Note:** The AVCAffe paper specifies 80 Mel filters, a 10 ms hop and a 1024-point FFT. Other implementation settings are preprocessing assumptions and should be documented accordingly.

## Video Preprocessing

Video preprocessing will use the released AVCAffe face crops.

The intended transformation follows the video configuration described in the paper:

| Parameter | Value |
|---|---|
| Frame sampling rate | 8 FPS |
| Window duration | 2 seconds |
| Frames per window | 16 |
| Frame resolution | 112 × 112 |
| Expected tensor shape | `3 × 16 × 112 × 112` |

The pipeline will:

1. Load the corresponding face crops.
2. Sample frames at 8 FPS.
3. Resize frames to 112 × 112.
4. Normalize pixel values.
5. Generate aligned 2-second video windows.

Video processing is currently pending completion and inspection of the face-crops download.

## Audio-Video Temporal Alignment

Audio and video features must correspond to the same time intervals.

The intended windowing strategy uses non-overlapping 2-second windows.

For window `k`, both modalities cover:

`[2k, 2k + 2)`

This ensures that audio and video inputs represent the same temporal segment.

The pipeline will also retain an audio-availability indicator (`has_audio`) to support different strategies for handling near-silent clips:

- Keep the audio as-is.
- Mask audio when unavailable.
- Drop clips based on a configurable setting.

The default handling strategy should be explicitly configured and validated before model training.

## No-Audio Investigation

An initial investigation examined the dataset's `no_audio_files.txt` metadata.

Findings:

- 18,632 clips are listed as no-audio clips.
- This represents approximately 32% of the dataset.
- Their median audio RMS is approximately 85 times lower than that of normal clips.
- Some sampled clips were completely silent.

The interpretation that these clips contain no speech is an inference rather than a confirmed dataset definition.

These clips are retained in the manifest, with an audio-availability indicator.

## Repository Structure

The current repository is organized around the following components:

```text
ml-CSD361/
│
├── configs/
│
├── notebooks/
│
├── docs/
│
├── src/
│   └── preprocessing/
│       ├── __init__.py
│       ├── manifest.py
│       ├── probe_clip.py
│       ├── audio_features.py
│       ├── test_audio.py
│       └── check_no_audio.py
│
├── .gitignore
└── README.md
```

Additional preprocessing modules will be added as the pipeline develops.

## Environment Setup

### Requirements

- Python 3.13
- NumPy
- OpenCV (headless)
- PyAV

The downloader additionally uses Requests.

### Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install numpy opencv-python-headless av requests
```

**Environment note:** Development on the original Windows machine is subject to Windows Smart App Control restrictions. PyTorch and certain compiled dependencies may be blocked. Local preprocessing therefore uses pure Python, NumPy, OpenCV and PyAV. PyTorch-dependent components should be tested in Google Colab or a suitable compute environment.

## Dataset Directory

The raw dataset is stored separately from the Git repository.

Example local directory:

```text
C:\dev\data\avcaffe\
│
├── ground_truths\
├── info\
├── videos\
│   └── shorter_segments\
│
├── face_crops\
│
└── manifest.csv
```

Keeping large datasets and generated features outside the Git repository prevents unnecessary repository growth.

**Important:** Raw dataset files, participant-level information and API credentials must not be committed to GitHub.

## Current Implementation Status

Project status as of November 11, 2026:

| Component | Status |
|---|---|
| GitHub repository setup | Completed |
| Dataset access and downloader | Completed |
| Ground-truth download | Completed |
| Metadata download | Completed |
| Video clip download | Completed |
| Face-crops download | In progress |
| Dataset manifest | Completed |
| Audio feature extraction | Core implementation completed |
| Audio validation | Completed |
| No-audio investigation | Completed |
| Video feature extraction | Pending |
| Face-crop alignment | Pending |
| Audio-video window alignment | Pending |
| Global normalization statistics | Pending |
| Dataset export/cache | Pending |
| PyTorch Dataset adapter | Pending |
| Sanity-check visualizations | Pending |
| Documentation and team handoff | Pending |

## Planned Work

The remaining development work is organized as follows:

1. Complete the face-crops download.
2. Inspect the face-crops directory structure.
3. Extend the manifest with face-crop paths.
4. Validate correspondence between video clips and face crops.
5. Implement video frame sampling and preprocessing.
6. Implement synchronized audio-video window generation.
7. Compute global normalization statistics using the training split only.
8. Implement feature caching and dataset export.
9. Develop a PyTorch Dataset adapter and test it in Colab or on the college cluster.
10. Perform sanity checks on features, labels, alignment and participant splits.
11. Complete technical documentation and team handoff.

## Important Considerations

- Training and validation must remain participant-disjoint.
- Normalization statistics must be computed using training data only.
- No-audio clips should not be silently discarded.
- Audio and video windows must use consistent timestamps.
- Dataset licensing and redistribution restrictions must be respected.
- Implementation assumptions should be distinguished from configurations explicitly specified by the original paper.

## References

1. Sarkar, Posen and Etemad (2023). *AVCAffe: A Large Scale Audio-Visual Dataset of Cognitive Load and Affect for Remote Work*. AAAI 2023.
2. [AVCAffe Dataset – Borealis Dataverse](https://doi.org/10.5683/SP3/PSWY62)
3. [AVCAffe Original Repository](https://github.com/pritamqu/AVCAffe)

---

**Project:** Affect-Aware Dynamic UI  
**Repository:** `lc1471/ml-CSD361`  
**Focus:** Multimodal dataset preprocessing  
**Primary modalities:** Audio and facial video
