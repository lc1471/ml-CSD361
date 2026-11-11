import av
import numpy as np

SR = 16000
N_FFT = 1024
HOP = 160          # 10 ms
N_MELS = 80
WIN = 200          # frames per 2 s window -> 80 x 200


def _as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def load_audio(path, sr=SR):
    """Decode a clip's audio to mono float32 at `sr`. Empty array if no audio stream."""
    chunks = []
    with av.open(path) as c:
        if not c.streams.audio:
            return np.zeros(0, np.float32)
        rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in c.decode(audio=0):
            for f in _as_list(rs.resample(frame)):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in _as_list(rs.resample(None)):
            chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)))
    return fb


_FB = mel_filterbank()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def log_mel(wave):
    """float32 waveform @16 kHz -> log-mel (80, T), 10 ms hop, 1024 FFT (center-padded)."""
    if len(wave) < N_FFT:
        wave = np.pad(wave, (0, N_FFT - len(wave)))
    x = np.pad(wave, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * _WINDOW, axis=1)) ** 2
    mel = spec @ _FB.T
    return np.log(mel + 1e-6).T.astype(np.float32)


def windows(mel, n=WIN):
    """Non-overlapping 2 s windows (80 x 200) from a clip's log-mel."""
    return [mel[:, i:i + n] for i in range(0, mel.shape[1] - n + 1, n)]
