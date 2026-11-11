import os, sys
import numpy as np
from audio_features import load_audio, log_mel, windows

DATA = r"C:\dev\data\avcaffe"
for rel in sys.argv[1:]:
    w = load_audio(os.path.join(DATA, rel))
    mel = log_mel(w)
    ws = windows(mel)
    print(os.path.basename(rel))
    print("  samples:", len(w), "| seconds:", round(len(w) / 16000, 2))
    print("  rms: %.5f | peak: %.5f" % (np.sqrt(np.mean(w ** 2)) if len(w) else 0, np.abs(w).max() if len(w) else 0))
    print("  log-mel:", mel.shape, "| windows:", len(ws), "x", ws[0].shape if ws else None)
    print("  log-mel mean/min/max: %.2f / %.2f / %.2f" % (mel.mean(), mel.min(), mel.max()))
