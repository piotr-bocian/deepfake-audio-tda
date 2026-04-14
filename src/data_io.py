from pathlib import Path
import librosa
import numpy as np


def load_audio(
    path: str | Path,
    sr: int = 16000,
    mono: bool = True,
    duration: float | None = 3.0,
) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=mono, duration=duration)
    return y, sr