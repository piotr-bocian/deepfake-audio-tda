from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def add_white_noise(y: np.ndarray, snr_db: float, random_state: int = 42) -> np.ndarray:
    """
    Add white Gaussian noise to a waveform at a target SNR (in dB).
    """
    if y.size == 0:
        return y

    rng = np.random.default_rng(random_state)

    signal_power = np.mean(y.astype(np.float64) ** 2)
    if signal_power <= 0:
        return y.copy()

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=y.shape)

    y_noisy = y.astype(np.float64) + noise

    peak = np.max(np.abs(y_noisy))
    if peak > 1.0:
        y_noisy = y_noisy / peak

    return y_noisy.astype(np.float32)


def compress_mp3(y: np.ndarray, sr: int, bitrate: str = "128k") -> np.ndarray:
    """
    Compress audio to MP3 using ffmpeg, then decode back to waveform.
    bitrate examples: '128k', '64k', '32k'
    """
    if y.size == 0:
        return y

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        wav_path = tmpdir / "input.wav"
        mp3_path = tmpdir / "compressed.mp3"

        sf.write(wav_path, y, sr)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(mp3_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg was not found in PATH. Install ffmpeg and make sure it is available."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg failed during MP3 compression. stderr: {e.stderr.decode(errors='ignore')}"
            ) from e

        y_mp3, _ = librosa.load(mp3_path, sr=sr, mono=True)

    return y_mp3.astype(np.float32)


def apply_degradation(
    y: np.ndarray,
    sr: int,
    degradation_type: str | None = None,
    degradation_value: float | str | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply a selected degradation to audio.

    Supported:
    - None
    - 'white_noise' with degradation_value = target SNR in dB
    - 'mp3' with degradation_value = bitrate string, e.g. '128k'
    """
    if degradation_type is None:
        return y

    if degradation_type == "white_noise":
        if degradation_value is None:
            raise ValueError("degradation_value must be provided for white_noise")
        return add_white_noise(y, snr_db=float(degradation_value), random_state=random_state)

    if degradation_type == "mp3":
        if degradation_value is None:
            raise ValueError("degradation_value must be provided for mp3")
        return compress_mp3(y, sr=sr, bitrate=str(degradation_value))

    raise ValueError(f"Unsupported degradation_type: {degradation_type}")