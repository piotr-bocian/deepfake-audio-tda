from __future__ import annotations

import numpy as np


def add_white_noise(y: np.ndarray, snr_db: float, random_state: int = 42) -> np.ndarray:
    """
    Add white Gaussian noise to a waveform at a target SNR (in dB).

    Parameters
    ----------
    y : np.ndarray
        Input waveform.
    snr_db : float
        Target signal-to-noise ratio in dB.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy waveform with the same shape as input.
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

    # avoid clipping explosions
    peak = np.max(np.abs(y_noisy))
    if peak > 1.0:
        y_noisy = y_noisy / peak

    return y_noisy.astype(np.float32)


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
    - degradation_type=None -> no change
    - degradation_type='white_noise' with degradation_value=<snr_db>

    Parameters
    ----------
    y : np.ndarray
        Input waveform.
    sr : int
        Sampling rate. Included for future extensibility.
    degradation_type : str | None
        Type of degradation.
    degradation_value : float | str | None
        Parameter of degradation.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Degraded waveform.
    """
    if degradation_type is None:
        return y

    if degradation_type == "white_noise":
        if degradation_value is None:
            raise ValueError("degradation_value must be provided for white_noise")
        return add_white_noise(y, snr_db=float(degradation_value), random_state=random_state)

    raise ValueError(f"Unsupported degradation_type: {degradation_type}")