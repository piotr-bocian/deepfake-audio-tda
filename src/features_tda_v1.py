from __future__ import annotations

import numpy as np
import librosa
from ripser import ripser


def extract_mfcc_frames(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T.astype(np.float32)


def subsample_point_cloud(
    points: np.ndarray,
    max_points: int = 80,
) -> np.ndarray:
    n = len(points)
    if n <= max_points:
        return points

    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return points[idx]


def finite_lifetimes(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.array([], dtype=np.float32)

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    mask = np.isfinite(deaths)
    lifetimes = deaths[mask] - births[mask]
    return lifetimes.astype(np.float32)


def diagram_stats(diagram: np.ndarray) -> list[float]:
    lifetimes = finite_lifetimes(diagram)

    if lifetimes.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(np.mean(lifetimes)),
        float(np.std(lifetimes)),
        float(np.max(lifetimes)),
        float(len(lifetimes)),
    ]


def extract_tda_features_from_mfcc_v1(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    max_points: int = 80,
    maxdim: int = 1,
) -> np.ndarray:
    points = extract_mfcc_frames(y, sr, n_mfcc=n_mfcc)
    points = subsample_point_cloud(points, max_points=max_points)

    points = (points - points.mean(axis=0, keepdims=True)) / (
        points.std(axis=0, keepdims=True) + 1e-8
    )

    dgms = ripser(points, maxdim=maxdim)["dgms"]

    h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2), dtype=np.float32)
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2), dtype=np.float32)

    feats = diagram_stats(h0) + diagram_stats(h1)
    return np.array(feats, dtype=np.float32)