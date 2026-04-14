from __future__ import annotations

import numpy as np
import librosa
from ripser import ripser
from sklearn.decomposition import PCA


def extract_mfcc_frames(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    return mfcc.T.astype(np.float32)  # (T, n_mfcc)


def subsample_point_cloud(
    points: np.ndarray,
    max_points: int = 80,
) -> np.ndarray:
    n = len(points)
    if n <= max_points:
        return points
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return points[idx]


def normalize_points(points: np.ndarray) -> np.ndarray:
    return (points - points.mean(axis=0, keepdims=True)) / (
        points.std(axis=0, keepdims=True) + 1e-8
    )


def maybe_apply_pca(points: np.ndarray, n_components: int | None = None) -> np.ndarray:
    if n_components is None:
        return points
    n_components = min(n_components, points.shape[0], points.shape[1])
    if n_components < 2:
        return points
    return PCA(n_components=n_components, random_state=42).fit_transform(points)


def finite_lifetimes(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.array([], dtype=np.float32)

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    mask = np.isfinite(deaths)
    lifetimes = deaths[mask] - births[mask]
    lifetimes = lifetimes[lifetimes > 0]
    return lifetimes.astype(np.float32)


def persistence_entropy(lifetimes: np.ndarray) -> float:
    if lifetimes.size == 0:
        return 0.0
    s = lifetimes.sum()
    if s <= 0:
        return 0.0
    p = lifetimes / s
    return float(-(p * np.log(p + 1e-12)).sum())


def topk_lifetimes(lifetimes: np.ndarray, k: int = 3) -> list[float]:
    if lifetimes.size == 0:
        return [0.0] * k
    vals = np.sort(lifetimes)[::-1][:k]
    vals = vals.tolist()
    while len(vals) < k:
        vals.append(0.0)
    return [float(v) for v in vals]


def diagram_stats(diagram: np.ndarray) -> list[float]:
    lifetimes = finite_lifetimes(diagram)

    if lifetimes.size == 0:
        # 10 cech na diagram
        return [0.0] * 10

    q25, q50, q75, q90 = np.quantile(lifetimes, [0.25, 0.5, 0.75, 0.9])

    feats = [
        float(np.mean(lifetimes)),
        float(np.std(lifetimes)),
        float(np.max(lifetimes)),
        float(np.sum(lifetimes)),
        float(q25),
        float(q50),
        float(q75),
        float(q90),
        float(len(lifetimes)),
        float(persistence_entropy(lifetimes)),
    ]
    feats.extend(topk_lifetimes(lifetimes, k=3))
    # razem 13 cech
    return feats


def extract_tda_features_from_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    max_points: int = 80,
    maxdim: int = 1,
    pca_components: int | None = 8,
    use_h0: bool = True,
    use_h1: bool = True,
) -> np.ndarray:
    points = extract_mfcc_frames(y, sr, n_mfcc=n_mfcc)
    points = subsample_point_cloud(points, max_points=max_points)
    points = normalize_points(points)
    points = maybe_apply_pca(points, n_components=pca_components)

    dgms = ripser(points, maxdim=maxdim)["dgms"]

    feats: list[float] = []

    if use_h0:
        h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2), dtype=np.float32)
        feats.extend(diagram_stats(h0))

    if use_h1:
        h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2), dtype=np.float32)
        feats.extend(diagram_stats(h1))

    return np.array(feats, dtype=np.float32)