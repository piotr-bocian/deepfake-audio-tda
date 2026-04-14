from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.paths import (
    LA_TRAIN_DIR,
    LA_DEV_DIR,
    LA_TRAIN_PROTOCOL,
    LA_DEV_PROTOCOL,
)
from src.make_features import build_feature_dataframe
from src.make_tda_features import build_tda_feature_dataframe


DATA_DIR = Path("../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = DATA_DIR / "results_priority_a_all.csv"

MFCC_COLS = [f"f{i}" for i in range(26)]
TDA_COLS = [f"t{i}" for i in range(26)]  # 26 features = 13 for H0 + 13 for H1


def encode_labels(series: pd.Series) -> pd.Series:
    return series.map({"bonafide": 0, "spoof": 1})


def compute_eer(y_true: pd.Series, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


def build_train_mfcc(sample_size: int, seed: int) -> Path:
    out_path = DATA_DIR / f"la_train_mfcc_s{sample_size}_seed{seed}.csv"

    df = build_feature_dataframe(
        LA_TRAIN_DIR,
        LA_TRAIN_PROTOCOL,
        n_mfcc=13,
        samples_per_class=sample_size,
        random_state=seed,
    )
    df.to_csv(out_path, index=False)
    return out_path


def build_train_tda(sample_size: int, seed: int) -> Path:
    out_path = DATA_DIR / f"la_train_tda_s{sample_size}_seed{seed}.csv"

    df = build_tda_feature_dataframe(
        LA_TRAIN_DIR,
        LA_TRAIN_PROTOCOL,
        n_mfcc=13,
        max_points=80,
        pca_components=None,
        use_h0=True,
        use_h1=True,
        samples_per_class=sample_size,
        random_state=seed,
    )
    df.to_csv(out_path, index=False)
    return out_path


def build_dev_mfcc_once() -> Path:
    out_path = DATA_DIR / "la_dev_mfcc_full.csv"
    if out_path.exists():
        return out_path

    df = build_feature_dataframe(
        LA_DEV_DIR,
        LA_DEV_PROTOCOL,
        n_mfcc=13,
        samples_per_class=None,
        random_state=42,
    )
    df.to_csv(out_path, index=False)
    return out_path


def build_dev_tda_once() -> Path:
    out_path = DATA_DIR / "la_dev_tda_full.csv"
    if out_path.exists():
        return out_path

    df = build_tda_feature_dataframe(
        LA_DEV_DIR,
        LA_DEV_PROTOCOL,
        n_mfcc=13,
        max_points=80,
        pca_components=None,
        use_h0=True,
        use_h1=True,
        samples_per_class=None,
        random_state=42,
    )
    df.to_csv(out_path, index=False)
    return out_path


def run_mfcc_rf(train_csv: Path, dev_csv: Path, rf_seed: int) -> dict:
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)

    X_train = train_df[MFCC_COLS]
    y_train = encode_labels(train_df["label"])

    X_dev = dev_df[MFCC_COLS]
    y_dev = encode_labels(dev_df["label"])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=rf_seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]

    return {
        "model": "mfcc_rf",
        "auc": roc_auc_score(y_dev, y_prob),
        "eer": compute_eer(y_dev, y_prob),
        "acc": accuracy_score(y_dev, y_pred),
    }


def run_tda_logreg(train_csv: Path, dev_csv: Path, logreg_seed: int) -> dict:
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)

    X_train = train_df[TDA_COLS]
    y_train = encode_labels(train_df["label"])

    X_dev = dev_df[TDA_COLS]
    y_dev = encode_labels(dev_df["label"])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=logreg_seed)),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]

    return {
        "model": "tda_logreg",
        "auc": roc_auc_score(y_dev, y_prob),
        "eer": compute_eer(y_dev, y_prob),
        "acc": accuracy_score(y_dev, y_pred),
    }


def load_hybrid_frames(
    mfcc_train_csv: Path,
    tda_train_csv: Path,
    mfcc_dev_csv: Path,
    tda_dev_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mfcc_train = pd.read_csv(mfcc_train_csv)
    mfcc_dev = pd.read_csv(mfcc_dev_csv)

    tda_train = pd.read_csv(tda_train_csv)
    tda_dev = pd.read_csv(tda_dev_csv)

    assert mfcc_train["filename"].is_unique
    assert mfcc_dev["filename"].is_unique
    assert tda_train["filename"].is_unique
    assert tda_dev["filename"].is_unique

    tda_train_small = tda_train[["filename"] + TDA_COLS]
    tda_dev_small = tda_dev[["filename"] + TDA_COLS]

    train_df = mfcc_train.merge(tda_train_small, on="filename", how="inner")
    dev_df = mfcc_dev.merge(tda_dev_small, on="filename", how="inner")

    assert len(train_df) == len(mfcc_train)
    assert len(dev_df) == len(mfcc_dev)

    return train_df, dev_df


def run_hybrid_rf(
    mfcc_train_csv: Path,
    tda_train_csv: Path,
    mfcc_dev_csv: Path,
    tda_dev_csv: Path,
    rf_seed: int,
) -> dict:
    train_df, dev_df = load_hybrid_frames(
        mfcc_train_csv, tda_train_csv, mfcc_dev_csv, tda_dev_csv
    )

    feature_cols = MFCC_COLS + TDA_COLS

    X_train = train_df[feature_cols]
    y_train = encode_labels(train_df["label"])

    X_dev = dev_df[feature_cols]
    y_dev = encode_labels(dev_df["label"])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=rf_seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]

    return {
        "model": "hybrid_rf",
        "auc": roc_auc_score(y_dev, y_prob),
        "eer": compute_eer(y_dev, y_prob),
        "acc": accuracy_score(y_dev, y_pred),
    }


def main() -> None:
    sample_sizes = [200, 500, 1000, 2000]
    seeds = [0, 1, 2, 3, 4]

    rows: list[dict] = []

    print("Building full dev feature sets once...")
    mfcc_dev_csv = build_dev_mfcc_once()
    tda_dev_csv = build_dev_tda_once()

    for sample_size in sample_sizes:
        for seed in seeds:
            print(f"Running sample_size={sample_size}, seed={seed}")

            mfcc_train_csv = build_train_mfcc(sample_size=sample_size, seed=seed)
            tda_train_csv = build_train_tda(sample_size=sample_size, seed=seed)

            rows.append({
                "sample_size": sample_size,
                "seed": seed,
                **run_mfcc_rf(
                    train_csv=mfcc_train_csv,
                    dev_csv=mfcc_dev_csv,
                    rf_seed=seed,
                ),
            })

            rows.append({
                "sample_size": sample_size,
                "seed": seed,
                **run_tda_logreg(
                    train_csv=tda_train_csv,
                    dev_csv=tda_dev_csv,
                    logreg_seed=seed,
                ),
            })

            rows.append({
                "sample_size": sample_size,
                "seed": seed,
                **run_hybrid_rf(
                    mfcc_train_csv=mfcc_train_csv,
                    tda_train_csv=tda_train_csv,
                    mfcc_dev_csv=mfcc_dev_csv,
                    tda_dev_csv=tda_dev_csv,
                    rf_seed=seed,
                ),
            })

            results_df = pd.DataFrame(rows)
            results_df.to_csv(RESULTS_PATH, index=False)

    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()