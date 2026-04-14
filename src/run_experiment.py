from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("../data")
RESULTS_PATH = DATA_DIR / "results_priority_a.csv"

MFCC_TRAIN = DATA_DIR / "la_train_mfcc.csv"
MFCC_DEV = DATA_DIR / "la_dev_mfcc.csv"
TDA_TRAIN = DATA_DIR / "la_train_tda.csv"
TDA_DEV = DATA_DIR / "la_dev_tda.csv"


MFCC_COLS = [f"f{i}" for i in range(26)]
TDA_COLS = [f"t{i}" for i in range(13)]


def encode_labels(series: pd.Series) -> pd.Series:
    return series.map({"bonafide": 0, "spoof": 1})


def compute_eer(y_true: pd.Series, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


def run_mfcc_rf(rf_seed: int = 42) -> dict:
    train_df = pd.read_csv(MFCC_TRAIN)
    dev_df = pd.read_csv(MFCC_DEV)

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


def run_tda_logreg(logreg_seed: int = 42) -> dict:
    train_df = pd.read_csv(TDA_TRAIN)
    dev_df = pd.read_csv(TDA_DEV)

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


def load_hybrid_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    mfcc_train = pd.read_csv(MFCC_TRAIN)
    mfcc_dev = pd.read_csv(MFCC_DEV)

    tda_train = pd.read_csv(TDA_TRAIN)
    tda_dev = pd.read_csv(TDA_DEV)

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


def run_hybrid_rf(rf_seed: int = 42) -> dict:
    train_df, dev_df = load_hybrid_frames()

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
    seeds = [0, 1, 2, 3, 4]

    rows = []

    for seed in seeds:
        print(f"Running experiments for seed={seed}")

        rows.append({
            "sample_size": "current",
            "seed": seed,
            **run_mfcc_rf(rf_seed=seed),
        })

        rows.append({
            "sample_size": "current",
            "seed": seed,
            **run_tda_logreg(logreg_seed=seed),
        })

        rows.append({
            "sample_size": "current",
            "seed": seed,
            **run_hybrid_rf(rf_seed=seed),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\\nSaved results to:", RESULTS_PATH)
    print(results_df)


if __name__ == "__main__":
    main()