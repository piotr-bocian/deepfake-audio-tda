from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.common.metrics import encode_labels, compute_eer


MFCC_COLS = [f"f{i}" for i in range(26)]
TDA_COLS = [f"t{i}" for i in range(26)]


def load_hybrid_frames_robust(
    mfcc_train_csv: str | Path,
    tda_train_csv: str | Path,
    mfcc_dev_csv: str | Path,
    tda_dev_csv: str | Path,
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


def run_hybrid_rf_robust(
    mfcc_train_csv: str | Path,
    tda_train_csv: str | Path,
    mfcc_dev_csv: str | Path,
    tda_dev_csv: str | Path,
    rf_seed: int = 42,
) -> dict:
    train_df, dev_df = load_hybrid_frames_robust(
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