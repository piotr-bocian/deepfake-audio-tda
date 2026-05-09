from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.common.metrics import encode_labels, compute_eer


MFCC_COLS = [f"f{i}" for i in range(26)]


def run_mfcc_rf_robust(
    train_csv: str | Path,
    dev_csv: str | Path,
    rf_seed: int = 42,
) -> dict:
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