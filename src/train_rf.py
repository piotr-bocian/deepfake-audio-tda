import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


FEATURE_COLS = [f"f{i}" for i in range(26)]


def encode_labels(series: pd.Series) -> pd.Series:
    return series.map({"bonafide": 0, "spoof": 1})


def compute_eer(y_true, y_prob) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


if __name__ == "__main__":
    train_df = pd.read_csv("../data/la_train_mfcc.csv")
    dev_df = pd.read_csv("../data/la_dev_mfcc.csv")

    X_train = train_df[FEATURE_COLS]
    y_train = encode_labels(train_df["label"])

    X_dev = dev_df[FEATURE_COLS]
    y_dev = encode_labels(dev_df["label"])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # TRAIN
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # DEV
    y_dev_pred = model.predict(X_dev)
    y_dev_prob = model.predict_proba(X_dev)[:, 1]

    dev_acc = accuracy_score(y_dev, y_dev_pred)
    dev_auc = roc_auc_score(y_dev, y_dev_prob)
    dev_eer = compute_eer(y_dev, y_dev_prob)
    cm = confusion_matrix(y_dev, y_dev_pred)

    print("=== RANDOM FOREST ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Dev accuracy:   {dev_acc:.4f}")
    print(f"Dev ROC-AUC:    {dev_auc:.4f}")
    print(f"Dev EER:        {dev_eer:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_dev, y_dev_pred, target_names=["bonafide", "spoof"]))