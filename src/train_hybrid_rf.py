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


MFCC_COLS = [f"f{i}" for i in range(26)]
TDA_COLS = [f"t{i}" for i in range(13)]
FEATURE_COLS = MFCC_COLS + TDA_COLS


def encode_labels(series: pd.Series) -> pd.Series:
    return series.map({"bonafide": 0, "spoof": 1})


def compute_eer(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


if __name__ == "__main__":
    mfcc_train = pd.read_csv("../data/la_train_mfcc.csv")
    mfcc_dev = pd.read_csv("../data/la_dev_mfcc.csv")

    tda_train = pd.read_csv("../data/la_train_tda_v2.csv")
    tda_dev = pd.read_csv("../data/la_dev_tda_v2.csv")

    # Bierzemy tylko potrzebne kolumny z TDA, żeby nie dublować metadanych
    tda_train_small = tda_train[["filename"] + TDA_COLS]
    tda_dev_small = tda_dev[["filename"] + TDA_COLS]

    train_df = mfcc_train.merge(tda_train_small, on="filename", how="inner")
    dev_df = mfcc_dev.merge(tda_dev_small, on="filename", how="inner")

    print("Train merged shape:", train_df.shape)
    print("Dev merged shape:", dev_df.shape)

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

    y_train_pred = model.predict(X_train)
    y_dev_pred = model.predict(X_dev)
    y_dev_prob = model.predict_proba(X_dev)[:, 1]

    print("\n=== HYBRID + RANDOM FOREST ===")
    print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Dev accuracy:   {accuracy_score(y_dev, y_dev_pred):.4f}")
    print(f"Dev ROC-AUC:    {roc_auc_score(y_dev, y_dev_prob):.4f}")
    print(f"Dev EER:        {compute_eer(y_dev, y_dev_prob):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_dev, y_dev_pred))

    print("\nClassification report:")
    print(classification_report(y_dev, y_dev_pred, target_names=["bonafide", "spoof"]))