import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def encode_labels(series: pd.Series) -> pd.Series:
    return series.map({"bonafide": 0, "spoof": 1})


def compute_eer(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


if __name__ == "__main__":
    train_df = pd.read_csv("../data/la_train_tda_v2h0.csv")
    dev_df = pd.read_csv("../data/la_dev_tda_v2h0.csv")

    feature_cols = [c for c in train_df.columns if c.startswith("t")]

    X_train = train_df[feature_cols]
    y_train = encode_labels(train_df["label"])

    X_dev = dev_df[feature_cols]
    y_dev = encode_labels(dev_df["label"])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=42)),
    ])

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_dev_pred = model.predict(X_dev)
    y_dev_prob = model.predict_proba(X_dev)[:, 1]

    print("=== TDA v2 + LOGREG ===")
    print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Dev accuracy:   {accuracy_score(y_dev, y_dev_pred):.4f}")
    print(f"Dev ROC-AUC:    {roc_auc_score(y_dev, y_dev_prob):.4f}")
    print(f"Dev EER:        {compute_eer(y_dev, y_dev_prob):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_dev, y_dev_pred))

    print("\nClassification report:")
    print(classification_report(y_dev, y_dev_pred, target_names=["bonafide", "spoof"]))