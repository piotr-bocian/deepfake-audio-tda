import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np

FEATURE_COLS = [f"f{i}" for i in range(26)]


def encode_labels(series: pd.Series) -> pd.Series:
    return series.map({"bonafide": 0, "spoof": 1})


if __name__ == "__main__":
    train_df = pd.read_csv("../data/la_train_mfcc.csv")
    dev_df = pd.read_csv("../data/la_dev_mfcc.csv")

    X_train = train_df[FEATURE_COLS]
    y_train = encode_labels(train_df["label"])

    X_dev = dev_df[FEATURE_COLS]
    y_dev = encode_labels(dev_df["label"])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]

    acc = accuracy_score(y_dev, y_pred)
    auc = roc_auc_score(y_dev, y_prob)
    cm = confusion_matrix(y_dev, y_pred)

    fpr, tpr, thresholds = roc_curve(y_dev, y_prob)
    fnr = 1 - tpr

    print("Accuracy:", acc)
    print("ROC-AUC :", auc)
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_dev, y_pred, target_names=["bonafide", "spoof"]))

    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    print("EER:", eer)

    y_train_pred = model.predict(X_train)
    print("Train accuracy:", accuracy_score(y_train, y_train_pred))

    print("\n=== SHUFFLE TEST x10 ===")
    shuffle_aucs = []

    for seed in range(10):
        rng = np.random.default_rng(seed)
        y_train_shuffled = rng.permutation(y_train.to_numpy())

        model_shuffled = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ])

        model_shuffled.fit(X_train, y_train_shuffled)
        y_prob_shuffled = model_shuffled.predict_proba(X_dev)[:, 1]
        auc_shuffled = roc_auc_score(y_dev, y_prob_shuffled)
        shuffle_aucs.append(auc_shuffled)
        print(f"seed={seed}: AUC={auc_shuffled:.4f}")

    print(f"mean shuffled AUC: {np.mean(shuffle_aucs):.4f}")
    print(f"std shuffled AUC:  {np.std(shuffle_aucs):.4f}")

    train_speakers = set(train_df["speaker_id"])
    dev_speakers = set(dev_df["speaker_id"])
    overlap = train_speakers & dev_speakers

    print("Train speakers:", len(train_speakers))
    print("Dev speakers:", len(dev_speakers))
    print("Speaker overlap:", len(overlap))
    print("Example overlap:", list(sorted(overlap))[:10])