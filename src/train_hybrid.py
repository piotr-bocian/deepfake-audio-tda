import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


mfcc_train = pd.read_csv("../data/la_train_mfcc.csv")
tda_train = pd.read_csv("../data/la_train_tda.csv")

mfcc_dev = pd.read_csv("../data/la_dev_mfcc.csv")
tda_dev = pd.read_csv("../data/la_dev_tda.csv")

assert mfcc_train["filename"].is_unique
assert tda_train["filename"].is_unique
assert mfcc_dev["filename"].is_unique
assert tda_dev["filename"].is_unique

# łączymy po filename
train_df = mfcc_train.merge(tda_train, on="filename")
dev_df = mfcc_dev.merge(tda_dev, on="filename")

assert len(train_df) == len(mfcc_train)
assert len(dev_df) == len(mfcc_dev)

assert (train_df["label_x"] == train_df["label_y"]).all()
assert (dev_df["label_x"] == dev_df["label_y"]).all()



train_df["label"] = train_df["label_x"]
dev_df["label"] = dev_df["label_x"]

FEATURE_COLS = [f"f{i}" for i in range(26)] + [f"t{i}" for i in range(13)]

def encode(y):
    return y.map({"bonafide": 0, "spoof": 1})

X_train = train_df[FEATURE_COLS]
y_train = encode(train_df["label"])

X_dev = dev_df[FEATURE_COLS]
y_dev = encode(dev_df["label"])

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=42)),
])

model.fit(X_train, y_train)

def compute_eer(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])

y_prob = model.predict_proba(X_dev)[:, 1]
y_pred = model.predict(X_dev)

print("HYBRID AUC:", roc_auc_score(y_dev, y_prob))
print("HYBRID ACC:", accuracy_score(y_dev, y_pred))
print("HYBRID EER:", compute_eer(y_dev, y_prob))