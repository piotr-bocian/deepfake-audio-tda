import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


mfcc_train = pd.read_csv("../data/la_train_mfcc.csv")
tda_train = pd.read_csv("../data/la_train_tda.csv")

mfcc_dev = pd.read_csv("../data/la_dev_mfcc.csv")
tda_dev = pd.read_csv("../data/la_dev_tda.csv")

# łączymy po filename
train_df = mfcc_train.merge(tda_train, on="filename")
dev_df = mfcc_dev.merge(tda_dev, on="filename")

FEATURE_COLS = [f"f{i}" for i in range(26)] + [f"t{i}" for i in range(8)]

def encode(y):
    return y.map({"bonafide": 0, "spoof": 1})

X_train = train_df[FEATURE_COLS]
y_train = encode(train_df["label_x"])

X_dev = dev_df[FEATURE_COLS]
y_dev = encode(dev_df["label_x"])

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=42)),
])

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_dev)[:, 1]
y_pred = model.predict(X_dev)

print("HYBRID AUC:", roc_auc_score(y_dev, y_prob))
print("HYBRID ACC:", accuracy_score(y_dev, y_pred))