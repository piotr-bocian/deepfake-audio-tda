from src.paths import (
    LA_TRAIN_DIR,
    LA_DEV_DIR,
    LA_TRAIN_PROTOCOL,
    LA_DEV_PROTOCOL,
)
from src.make_features import build_feature_dataframe

train_df = build_feature_dataframe(
    LA_TRAIN_DIR,
    LA_TRAIN_PROTOCOL,
    n_mfcc=13,
    samples_per_class=200,
    random_state=42,
)

dev_df = build_feature_dataframe(
    LA_DEV_DIR,
    LA_DEV_PROTOCOL,
    n_mfcc=13,
    samples_per_class=200,
    random_state=42,
)

print("TRAIN")
print(train_df.head())
print(train_df["label"].value_counts())
print(train_df.shape)

print("\nDEV")
print(dev_df.head())
print(dev_df["label"].value_counts())
print(dev_df.shape)

train_df.to_csv("../data/la_train_mfcc.csv", index=False)
dev_df.to_csv("../data/la_dev_mfcc.csv", index=False)
