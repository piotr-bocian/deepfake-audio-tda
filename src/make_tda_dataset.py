from src.paths import (
    LA_TRAIN_DIR,
    LA_DEV_DIR,
    LA_TRAIN_PROTOCOL,
    LA_DEV_PROTOCOL,
)
from src.make_tda_features import build_tda_feature_dataframe


if __name__ == "__main__":
    train_df = build_tda_feature_dataframe(
        LA_TRAIN_DIR,
        LA_TRAIN_PROTOCOL,
        n_mfcc=13,
        max_points=80,
        pca_components=None,
        use_h0=True,
        use_h1=True,
        samples_per_class=200,
        random_state=42,
    )

    dev_df = build_tda_feature_dataframe(
        LA_DEV_DIR,
        LA_DEV_PROTOCOL,
        n_mfcc=13,
        max_points=80,
        pca_components=None,
        use_h0=True,
        use_h1=True,
        samples_per_class=None,
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

    train_df.to_csv("../data/la_train_tda.csv", index=False)
    dev_df.to_csv("../data/la_dev_tda.csv", index=False)