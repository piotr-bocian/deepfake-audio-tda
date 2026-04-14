from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from src.paths import (
    DATA_DIR,
    LA_TRAIN_DIR,
    LA_DEV_DIR,
    LA_TRAIN_PROTOCOL,
    LA_DEV_PROTOCOL,
)
from src.make_features import build_feature_dataframe
from src.make_tda_features_v1 import build_tda_v1_feature_dataframe


RESULTS_DIR = DATA_DIR / "thesis_outputs"
CACHE_DIR = DATA_DIR / "thesis_cache"

SAMPLE_SIZES = [200, 500, 1000, 2000]
MFCC_COLS = [f"f{i}" for i in range(26)]
TDA_V1_COLS = [f"t{i}" for i in range(8)]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def encode_labels(series: pd.Series) -> pd.Series:
    return series.map({"bonafide": 0, "spoof": 1})


def compute_eer(y_true: pd.Series, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


def evaluate_logreg(X_train, y_train, X_dev, y_dev) -> dict[str, float]:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]
    return {
        "accuracy": accuracy_score(y_dev, y_pred),
        "roc_auc": roc_auc_score(y_dev, y_prob),
        "eer": compute_eer(y_dev, y_prob),
    }


def evaluate_rf(X_train, y_train, X_dev, y_dev) -> dict[str, float]:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]
    return {
        "accuracy": accuracy_score(y_dev, y_pred),
        "roc_auc": roc_auc_score(y_dev, y_prob),
        "eer": compute_eer(y_dev, y_prob),
    }


def cache_paths(sample_size: int) -> dict[str, Path]:
    base = CACHE_DIR / f"n_{sample_size}"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "mfcc_train": base / "la_train_mfcc.csv",
        "mfcc_dev": base / "la_dev_mfcc.csv",
        "tda_train": base / "la_train_tda_v1.csv",
        "tda_dev": base / "la_dev_tda_v1.csv",
    }


def build_or_load_mfcc(sample_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = cache_paths(sample_size)

    if paths["mfcc_train"].exists() and paths["mfcc_dev"].exists():
        train_df = pd.read_csv(paths["mfcc_train"])
        dev_df = pd.read_csv(paths["mfcc_dev"])
        return train_df, dev_df

    train_df = build_feature_dataframe(
        LA_TRAIN_DIR,
        LA_TRAIN_PROTOCOL,
        n_mfcc=13,
        samples_per_class=sample_size,
        random_state=42,
    )
    dev_df = build_feature_dataframe(
        LA_DEV_DIR,
        LA_DEV_PROTOCOL,
        n_mfcc=13,
        samples_per_class=sample_size,
        random_state=42,
    )

    train_df.to_csv(paths["mfcc_train"], index=False)
    dev_df.to_csv(paths["mfcc_dev"], index=False)
    return train_df, dev_df


def build_or_load_tda_v1(sample_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = cache_paths(sample_size)

    if paths["tda_train"].exists() and paths["tda_dev"].exists():
        train_df = pd.read_csv(paths["tda_train"])
        dev_df = pd.read_csv(paths["tda_dev"])
        return train_df, dev_df

    train_df = build_tda_v1_feature_dataframe(
        LA_TRAIN_DIR,
        LA_TRAIN_PROTOCOL,
        n_mfcc=13,
        max_points=80,
        samples_per_class=sample_size,
        random_state=42,
    )
    dev_df = build_tda_v1_feature_dataframe(
        LA_DEV_DIR,
        LA_DEV_PROTOCOL,
        n_mfcc=13,
        max_points=80,
        samples_per_class=sample_size,
        random_state=42,
    )

    train_df.to_csv(paths["tda_train"], index=False)
    dev_df.to_csv(paths["tda_dev"], index=False)
    return train_df, dev_df


def run_all_models_for_sample_size(sample_size: int) -> list[dict]:
    mfcc_train, mfcc_dev = build_or_load_mfcc(sample_size)
    tda_train, tda_dev = build_or_load_tda_v1(sample_size)

    y_train_mfcc = encode_labels(mfcc_train["label"])
    y_dev_mfcc = encode_labels(mfcc_dev["label"])

    # 1. MFCC + LogReg
    res1 = evaluate_logreg(
        mfcc_train[MFCC_COLS],
        y_train_mfcc,
        mfcc_dev[MFCC_COLS],
        y_dev_mfcc,
    )

    # 2. MFCC + RF
    res2 = evaluate_rf(
        mfcc_train[MFCC_COLS],
        y_train_mfcc,
        mfcc_dev[MFCC_COLS],
        y_dev_mfcc,
    )

    # 3. TDAv1 + LogReg
    y_train_tda = encode_labels(tda_train["label"])
    y_dev_tda = encode_labels(tda_dev["label"])

    res3 = evaluate_logreg(
        tda_train[TDA_V1_COLS],
        y_train_tda,
        tda_dev[TDA_V1_COLS],
        y_dev_tda,
    )

    # 4. Hybrid + RF
    tda_train_small = tda_train[["filename"] + TDA_V1_COLS]
    tda_dev_small = tda_dev[["filename"] + TDA_V1_COLS]

    hybrid_train = mfcc_train.merge(tda_train_small, on="filename", how="inner")
    hybrid_dev = mfcc_dev.merge(tda_dev_small, on="filename", how="inner")

    hybrid_cols = MFCC_COLS + TDA_V1_COLS
    y_train_h = encode_labels(hybrid_train["label"])
    y_dev_h = encode_labels(hybrid_dev["label"])

    res4 = evaluate_rf(
        hybrid_train[hybrid_cols],
        y_train_h,
        hybrid_dev[hybrid_cols],
        y_dev_h,
    )

    return [
        {"sample_size": sample_size, "model": "MFCC + LogReg", **res1},
        {"sample_size": sample_size, "model": "MFCC + RF", **res2},
        {"sample_size": sample_size, "model": "TDAv1 + LogReg", **res3},
        {"sample_size": sample_size, "model": "Hybrid + RF", **res4},
    ]


def save_latex_table_n200(df: pd.DataFrame) -> None:
    n200 = df[df["sample_size"] == 200].copy()
    n200 = n200[["model", "accuracy", "roc_auc", "eer"]]
    n200 = n200.rename(columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "eer": "EER",
    })

    latex = n200.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        caption="Wyniki modeli dla sample size = 200.",
        label="tab:results_n200",
    )

    (RESULTS_DIR / "table_results_n200.tex").write_text(latex, encoding="utf-8")


def save_metric_plot(df: pd.DataFrame, metric: str, filename: str) -> None:
    best_models = ["MFCC + RF", "Hybrid + RF"]
    sub = df[df["model"].isin(best_models)].copy()

    pivot = sub.pivot(index="sample_size", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Sample size per class")
    ax.set_ylabel(metric.upper() if metric != "roc_auc" else "ROC-AUC")
    ax.set_title(f"{metric.upper() if metric != 'roc_auc' else 'ROC-AUC'} vs sample size")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ensure_dirs()

    all_rows = []
    for n in SAMPLE_SIZES:
        print(f"\n=== Running experiments for sample_size={n} ===")
        all_rows.extend(run_all_models_for_sample_size(n))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_DIR / "all_results.csv", index=False)

    save_latex_table_n200(results_df)
    save_metric_plot(results_df, metric="roc_auc", filename="roc_auc_vs_sample_size.png")
    save_metric_plot(results_df, metric="accuracy", filename="accuracy_vs_sample_size.png")
    save_metric_plot(results_df, metric="eer", filename="eer_vs_sample_size.png")

    print("\nSaved:")
    print(RESULTS_DIR / "all_results.csv")
    print(RESULTS_DIR / "table_results_n200.tex")
    print(RESULTS_DIR / "roc_auc_vs_sample_size.png")
    print(RESULTS_DIR / "accuracy_vs_sample_size.png")
    print(RESULTS_DIR / "eer_vs_sample_size.png")