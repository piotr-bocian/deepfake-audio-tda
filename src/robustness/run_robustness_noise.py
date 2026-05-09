from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.common.paths import (
    LA_TRAIN_DIR,
    LA_DEV_DIR,
    LA_TRAIN_PROTOCOL,
    LA_DEV_PROTOCOL,
)
from src.robustness.make_features_robust import build_feature_dataframe_robust
from src.robustness.make_tda_features_robust import build_tda_feature_dataframe_robust
from src.robustness.train_rf_robust import run_mfcc_rf_robust
from src.robustness.train_hybrid_rf_robust import run_hybrid_rf_robust


DATA_DIR = Path("../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = DATA_DIR / "results_robustness_noise.csv"


def build_clean_train_mfcc(sample_size: int, seed: int) -> Path:
    out_path = DATA_DIR / f"robust_train_mfcc_clean_s{sample_size}_seed{seed}.csv"

    df = build_feature_dataframe_robust(
        LA_TRAIN_DIR,
        LA_TRAIN_PROTOCOL,
        n_mfcc=13,
        samples_per_class=sample_size,
        random_state=seed,
        degradation_type=None,
        degradation_value=None,
    )
    if df.empty:
        raise ValueError(f"Generated empty dataframe: {out_path}")
    df.to_csv(out_path, index=False)
    return out_path


def build_clean_train_tda(sample_size: int, seed: int) -> Path:
    out_path = DATA_DIR / f"robust_train_tda_clean_s{sample_size}_seed{seed}.csv"

    df = build_tda_feature_dataframe_robust(
        LA_TRAIN_DIR,
        LA_TRAIN_PROTOCOL,
        n_mfcc=13,
        max_points=80,
        pca_components=None,
        use_h0=True,
        use_h1=True,
        samples_per_class=sample_size,
        random_state=seed,
        degradation_type=None,
        degradation_value=None,
    )
    if df.empty:
        raise ValueError(f"Generated empty dataframe: {out_path}")
    df.to_csv(out_path, index=False)
    return out_path


def build_noisy_dev_mfcc(snr_db: float, degradation_seed: int = 42) -> Path:
    out_path = DATA_DIR / f"robust_dev_mfcc_noise_{int(snr_db)}db.csv"
    if out_path.exists():
        return out_path

    df = build_feature_dataframe_robust(
        LA_DEV_DIR,
        LA_DEV_PROTOCOL,
        n_mfcc=13,
        samples_per_class=None,
        random_state=42,
        degradation_type="white_noise",
        degradation_value=snr_db,
        degradation_seed=degradation_seed,
    )
    if df.empty:
        raise ValueError(f"Generated empty dataframe: {out_path}")
    df.to_csv(out_path, index=False)
    return out_path


def build_noisy_dev_tda(snr_db: float, degradation_seed: int = 42) -> Path:
    out_path = DATA_DIR / f"robust_dev_tda_noise_{int(snr_db)}db.csv"
    if out_path.exists():
        return out_path

    df = build_tda_feature_dataframe_robust(
        LA_DEV_DIR,
        LA_DEV_PROTOCOL,
        n_mfcc=13,
        max_points=80,
        pca_components=None,
        use_h0=True,
        use_h1=True,
        samples_per_class=None,
        random_state=42,
        degradation_type="white_noise",
        degradation_value=snr_db,
        degradation_seed=degradation_seed,
    )
    if df.empty:
        raise ValueError(f"Generated empty dataframe: {out_path}")
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    sample_size = 2000
    seeds = [0,1,2]
    snr_levels = [20,10,5]

    rows: list[dict] = []

    for snr_db in snr_levels:
        print(f"Preparing degraded dev for white noise at {snr_db} dB SNR")

        mfcc_dev_csv = build_noisy_dev_mfcc(snr_db=snr_db, degradation_seed=42)
        tda_dev_csv = build_noisy_dev_tda(snr_db=snr_db, degradation_seed=42)

        for seed in seeds:
            print(f"Running sample_size={sample_size}, seed={seed}, noise={snr_db} dB")

            mfcc_train_csv = build_clean_train_mfcc(sample_size=sample_size, seed=seed)
            tda_train_csv = build_clean_train_tda(sample_size=sample_size, seed=seed)

            rows.append({
                "condition": f"white_noise_{snr_db}dB",
                "sample_size": sample_size,
                "seed": seed,
                **run_mfcc_rf_robust(
                    train_csv=mfcc_train_csv,
                    dev_csv=mfcc_dev_csv,
                    rf_seed=seed,
                ),
            })

            rows.append({
                "condition": f"white_noise_{snr_db}dB",
                "sample_size": sample_size,
                "seed": seed,
                **run_hybrid_rf_robust(
                    mfcc_train_csv=mfcc_train_csv,
                    tda_train_csv=tda_train_csv,
                    mfcc_dev_csv=mfcc_dev_csv,
                    tda_dev_csv=tda_dev_csv,
                    rf_seed=seed,
                ),
            })

            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    print(f"Saved robustness results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()