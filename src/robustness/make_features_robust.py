from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.common.data_io import load_audio
from src.common.protocols import parse_asvspoof2019_la_cm
from src.clean.features_mfcc import extract_mfcc_stats
from src.robustness.audio_augment import apply_degradation


def build_feature_dataframe_robust(
    audio_dir: str | Path,
    protocol_path: str | Path,
    n_mfcc: int = 13,
    max_files: int | None = None,
    samples_per_class: int | None = None,
    random_state: int = 42,
    degradation_type: str | None = None,
    degradation_value: float | str | None = None,
    degradation_seed: int = 42,
) -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    labels_df = parse_asvspoof2019_la_cm(protocol_path)

    if samples_per_class is not None:
        bonafide_df = labels_df[labels_df["label"] == "bonafide"].sample(
            n=min(samples_per_class, (labels_df["label"] == "bonafide").sum()),
            random_state=random_state,
        )
        spoof_df = labels_df[labels_df["label"] == "spoof"].sample(
            n=min(samples_per_class, (labels_df["label"] == "spoof").sum()),
            random_state=random_state,
        )
        labels_df = pd.concat([bonafide_df, spoof_df], ignore_index=True)
        labels_df = labels_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    elif max_files is not None:
        labels_df = labels_df.iloc[:max_files].copy()

    rows = []

    for i, (_, meta) in enumerate(
        tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"MFCC robust {audio_dir.name}")
    ):
        audio_path = audio_dir / meta["filename"]

        if not audio_path.exists():
            print(f"Missing audio file: {audio_path}")
            continue

        try:
            y, sr = load_audio(audio_path, sr=16000, mono=True, duration=3.0)

            y = apply_degradation(
                y,
                sr,
                degradation_type=degradation_type,
                degradation_value=degradation_value,
                random_state=degradation_seed + i,
            )

            x = extract_mfcc_stats(y, sr, n_mfcc=n_mfcc)

            row = {
                "filename": meta["filename"],
                "speaker_id": meta["speaker_id"],
                "attack_id": meta["attack_id"],
                "label": meta["label"],
                **{f"f{i}": float(val) for i, val in enumerate(x)},
            }
            rows.append(row)

        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")

    return pd.DataFrame(rows)