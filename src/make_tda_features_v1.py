from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data_io import load_audio
from src.features_tda_v1 import extract_tda_features_from_mfcc_v1
from src.protocols import parse_asvspoof2019_la_cm


def build_tda_v1_feature_dataframe(
    audio_dir: str | Path,
    protocol_path: str | Path,
    n_mfcc: int = 13,
    max_points: int = 80,
    samples_per_class: int | None = None,
    max_files: int | None = None,
    random_state: int = 42,
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

    for _, meta in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"TDAv1 {audio_dir.name}"):
        audio_path = audio_dir / meta["filename"]

        if not audio_path.exists():
            print(f"Missing audio file: {audio_path}")
            continue

        try:
            y, sr = load_audio(audio_path, sr=16000, mono=True, duration=3.0)
            x = extract_tda_features_from_mfcc_v1(
                y,
                sr,
                n_mfcc=n_mfcc,
                max_points=max_points,
                maxdim=1,
            )

            row = {
                "filename": meta["filename"],
                "speaker_id": meta["speaker_id"],
                "attack_id": meta["attack_id"],
                "label": meta["label"],
                **{f"t{i}": float(val) for i, val in enumerate(x)},
            }
            rows.append(row)

        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")

    return pd.DataFrame(rows)