from pathlib import Path
import pandas as pd


def parse_asvspoof2019_la_cm(protocol_path: str | Path) -> pd.DataFrame:
    protocol_path = Path(protocol_path)

    rows = []

    with protocol_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(f"Unexpected protocol format in line: {line}")

            speaker_id = parts[0]
            file_id = parts[1]
            attack_id = parts[2]
            subset_tag = parts[3]
            label = parts[4].lower()

            rows.append({
                "speaker_id": speaker_id,
                "file_id": file_id,
                "filename": f"{file_id}.flac",
                "attack_id": attack_id,
                "subset_tag": subset_tag,
                "label": label,
            })

    return pd.DataFrame(rows)