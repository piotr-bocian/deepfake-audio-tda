from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

LA_DIR = DATA_DIR / "LA"
LA_TRAIN_DIR = LA_DIR / "train" / "flac"
LA_DEV_DIR = LA_DIR / "dev" / "flac"
LA_PROTOCOL_DIR = LA_DIR / "protocols"

LA_TRAIN_PROTOCOL = LA_PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
LA_DEV_PROTOCOL = LA_PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
LA_EVAL_PROTOCOL = LA_PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"