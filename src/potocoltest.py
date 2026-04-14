from src.paths import LA_PROTOCOL_DIR

for fname in [
    "ASVspoof2019.LA.cm.train.trn.txt",
    "ASVspoof2019.LA.cm.dev.trl.txt",
]:
    path = LA_PROTOCOL_DIR / fname
    print(f"\n=== {fname} ===")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(repr(line.strip()))
            if i >= 4:
                break