"""
Run this LOCALLY on your laptop.
Converts all train/test CSVs into two compressed numpy files:
  outputs/train_data.npz
  outputs/test_data.npz
Then upload only those 2 files to Kaggle as a dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "train" / "train"
TEST_DIR  = BASE_DIR / "test"  / "test"
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEAT_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

def convert(root_dir, out_path, has_labels=True):
    sequences, labels, file_ids, users = [], [], [], []
    all_dirs = sorted(root_dir.iterdir())
    for i, user_dir in enumerate(all_dirs):
        print(f"  {user_dir.name}  ({i+1}/{len(all_dirs)})", end="\r")
        for csv_path in sorted(user_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            sequences.append(df[FEAT_COLS].values.astype(np.float32))
            if has_labels:
                labels.append(int(df["label"].iloc[0]))
            file_ids.append(int(df["file_id"].iloc[0]))
            users.append(user_dir.name)

    X = np.array(sequences, dtype=np.float32)
    save_dict = dict(X=X, file_ids=np.array(file_ids),
                     users=np.array(users))
    if has_labels:
        save_dict["y"] = np.array(labels, dtype=np.int32)

    np.savez_compressed(out_path, **save_dict)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved {out_path.name}  ({X.shape}, {size_mb:.1f} MB)")

print("Converting train …")
convert(TRAIN_DIR, OUT_DIR / "train_data.npz", has_labels=True)

print("Converting test …")
convert(TEST_DIR,  OUT_DIR / "test_data.npz",  has_labels=False)

print("\nDone. Upload outputs/train_data.npz and outputs/test_data.npz to Kaggle.")
