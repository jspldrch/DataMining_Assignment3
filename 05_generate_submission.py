"""
Assignment 3 – Step 5: Generate Kaggle Submission
Loads the trained model from 03_model_temporal.py and predicts on test data.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

def _find_base_dir():
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for comp_dir in kaggle_input.iterdir():
            if (comp_dir / "train" / "train").exists():
                return comp_dir, Path("/kaggle/working")
    try:
        import google.colab
        p = Path("/content/DataMining_Assignment3")
        return p, p / "outputs"
    except ImportError:
        pass
    p = Path(__file__).parent
    return p, p / "outputs"

BASE_DIR, OUT_DIR = _find_base_dir()

model    = joblib.load(OUT_DIR / "best_model.pkl")
X_test   = np.load(OUT_DIR / "X_test_features.npy")
test_ids = np.load(OUT_DIR / "test_file_ids.npy")

preds = model.predict(X_test)

submission = pd.DataFrame({"Id": test_ids, "Label": preds})
submission = submission.sort_values("Id").reset_index(drop=True)
submission.to_csv(OUT_DIR / "submission.csv", index=False)

print(f"Saved {len(submission)} predictions to {OUT_DIR / 'submission.csv'}")
print(submission["Label"].value_counts().sort_index().to_string())
