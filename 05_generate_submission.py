"""
Assignment 3 – Step 5: Generate Kaggle Submission
Loads the trained model from 03_model_temporal.py and predicts on test data.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

try:
    import google.colab; IN_COLAB = True
except ImportError:
    IN_COLAB = False

BASE_DIR = Path("/content/DataMining_Assignment3") if IN_COLAB else Path(__file__).parent
OUT_DIR  = BASE_DIR / "outputs"

model    = joblib.load(OUT_DIR / "best_model.pkl")
X_test   = np.load(OUT_DIR / "X_test_features.npy")
test_ids = np.load(OUT_DIR / "test_file_ids.npy")

preds = model.predict(X_test)

submission = pd.DataFrame({"Id": test_ids, "Label": preds})
submission = submission.sort_values("Id").reset_index(drop=True)
submission.to_csv(BASE_DIR / "submission.csv", index=False)

print(f"Saved {len(submission)} predictions to submission.csv")
print(submission["Label"].value_counts().sort_index().to_string())
