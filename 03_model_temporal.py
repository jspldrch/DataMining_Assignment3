"""
Assignment 3 – Step 3: Model with Temporal Alignment
Addresses grading question 3 (10%):
  "Please describe how you align the activity labels with the sequential
   accelerometer readings (e.g., what temporal features you added or how
   your model captures temporal dependencies)."

Temporal alignment strategy:
  Each CSV file = one 5-minute window (300 seconds) with a SINGLE label.
  We do NOT predict per-second; we predict once per file.
  To capture temporal patterns, we use:
    (a) Segment-level features: divide 300s into N equal segments, compute
        statistics per segment → encodes HOW the signal evolves over time.
    (b) Autocorrelation features: correlation of signal with itself at lags
        1, 5, 10, 30s → captures periodicity and rhythm.
    (c) Trend features: slope of linear regression over the window →
        distinguishes activities that ramp up/down vs. stay constant.
    (d) Frequency domain: dominant frequency via FFT/Welch → activity-specific
        motion patterns (e.g., walking cadence ~1-2 Hz).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Using Random Forest only. Install with: pip install xgboost")

try:
    import google.colab; IN_COLAB = True
except ImportError:
    IN_COLAB = False

BASE_DIR  = Path("/content/DataMining_Assignment3") if IN_COLAB else Path(__file__).parent
TRAIN_DIR = BASE_DIR / "train" / "train"
TEST_DIR  = BASE_DIR / "test"  / "test"
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEAT_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


# ── Feature extraction (full temporal-aware feature set) ──────────────────────

def extract_features(X_3d: np.ndarray) -> np.ndarray:
    """
    X_3d: shape (N, 300, 6)  –  N windows, 300 seconds, 6 channels

    Returns feature matrix of shape (N, D).

    Feature groups:
      A. Global statistics     (9 stats × 6 channels = 54)
      B. Vector magnitude      (3 stats = 3)
      C. Temporal segments     (10 segments × 2 stats × 6 ch = 120)
      D. Autocorrelation       (4 lags × 6 ch = 24)
      E. Linear trend (slope)  (6 ch = 6)
      F. Spectral features     (5 features × 6 ch = 30)
      Total: 237 features
    """
    N, T, C = X_3d.shape
    feature_parts = []

    # ── A. Global statistics ──────────────────────────────────────────────────
    for c in range(C):
        s = X_3d[:, :, c]
        feature_parts += [
            s.mean(axis=1),
            s.std(axis=1),
            s.min(axis=1),
            s.max(axis=1),
            s.max(axis=1) - s.min(axis=1),
            np.median(s, axis=1),
            np.percentile(s, 75, axis=1) - np.percentile(s, 25, axis=1),
            np.array([skew(row) for row in s]),
            np.array([kurtosis(row) for row in s]),
        ]

    # ── B. Vector magnitude of mean axes ─────────────────────────────────────
    mag = np.sqrt((X_3d[:, :, :3] ** 2).sum(axis=2))   # (N, 300)
    feature_parts += [
        mag.mean(axis=1),
        mag.std(axis=1),
        mag.max(axis=1) - mag.min(axis=1),
    ]

    # ── C. Temporal segment features ─────────────────────────────────────────
    # Divide 300 seconds into 10 equal segments (30s each).
    # Each segment gets mean and std per channel.
    n_segments = 10
    seg_len = T // n_segments
    for i in range(n_segments):
        seg = X_3d[:, i * seg_len:(i + 1) * seg_len, :]  # (N, 30, 6)
        feature_parts.append(seg.mean(axis=1))            # (N, 6)
        feature_parts.append(seg.std(axis=1))

    # ── D. Autocorrelation at lags 1, 5, 10, 30 ──────────────────────────────
    # Captures rhythmic/periodic structure of the motion signal.
    for lag in [1, 5, 10, 30]:
        acorr = np.zeros((N, C), dtype=np.float32)
        for c in range(C):
            s = X_3d[:, :, c]
            # Pearson correlation between s[t] and s[t+lag]
            s1, s2 = s[:, :-lag], s[:, lag:]
            mu1, mu2 = s1.mean(axis=1, keepdims=True), s2.mean(axis=1, keepdims=True)
            num = ((s1 - mu1) * (s2 - mu2)).mean(axis=1)
            den = s1.std(axis=1) * s2.std(axis=1) + 1e-10
            acorr[:, c] = num / den
        feature_parts.append(acorr)

    # ── E. Linear trend (slope) over time ─────────────────────────────────────
    # Captures whether a signal is increasing/decreasing during the window.
    t_vec = np.arange(T, dtype=np.float32)
    t_norm = t_vec - t_vec.mean()
    slopes = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        s = X_3d[:, :, c]
        slopes[:, c] = (s * t_norm).sum(axis=1) / (t_norm ** 2).sum()
    feature_parts.append(slopes)

    # ── F. Spectral features (Welch PSD) ─────────────────────────────────────
    # Frequency content distinguishes activities (walking ~1-2 Hz, etc.)
    fs = 1.0  # 1 Hz
    spec = np.zeros((N, 5 * C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            signal = X_3d[n, :, c] - X_3d[n, :, c].mean()
            freqs, psd = welch(signal, fs=fs, nperseg=min(64, T))
            dom_freq    = freqs[np.argmax(psd)]
            psd_norm    = psd / (psd.sum() + 1e-10)
            spec_ent    = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            p_low       = psd[(freqs >= 0.0) & (freqs < 0.5)].sum()
            p_mid       = psd[(freqs >= 0.5) & (freqs < 2.0)].sum()
            p_high      = psd[(freqs >= 2.0)].sum()
            spec[n, c * 5:(c + 1) * 5] = [dom_freq, spec_ent, p_low, p_mid, p_high]
    feature_parts.append(spec)

    # ── Concatenate ───────────────────────────────────────────────────────────
    flat_parts = []
    for p in feature_parts:
        if p.ndim == 1:
            flat_parts.append(p.reshape(-1, 1))
        else:
            flat_parts.append(p)

    return np.hstack(flat_parts).astype(np.float32)


def load_dataset(root_dir: Path):
    sequences, labels, file_ids = [], [], []
    for user_dir in sorted(root_dir.iterdir()):
        for csv_path in sorted(user_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            sequences.append(df[FEAT_COLS].values.astype(np.float32))
            if "label" in df.columns:
                labels.append(int(df["label"].iloc[0]))
            file_ids.append(int(df["file_id"].iloc[0]))
    X = np.array(sequences)
    X = np.clip(np.nan_to_num(X, nan=0.0), -10, 10)
    return X, np.array(labels) if labels else None, np.array(file_ids)


# ── Load & extract features ────────────────────────────────────────────────────
print("Loading training data …")
X_train_raw, y_train, train_ids = load_dataset(TRAIN_DIR)
print(f"  Train: {X_train_raw.shape}")

print("Extracting temporal features (train) …")
X_train = extract_features(X_train_raw)
print(f"  Feature matrix: {X_train.shape}  ({X_train.shape[1]} features per window)")

print("Loading test data …")
X_test_raw, _, test_ids = load_dataset(TEST_DIR)
print(f"  Test: {X_test_raw.shape}")

print("Extracting temporal features (test) …")
X_test = extract_features(X_test_raw)

# ── Cross-validation ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL EVALUATION – 5-fold stratified CV")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest (200 trees)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]),
}

if HAS_XGB:
    models["XGBoost"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(
            n_estimators=300, learning_rate=0.1,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )),
    ])

cv_results = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<35s}  acc = {scores.mean():.4f} ± {scores.std():.4f}")

# ── Train final model on all training data ────────────────────────────────────
best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest model: {best_name}  (acc={cv_results[best_name].mean():.4f})")

final_model = models[best_name]
final_model.fit(X_train, y_train)

# In-sample classification report (just for analysis; not a test metric)
y_pred_train = final_model.predict(X_train)
print("\nIn-sample Classification Report:")
print(classification_report(y_train, y_pred_train, digits=4))

# Confusion matrix
cm = confusion_matrix(y_train, y_pred_train)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (in-sample) – {best_name}")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance (if RF or XGB)
clf = final_model.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    top_k = 20
    idx = np.argsort(importances)[-top_k:][::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(top_k), importances[idx])
    ax.set_xticks(np.arange(top_k))
    ax.set_xticklabels([f"f{i}" for i in idx], rotation=45, ha="right")
    ax.set_title(f"Top {top_k} Feature Importances")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_feature_importance.png", dpi=150)
    plt.close()

# ── Save model and features for prediction ────────────────────────────────────
joblib.dump(final_model, OUT_DIR / "best_model.pkl")
np.save(OUT_DIR / "X_test_features.npy",  X_test)
np.save(OUT_DIR / "test_file_ids.npy",    test_ids)

print(f"\nModel saved to: {OUT_DIR / 'best_model.pkl'}")
print(f"Outputs saved to: {OUT_DIR}")
print("\nRun 05_generate_submission.py to produce the Kaggle CSV.")
