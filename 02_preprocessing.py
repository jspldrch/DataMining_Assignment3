"""
Assignment 3 – Step 2: Preprocessing & Feature Engineering
Addresses grading question 2 (10%):
  "What preprocessing techniques did you use to improve performance?
   How much improvement did each technique achieve?"

Strategy: incrementally add preprocessing/feature steps and measure CV accuracy
at each step to quantify the individual contribution of each technique.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import google.colab; IN_COLAB = True
except ImportError:
    IN_COLAB = False

BASE_DIR  = Path("/content/DataMining_Assignment3") if IN_COLAB else Path(__file__).parent
TRAIN_DIR = BASE_DIR / "train" / "train"
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEAT_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

# ── Load raw sequences ─────────────────────────────────────────────────────────
print("Loading training data …")
sequences, labels, file_ids = [], [], []

for user_dir in sorted(TRAIN_DIR.iterdir()):
    for csv_path in sorted(user_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        sequences.append(df[FEAT_COLS].values.astype(np.float32))  # (300, 6)
        labels.append(int(df["label"].iloc[0]))
        file_ids.append(int(df["file_id"].iloc[0]))

X_raw = np.array(sequences)  # (N, 300, 6)
y     = np.array(labels)
print(f"  Loaded {len(y):,} windows. Shape: {X_raw.shape}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def evaluate(X_feat, label, clf=None):
    """Run CV and return mean ± std accuracy."""
    if clf is None:
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    scores = cross_val_score(pipe, X_feat, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {label:<55s}  acc = {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean(), scores.std()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 – Raw baseline: flatten mean/std over the entire 300-second window
# ═══════════════════════════════════════════════════════════════════════════════
def feat_step0(X):
    """12 features: global mean and std of each channel."""
    return np.hstack([
        X.mean(axis=1),   # (N, 6)
        X.std(axis=1),    # (N, 6)
    ])

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Handle missing / constant channels
#   Some windows have near-zero std channels (sensor stuck).
#   Replace NaN and clip extreme values to ±10.
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_clip(X):
    """Clip outlier values to ±10 and fill NaN with channel median."""
    X = np.where(np.isnan(X), np.nanmedian(X, axis=1, keepdims=True), X)
    X = np.clip(X, -10, 10)
    return X

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Extended statistical features
# ═══════════════════════════════════════════════════════════════════════════════
def feat_step2(X):
    """
    Per-channel over 300s:
      mean, std, min, max, range, median, IQR, skewness, kurtosis
    = 9 × 6 = 54 features
    """
    feats = []
    for ch in range(X.shape[2]):
        s = X[:, :, ch]
        feats += [
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
    return np.column_stack(feats)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Signal magnitude (vector norm) features
# ═══════════════════════════════════════════════════════════════════════════════
def feat_step3(X):
    """Add vector magnitude (|a|) mean and std (2 extra features)."""
    mag = np.sqrt((X[:, :, :3] ** 2).sum(axis=2))  # (N, 300) using mean axes
    return np.column_stack([
        feat_step2(X),
        mag.mean(axis=1),
        mag.std(axis=1),
        mag.max(axis=1) - mag.min(axis=1),
    ])

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Temporal segment features (divide 300s into 10 segments of 30s)
#   Captures HOW signals change over the 5-minute window.
# ═══════════════════════════════════════════════════════════════════════════════
def feat_step4(X, n_segments=10):
    """
    Divide each 300-step sequence into n_segments equal parts.
    Compute per-segment mean and std for each channel.
    = n_segments × 6 channels × 2 stats = 120 features
    """
    N, T, C = X.shape
    seg_len  = T // n_segments
    seg_feats = []
    for i in range(n_segments):
        seg = X[:, i * seg_len:(i + 1) * seg_len, :]   # (N, seg_len, C)
        seg_feats.append(seg.mean(axis=1))
        seg_feats.append(seg.std(axis=1))
    return np.hstack([feat_step3(X), np.hstack(seg_feats)])

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Frequency-domain features (FFT / spectral power bands)
#   Physical activities have characteristic frequency signatures.
# ═══════════════════════════════════════════════════════════════════════════════
def spectral_features(X):
    """
    For each channel:
      dominant frequency, spectral entropy, power in 3 bands:
        low (0-0.5 Hz), mid (0.5-2 Hz), high (2-5 Hz)
    = 5 features × 6 channels = 30 features
    """
    N, T, C = X.shape
    fs = 1.0  # 1 Hz (one reading per second)
    out = np.zeros((N, 5 * C), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            signal = X[n, :, c] - X[n, :, c].mean()  # detrend
            freqs, psd = welch(signal, fs=fs, nperseg=min(64, T))

            # dominant frequency
            dom_freq = freqs[np.argmax(psd)]

            # spectral entropy
            psd_norm = psd / (psd.sum() + 1e-10)
            spec_ent = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

            # band powers
            p_low  = psd[(freqs >= 0.0) & (freqs < 0.5)].sum()
            p_mid  = psd[(freqs >= 0.5) & (freqs < 2.0)].sum()
            p_high = psd[(freqs >= 2.0) & (freqs < 5.0)].sum()

            out[n, c * 5]     = dom_freq
            out[n, c * 5 + 1] = spec_ent
            out[n, c * 5 + 2] = p_low
            out[n, c * 5 + 3] = p_mid
            out[n, c * 5 + 4] = p_high

    return out


def feat_step5(X):
    return np.hstack([feat_step4(X), spectral_features(X)])


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE each step
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PREPROCESSING ABLATION – incremental CV accuracy (RF, 5-fold)")
print("=" * 72)

results = {}

# Baseline (no clipping, only global mean/std)
results["Step 0: global mean + std (12 feat)"] = evaluate(feat_step0(X_raw), "Step 0: global mean + std (12 feat)")

# Step 1: with clipping
X_clip = preprocess_clip(X_raw)
results["Step 1: + clip outliers"] = evaluate(feat_step0(X_clip), "Step 1: + clip outliers")

# Step 2: rich statistical features
results["Step 2: + extended stats (54 feat)"] = evaluate(feat_step2(X_clip), "Step 2: + extended stats (54 feat)")

# Step 3: magnitude
results["Step 3: + vector magnitude (57 feat)"] = evaluate(feat_step3(X_clip), "Step 3: + vector magnitude (57 feat)")

# Step 4: temporal segments
results["Step 4: + temporal segments (177 feat)"] = evaluate(feat_step4(X_clip), "Step 4: + temporal segments (177 feat)")

# Step 5: frequency features
print("  [Computing spectral features – may take a minute …]")
results["Step 5: + frequency domain (207 feat)"] = evaluate(feat_step5(X_clip), "Step 5: + frequency domain (207 feat)")

# ── Plot progression ───────────────────────────────────────────────────────────
steps  = list(results.keys())
means  = [v[0] for v in results.values()]
stds   = [v[1] for v in results.values()]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(steps))
ax.bar(x, means, yerr=stds, capsize=4, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(steps))))
ax.set_xticks(x)
ax.set_xticklabels([s.split(":")[0] for s in steps], rotation=15, ha="right")
ax.set_ylabel("CV Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Preprocessing Step Contributions (Random Forest, 5-fold CV)")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_preprocessing_steps.png", dpi=150)
plt.close()

# ── Save feature matrix for later scripts ─────────────────────────────────────
print("\nSaving best feature matrix …")
X_best = feat_step5(X_clip)
np.save(OUT_DIR / "X_train_features.npy", X_best)
np.save(OUT_DIR / "y_train.npy", y)
np.save(OUT_DIR / "file_ids_train.npy", np.array(file_ids))

summary = pd.DataFrame(results, index=["mean_acc", "std_acc"]).T
summary.to_csv(OUT_DIR / "02_preprocessing_results.csv")

print(f"\nBest feature set: {X_best.shape[1]} features")
print(f"Outputs saved to: {OUT_DIR}")
