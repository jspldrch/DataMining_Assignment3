"""
model_run12.py — SIMPLIFIED version based on run08 learnings

"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Set paths for loading npz files
# Option 1: If running on Kaggle with uploaded npz files
KAGGLE_INPUT = Path("/kaggle/input")
# Option 2: If running locally with npz in outputs folder
LOCAL_OUTPUT = Path(__file__).parent / "outputs"

# Auto-detect where npz files are
def find_npz_file(filename):
    """Search for npz file in common locations"""
    # Check Kaggle input first
    if KAGGLE_INPUT.exists():
        for path in KAGGLE_INPUT.rglob(filename):
            if path.name == filename:
                return path
    
    # Check local outputs folder
    npz_path = LOCAL_OUTPUT / filename
    if npz_path.exists():
        return npz_path
    
    # Check current directory
    if Path(filename).exists():
        return Path(filename)
    
    raise FileNotFoundError(f"Cannot find {filename} in /kaggle/input, outputs/, or current directory")

# Output directory
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else LOCAL_OUTPUT
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading from NPZ Files
# ──────────────────────────────────────────────────────────────────────────────
print("="*60)
print("LOADING DATA FROM NPZ FILES")
print("="*60)

# Find and load training data
train_npz_path = find_npz_file("train_data.npz")
print(f"Loading training data from: {train_npz_path}")
tr = np.load(train_npz_path, allow_pickle=True)

# Find and load test data
test_npz_path = find_npz_file("test_data.npz")
print(f"Loading test data from: {test_npz_path}")
te = np.load(test_npz_path, allow_pickle=True)

# Extract data
X_train_raw = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_train = tr["y"].astype(np.int32)
train_users = tr["users"]
X_test_raw = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
test_ids = te["file_ids"]
test_users = te["users"]

unique_users = np.unique(train_users)
unique, counts = np.unique(y_train, return_counts=True)
print(f"\nTrain shape: {X_train_raw.shape}")
print(f"Test shape: {X_test_raw.shape}")
print(f"Number of users: {len(unique_users)}")
print(f"\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# Per-User Normalization (same as run07 - it works!)
# ──────────────────────────────────────────────────────────────────────────────
def user_normalize(X, user_ids):
    """Normalize each user's windows by that user's global mean and std"""
    X_out = X.copy()
    for uid in np.unique(user_ids):
        idx = np.where(user_ids == uid)[0]
        if len(idx) == 0:
            continue
        data = X[idx]
        mu = data.mean(axis=(0, 1), keepdims=True)
        sigma = data.std(axis=(0, 1), keepdims=True) + 1e-8
        X_out[idx] = (data - mu) / sigma
    return X_out

print("\nApplying per-user normalization...")
X_train = user_normalize(X_train_raw, train_users)
X_test = user_normalize(X_test_raw, test_users)
print("  Normalization complete")

# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction (run07 features + ONLY 2 safe new features)
# NO feature explosion! Target: ~250 features max
# ──────────────────────────────────────────────────────────────────────────────
def stats9(s):
    """9 statistical features for each signal"""
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s,1),
            np.percentile(s,75,1)-np.percentile(s,25,1),
            np.array([skew(r) for r in s]),
            np.array([kurtosis(r) for r in s])]

def spectral5(s):
    """5 spectral features"""
    N, T = s.shape
    out = np.zeros((N, 5), dtype=np.float32)
    for n in range(N):
        sig = s[n] - s[n].mean()
        freqs, psd = welch(sig, fs=1.0, nperseg=min(64, T))
        pn = psd / (psd.sum() + 1e-10)
        out[n] = [freqs[np.argmax(psd)],
                  -np.sum(pn * np.log(pn + 1e-10)),
                  psd[(freqs>=0)&(freqs<0.5)].sum(),
                  psd[(freqs>=0.5)&(freqs<2)].sum(),
                  psd[freqs>=2].sum()]
    return out

def ac(s, lag):
    """Autocorrelation at given lag"""
    s1, s2 = s[:,:-lag], s[:,lag:]
    num = ((s1-s1.mean(1,keepdims=True))*(s2-s2.mean(1,keepdims=True))).mean(1)
    return num / (s1.std(1)*s2.std(1)+1e-10)

def seg(s, n_seg):
    """Segment statistics"""
    N, T = s.shape
    sl = T // n_seg
    out = []
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1)]
    return out

def xcorr(a, b):
    """Cross-correlation between two signals"""
    ca = a - a.mean(1,keepdims=True)
    cb = b - b.mean(1,keepdims=True)
    return (ca*cb).mean(1) / (ca.std(1)*cb.std(1)+1e-10)

def permutation_entropy(s, order=3):
    """Safe new feature #1: measures signal regularity"""
    N, T = s.shape
    out = np.zeros(N, dtype=np.float32)
    for n in range(N):
        if T <= order:
            out[n] = 0
            continue
        patterns = np.array([np.argsort(s[n][i:i+order]) for i in range(T-order)])
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        p = counts / counts.sum()
        out[n] = -np.sum(p * np.log2(p + 1e-10))
    return out

def energy_ratio(s):
    """Safe new feature #2: low_freq_energy / high_freq_energy"""
    N, T = s.shape
    out = np.zeros(N, dtype=np.float32)
    for n in range(N):
        fft = np.abs(np.fft.rfft(s[n] - s[n].mean()))
        freqs = np.fft.rfftfreq(T, 1)
        low_energy = fft[freqs < 0.3].sum()
        high_energy = fft[freqs >= 0.5].sum()
        out[n] = low_energy / (high_energy + 1e-8)
    return out

def extract_features(X):
    """Extract features (run07 features + 2 safe new ones)"""
    N, T, _ = X.shape
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]
    
    # Jerk (derivative, removes gravity)
    jx = np.diff(mx, axis=1)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)
    
    # Magnitudes (rotation-invariant)
    mag_mean = np.sqrt(mx**2 + my**2 + mz**2)
    mag_std = np.sqrt(sx**2 + sy**2 + sz**2)
    mag_jerk = np.sqrt(jx**2 + jy**2 + jz**2)
    
    parts = []
    
    print("  Extracting std channel statistics...")
    for ch in [sx, sy, sz]:
        parts += stats9(ch)
    parts += stats9(mag_std)
    parts += stats9(mag_mean)
    
    print("  Extracting jerk statistics...")
    for ch in [jx, jy, jz]:
        parts += stats9(ch)
    parts += stats9(mag_jerk)
    
    print("  Extracting segment features...")
    for sig in [mag_std, mag_jerk]:
        for ns in [10, 20]:
            parts += seg(sig, ns)
    for ch in [sx, sy, sz, jx, jy, jz]:
        parts += seg(ch, 10)
    
    print("  Extracting autocorrelation...")
    for lag in [1, 2, 5, 10, 20, 30, 60]:
        parts.append(ac(mag_jerk, lag))
    for ch in [sx, sy, sz]:
        for lag in [1, 5, 10, 30]:
            parts.append(ac(ch, lag))
    
    print("  Extracting spectral features...")
    for sig in [mag_jerk, mag_std, sx, sy, sz]:
        parts.append(spectral5(sig))
    
    print("  Extracting cross-correlations...")
    for a,b in [(jx,jy),(jx,jz),(jy,jz)]:
        parts.append(xcorr(a,b))
    for a,b in [(sx,sy),(sx,sz),(sy,sz)]:
        parts.append(xcorr(a,b))
    
    print("  Extracting zero-crossing and peak rates...")
    cj = mag_jerk - mag_jerk.mean(1, keepdims=True)
    parts.append((np.diff(np.sign(cj), axis=1) != 0).sum(1) / T)
    
    pr = np.zeros(N, dtype=np.float32)
    for n in range(N):
        peaks = find_peaks(mag_jerk[n], height=mag_jerk[n].mean())[0]
        pr[n] = len(peaks) / T
    parts.append(pr)
    
    # ONLY 2 safe new features (not 100+ like run08)
    print("  Extracting additional features (entropy + energy ratio)...")
    parts.append(permutation_entropy(mag_jerk).reshape(-1, 1))
    parts.append(energy_ratio(mag_std).reshape(-1, 1))
    
    # Add mean channels back (run07 didn't have these - adding carefully)
    print("  Extracting mean channel statistics...")
    for ch in [mx, my, mz]:
        parts += stats9(ch)
    
    # Combine all features
    result = np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1) for p in parts
    ]).astype(np.float32)
    
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction with Selection
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FEATURE EXTRACTION")
print("="*60)

print("Extracting features from training data...")
X_train_feat = extract_features(X_train)
print(f"  Train feature shape: {X_train_feat.shape}")

print("\nExtracting features from test data...")
X_test_feat = extract_features(X_test)
print(f"  Test feature shape: {X_test_feat.shape}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

# Feature selection: keep only top 60% by importance
print("\nSelecting important features...")
selector = SelectFromModel(
    lgb.LGBMClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1),
    threshold='median',  # Keep features above median importance
    max_features=200     # Cap at 200 features
)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"  Original features: {X_train_feat.shape[1]}")
print(f"  Selected features: {X_train_selected.shape[1]}")
print(f"  Reduction: {(1 - X_train_selected.shape[1]/X_train_feat.shape[1])*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# Training with Cross-Validation (User-based, like run07)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LEAVE-USER-OUT CROSS-VALIDATION")
print("="*60)

# User-based folds
user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids = np.array([user_folds[u] for u in train_users])

# Store OOF predictions for ensemble weight optimization
oof_preds = np.zeros((len(y_train), 6))

CONFIGS = [
    dict(num_leaves=31, learning_rate=0.05, colsample_bytree=0.7, subsample=0.7),
    dict(num_leaves=63, learning_rate=0.03, colsample_bytree=0.8, subsample=0.8),
    dict(num_leaves=127, learning_rate=0.02, colsample_bytree=0.7, subsample=0.7),
]

fold_accs = []
for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")
    
    probas = []
    for cfg_idx, cfg in enumerate(CONFIGS):
        m = lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=cfg['num_leaves'],
            learning_rate=cfg['learning_rate'],
            colsample_bytree=cfg['colsample_bytree'],
            subsample=cfg['subsample'],
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=SEED + fold + cfg_idx * 100,
            n_jobs=-1,
            verbose=-1
        )
        m.fit(X_train_selected[tr_idx], y_train[tr_idx])
        probas.append(m.predict_proba(X_train_selected[va_idx]))
    
    oof_preds[va_idx] = np.mean(probas, axis=0)
    fold_acc = accuracy_score(y_train[va_idx], oof_preds[va_idx].argmax(axis=1))
    fold_accs.append(fold_acc)
    print(f"  Fold accuracy = {fold_acc:.4f}")

# Calculate ensemble performance
ensemble_preds = oof_preds.argmax(axis=1)
ensemble_acc = accuracy_score(y_train, ensemble_preds)
print(f"\n{'='*40}")
print(f"Ensemble OOF accuracy: {ensemble_acc:.4f}")
print(f"Mean fold accuracy: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")
print(f"{'='*40}")

# ──────────────────────────────────────────────────────────────────────────────
# Final Training on All Data
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL TRAINING ON ALL DATA")
print("="*60)

final_models = []
final_probas = []

total_models = 5 * len(CONFIGS)  # 5 seeds × 3 configs = 15 models
model_count = 0

for seed in range(5):  # 5 seeds
    for cfg_idx, cfg in enumerate(CONFIGS):
        model_count += 1
        print(f"  Training model {model_count}/{total_models}...", end=" ")
        
        m = lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=cfg['num_leaves'],
            learning_rate=cfg['learning_rate'],
            colsample_bytree=cfg['colsample_bytree'],
            subsample=cfg['subsample'],
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=SEED + seed * 100 + cfg_idx,
            n_jobs=-1,
            verbose=-1
        )
        m.fit(X_train_selected, y_train)
        final_models.append(m)
        final_probas.append(m.predict_proba(X_test_selected))
        print("done")

# Average predictions
final_proba = np.mean(final_probas, axis=0)
final_preds = final_proba.argmax(axis=1)

# Gentle post-processing (only for class 4, the most problematic)
print("\n" + "="*60)
print("POST-PROCESSING")
print("="*60)

pred_counts = pd.Series(final_preds).value_counts().sort_index()
expected_class4 = int(len(final_preds) * counts[4] / len(y_train))

print("\nRaw prediction distribution:")
for c in range(6):
    actual = pred_counts.get(c, 0)
    expected = int(len(final_preds) * counts[c] / len(y_train))
    print(f"  Class {c}: {actual:5d} (expected: {expected:5d})")

if pred_counts.get(4, 0) < expected_class4 * 0.5:
    print(f"\nClass 4 is under-predicted ({pred_counts.get(4, 0)} vs expected {expected_class4})")
    print("Applying gentle adjustment...")
    
    # Find low-confidence predictions to convert
    confidence = final_proba.max(axis=1)
    low_conf_idx = np.where(confidence < 0.55)[0]
    
    if len(low_conf_idx) > 0:
        n_convert = min(20, len(low_conf_idx))
        convert_idx = np.random.choice(low_conf_idx, n_convert, replace=False)
        final_preds[convert_idx] = 4
        print(f"  Converted {n_convert} low-confidence predictions to class 4")

# Final distribution
print("\nFinal prediction distribution:")
for c in range(6):
    print(f"  Class {c}: {np.sum(final_preds == c):5d}")

# ──────────────────────────────────────────────────────────────────────────────
# Save Submission
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SAVING SUBMISSION")
print("="*60)

submission = pd.DataFrame({"Id": test_ids, "Label": final_preds})
submission = submission.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run12.csv"
submission.to_csv(out_path, index=False)

print(f"\n✅ Submission saved to: {out_path}")
print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")

# Optional: Save a backup with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = OUT_DIR / f"submission_run12_{timestamp}.csv"
submission.to_csv(backup_path, index=False)
print(f"   Backup saved to: {backup_path}")

# Display sample
print("\nSubmission sample (first 10 rows):")
print(submission.head(10))

# Summary
print("\n" + "="*60)
print("RUN12 SUMMARY")
print("="*60)
print(f"Training samples: {X_train_raw.shape[0]}")
print(f"Test samples: {X_test_raw.shape[0]}")
print(f"Original features: {X_train_feat.shape[1]}")
print(f"Selected features: {X_train_selected.shape[1]}")
print(f"Feature reduction: {(1 - X_train_selected.shape[1]/X_train_feat.shape[1])*100:.1f}%")
print(f"CV ensemble accuracy: {ensemble_acc:.4f}")
print(f"Number of models in ensemble: {len(final_models)}")
print(f"\n✅ Done! Submit {out_path} to Kaggle")
print(f"Expected score: 0.775-0.785")
print("="*60)

# Optional: Show feature importance from final model
print("\nTop 10 features by importance (from final LightGBM):")
importance = final_models[0].feature_importances_
top_indices = np.argsort(importance)[-10:][::-1]
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Feature {idx}: {importance[idx]}")