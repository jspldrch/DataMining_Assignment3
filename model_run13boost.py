"""
model_run19_two_stage.py — Two-stage training for rare classes
Stage 1: Binary classifier for Class 2 (Walking Downstairs)
Stage 2: Multiclass for remaining classes
Stage 3: Ensemble with class-specific weights

Expected improvement: 0.7707 → 0.775-0.780
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD NPZ DATA (use run07's proven data source)
# ──────────────────────────────────────────────────────────────────────────────
def find_npz(name):
    hits = glob.glob(f"**/{name}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} not found")

print("Loading NPZ data...")
tr = np.load(find_npz("train_data.npz"), allow_pickle=True)
te = np.load(find_npz("test_data.npz"), allow_pickle=True)

X_train_raw = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_train = tr["y"].astype(np.int32)
train_users = tr["users"]
X_test_raw = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
test_ids = te["file_ids"]
test_users = te["users"]

unique, counts = np.unique(y_train, return_counts=True)
print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# PER-USER NORMALIZATION (same as run07)
# ──────────────────────────────────────────────────────────────────────────────
def user_normalize(X, user_ids):
    X_out = X.copy()
    for uid in np.unique(user_ids):
        idx = np.where(user_ids == uid)[0]
        if len(idx) > 0:
            data = X[idx]
            mu = data.mean(axis=(0, 1), keepdims=True)
            sigma = data.std(axis=(0, 1), keepdims=True) + 1e-8
            X_out[idx] = (data - mu) / sigma
    return X_out

print("\nPer-user normalization...")
X_train = user_normalize(X_train_raw, train_users)
X_test = user_normalize(X_test_raw, test_users)

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (same as run07 - 373 features)
# ──────────────────────────────────────────────────────────────────────────────
def stats9(s):
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s,1),
            np.percentile(s,75,1)-np.percentile(s,25,1),
            np.array([skew(r) for r in s]),
            np.array([kurtosis(r) for r in s])]

def spectral5(s):
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
    s1, s2 = s[:,:-lag], s[:,lag:]
    num = ((s1-s1.mean(1,keepdims=True))*(s2-s2.mean(1,keepdims=True))).mean(1)
    return num / (s1.std(1)*s2.std(1)+1e-10)

def seg(s, n_seg):
    N, T = s.shape
    sl = T // n_seg
    out = []
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1)]
    return out

def xcorr(a, b):
    ca = a - a.mean(1,keepdims=True)
    cb = b - b.mean(1,keepdims=True)
    return (ca*cb).mean(1) / (ca.std(1)*cb.std(1)+1e-10)

def extract_features(X):
    N, T, _ = X.shape
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]
    
    jx = np.diff(mx, axis=1)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)
    
    mag_mean = np.sqrt(mx**2 + my**2 + mz**2)
    mag_std = np.sqrt(sx**2 + sy**2 + sz**2)
    mag_jerk = np.sqrt(jx**2 + jy**2 + jz**2)
    
    parts = []
    
    for ch in [sx, sy, sz]:
        parts += stats9(ch)
    parts += stats9(mag_std)
    parts += stats9(mag_mean)
    
    for ch in [jx, jy, jz]:
        parts += stats9(ch)
    parts += stats9(mag_jerk)
    
    for sig in [mag_std, mag_jerk]:
        for ns in [10, 20]:
            parts += seg(sig, ns)
    for ch in [sx, sy, sz, jx, jy, jz]:
        parts += seg(ch, 10)
    
    for lag in [1, 2, 5, 10, 20, 30, 60]:
        parts.append(ac(mag_jerk, lag))
    for ch in [sx, sy, sz]:
        for lag in [1, 5, 10, 30]:
            parts.append(ac(ch, lag))
    
    for sig in [mag_jerk, mag_std, sx, sy, sz]:
        parts.append(spectral5(sig))
    
    for a,b in [(jx,jy),(jx,jz),(jy,jz)]:
        parts.append(xcorr(a,b))
    for a,b in [(sx,sy),(sx,sz),(sy,sz)]:
        parts.append(xcorr(a,b))
    
    cj = mag_jerk - mag_jerk.mean(1, keepdims=True)
    parts.append((np.diff(np.sign(cj), axis=1) != 0).sum(1) / T)
    
    pr = np.zeros(N, dtype=np.float32)
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n], height=mag_jerk[n].mean())[0]) / T
    parts.append(pr)
    
    return np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1) for p in parts
    ]).astype(np.float32)

print("\nExtracting features...")
X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)
print(f"  Train: {X_train_feat.shape}, Test: {X_test_feat.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

# ──────────────────────────────────────────────────────────────────────────────
# TWO-STAGE TRAINING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TWO-STAGE TRAINING FOR RARE CLASSES")
print("="*60)

# Stage 1: Binary classifier for Class 2 (vs all others)
print("\nStage 1: Training Class 2 detector...")
y_class2_binary = (y_train == 2).astype(int)

# Oversample Class 2 for binary classifier
class2_indices = np.where(y_class2_binary == 1)[0]
class2_count = len(class2_indices)
non_class2_indices = np.where(y_class2_binary == 0)[0]
non_class2_sampled = np.random.choice(non_class2_indices, class2_count * 2, replace=False)

balanced_indices = np.concatenate([class2_indices, non_class2_sampled])
X_balanced = X_train_scaled[balanced_indices]
y_balanced = y_class2_binary[balanced_indices]

binary_model = lgb.LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)
binary_model.fit(X_balanced, y_balanced)

# Predict probability of Class 2
class2_proba = binary_model.predict_proba(X_test_scaled)[:, 1]
print(f"  Class 2 detection model trained")

# Stage 2: Train separate model for Class 4
print("\nStage 2: Training Class 4 detector...")
y_class4_binary = (y_train == 4).astype(int)

class4_indices = np.where(y_class4_binary == 1)[0]
class4_count = len(class4_indices)
non_class4_indices = np.where(y_class4_binary == 0)[0]
non_class4_sampled = np.random.choice(non_class4_indices, class4_count * 3, replace=False)

balanced_indices4 = np.concatenate([class4_indices, non_class4_sampled])
X_balanced4 = X_train_scaled[balanced_indices4]
y_balanced4 = y_class4_binary[balanced_indices4]

binary_model4 = lgb.LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)
binary_model4.fit(X_balanced4, y_balanced4)

class4_proba = binary_model4.predict_proba(X_test_scaled)[:, 1]
print(f"  Class 4 detection model trained")

# Stage 3: Main multiclass model (all classes)
print("\nStage 3: Training main multiclass model...")

# Give higher weight to rare classes
sample_weights = np.ones(len(y_train))
sample_weights[y_train == 2] = 5.0  # 5x weight for Class 2
sample_weights[y_train == 4] = 8.0  # 8x weight for Class 4
sample_weights[y_train == 5] = 2.0  # 2x weight for Class 5

CONFIGS = [
    dict(num_leaves=31, learning_rate=0.05, colsample_bytree=0.7, subsample=0.7),
    dict(num_leaves=63, learning_rate=0.03, colsample_bytree=0.8, subsample=0.8),
    dict(num_leaves=127, learning_rate=0.02, colsample_bytree=0.7, subsample=0.7),
]

main_probas = []
for seed in range(5):
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **cfg
        )
        m.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        main_probas.append(m.predict_proba(X_test_scaled))

main_proba = np.mean(main_probas, axis=0)
print(f"  Main model trained (15 models)")

# ──────────────────────────────────────────────────────────────────────────────
# ENSEMBLE WITH RARE CLASS DETECTORS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ENSEMBLE WITH RARE CLASS DETECTORS")
print("="*60)

# Combine predictions: if binary detector is confident, override main model
final_preds = main_proba.argmax(axis=1)

# Override Class 2 predictions
class2_confidence = class2_proba
class2_override_idx = np.where(class2_confidence > 0.6)[0]  # Threshold
final_preds[class2_override_idx] = 2
print(f"  Overrode {len(class2_override_idx)} predictions to Class 2")

# Override Class 4 predictions
class4_confidence = class4_proba
class4_override_idx = np.where(class4_confidence > 0.65)[0]  # Higher threshold for Class 4
final_preds[class4_override_idx] = 4
print(f"  Overrode {len(class4_override_idx)} predictions to Class 4")

# ──────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL PREDICTIONS")
print("="*60)

pred_counts = pd.Series(final_preds).value_counts().sort_index()
print("\nPrediction distribution:")
for c in range(6):
    expected = int(len(final_preds) * counts[c] / len(y_train))
    print(f"  Class {c}: {pred_counts.get(c, 0):5d} (expected: {expected:5d})")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"Id": test_ids, "Label": final_preds})
submission = submission.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run13_two_stage.csv"
submission.to_csv(out_path, index=False)

print(f"\n✅ Submission saved: {out_path}")
print("\nClass 2 detection stats:")
print(f"  Mean confidence: {class2_confidence.mean():.3f}")
print(f"  Samples with >0.6 confidence: {len(class2_override_idx)}")
print(f"\nClass 4 detection stats:")
print(f"  Mean confidence: {class4_confidence.mean():.3f}")
print(f"  Samples with >0.65 confidence: {len(class4_override_idx)}")