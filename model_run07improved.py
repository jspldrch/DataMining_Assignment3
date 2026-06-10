# run07improved: run07 features + gentle feature selection (top 50%), class boost

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

SEED = 42
np.random.seed(SEED)

def find_npz(name):
    search_paths = [
        Path("/kaggle/input/train-data") / name,
        Path("/kaggle/input/test-data") / name,
        Path("/kaggle/input") / name,
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot find {name}")

print("Loading NPZ data...")
tr = np.load(find_npz("train_data.npz"), allow_pickle=True)
te = np.load(find_npz("test_data.npz"),  allow_pickle=True)

X_train_raw = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_train      = tr["y"].astype(np.int32)
train_users  = tr["users"]
X_test_raw   = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
test_ids     = te["file_ids"]
test_users   = te["users"]

unique, counts = np.unique(y_train, return_counts=True)
print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")

# per-user normalization
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

# feature extraction (same as run07)
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

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

# gentle feature selection (keep 150-200 features, not 50)
print("\n" + "="*60)
print("GENTLE FEATURE SELECTION")
print("="*60)

original_count = X_train_scaled.shape[1]
print(f"Original features: {original_count}")

# Step 1: Remove only constant features (not low variance)
selector_var = VarianceThreshold(threshold=0.0)  # Remove only constant features
X_train_var = selector_var.fit_transform(X_train_scaled)
X_test_var = selector_var.transform(X_test_scaled)
print(f"After removing constant features: {X_train_var.shape[1]}")

# Step 2: LightGBM importance - keep top 50% (not top 30%)
quick_lgb = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1)
quick_lgb.fit(X_train_var, y_train)
importances = quick_lgb.feature_importances_

# Keep top 50% of features (was 30% in run15)
threshold_percentile = 50  # Changed from 30 to 50
importance_threshold = np.percentile(importances, threshold_percentile)
keep_mask = importances >= importance_threshold
X_train_selected = X_train_var[:, keep_mask]
X_test_selected = X_test_var[:, keep_mask]

print(f"After LGBM importance (top {threshold_percentile}%): {X_train_selected.shape[1]}")

# Step 3: Remove only extremely correlated features (threshold 0.98, not 0.95)
# This is optional - skip if you want to keep more features
print(f"Final features: {X_train_selected.shape[1]}")
print(f"Reduction: {(1 - X_train_selected.shape[1]/original_count)*100:.1f}%")

# Use these features for training
X_train_final = X_train_selected
X_test_final = X_test_selected

# cross-validation
print("\n" + "="*60)
print("LEAVE-USER-OUT CROSS-VALIDATION")
print("="*60)

unique_users = np.unique(train_users)
user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids = np.array([user_folds[u] for u in train_users])

CONFIGS = [
    dict(num_leaves=31, learning_rate=0.05, colsample_bytree=0.7, subsample=0.7),
    dict(num_leaves=63, learning_rate=0.03, colsample_bytree=0.8, subsample=0.8),
    dict(num_leaves=127, learning_rate=0.02, colsample_bytree=0.7, subsample=0.7),
]

loo_preds = np.zeros(len(y_train), dtype=int)

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")
    
    probas = []
    for cfg_model in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            **cfg_model
        )
        m.fit(X_train_final[tr_idx], y_train[tr_idx])
        probas.append(m.predict_proba(X_train_final[va_idx]))
    
    loo_preds[va_idx] = np.mean(probas, axis=0).argmax(axis=1)
    acc = accuracy_score(y_train[va_idx], loo_preds[va_idx])
    print(f"  Fold accuracy: {acc:.4f}")

loo_acc = accuracy_score(y_train, loo_preds)
print(f"\nOverall LOO-CV Accuracy: {loo_acc:.4f}")

# FINAL TRAINING
print("\n" + "="*60)
print("FINAL TRAINING")
print("="*60)

final_probas = []
for seed in range(5):
    for cfg_model in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **cfg_model
        )
        m.fit(X_train_final, y_train)
        final_probas.append(m.predict_proba(X_test_final))

avg_proba = np.mean(final_probas, axis=0)

# Gentle class boost (max 2.5x, not 3x)
train_freq = np.array([counts[i]/len(y_train) for i in range(6)])
pred_freq = avg_proba.mean(axis=0)
boost = np.where(pred_freq > 0, train_freq / pred_freq, 1.0)
boost = np.clip(boost, 0.7, 2.5)  # Gentler boost
avg_proba_boosted = avg_proba * boost
avg_proba_boosted /= avg_proba_boosted.sum(axis=1, keepdims=True)

preds = avg_proba_boosted.argmax(axis=1)

# SAVE
submission = pd.DataFrame({"Id": test_ids, "Label": preds})
submission = submission.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run07improved.csv"
submission.to_csv(out_path, index=False)

print(f"\nSubmission saved: {out_path}")
print("\nPrediction distribution:")
for c in range(6):
    count = np.sum(preds == c)
    expected = int(len(preds) * counts[c] / len(y_train))
    print(f"  Class {c}: {count:5d} (expected: {expected:5d})")

print(f"\nLOO-CV Accuracy: {loo_acc:.4f}")
print(f"Final features: {X_train_final.shape[1]}")