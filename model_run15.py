"""
model_run15.py — Aggressive Feature Selection + LightGBM
Uses original CSV data (not NPZ) with aggressive feature pruning

Key characteristics:
  - Loads original CSV files
  - Per-user normalization
  - Aggressive feature selection (373 → 50 features, 86.6% reduction)
  - LightGBM ensemble (15 models)
  
Score: 0.7565
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

SEED = 42
np.random.seed(SEED)

# ============================================================================
# LOAD ORIGINAL CSV DATA
# ============================================================================
def load_csv_data(train_path, test_path):
    train_path = Path(train_path)
    test_path = Path(test_path)
    feature_cols = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
    seq_len = 300
    
    # Load training data
    print("Loading training data...")
    X_train, y_train, train_users, train_file_ids = [], [], [], []
    
    for user_dir in sorted(train_path.iterdir()):
        if not user_dir.is_dir():
            continue
        user_name = user_dir.name
        for csv_path in sorted(user_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            features = df[feature_cols].values.astype(np.float32)
            if len(features) != seq_len:
                if len(features) < seq_len:
                    pad = np.zeros((seq_len - len(features), len(feature_cols)))
                    features = np.vstack([features, pad])
                else:
                    features = features[:seq_len]
            X_train.append(features)
            y_train.append(int(df["label"].iloc[0]))
            train_users.append(user_name)
            train_file_ids.append(int(df["file_id"].iloc[0]))
    
    # Load test data
    print("Loading test data...")
    X_test, test_ids, test_users = [], [], []
    
    for user_dir in sorted(test_path.iterdir()):
        if not user_dir.is_dir():
            continue
        user_name = user_dir.name
        for csv_path in sorted(user_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            features = df[feature_cols].values.astype(np.float32)
            if len(features) != seq_len:
                if len(features) < seq_len:
                    pad = np.zeros((seq_len - len(features), len(feature_cols)))
                    features = np.vstack([features, pad])
                else:
                    features = features[:seq_len]
            X_test.append(features)
            test_ids.append(int(df["file_id"].iloc[0]))
            test_users.append(user_name)
    
    return (np.array(X_train), np.array(y_train), np.array(train_users), 
            np.array(train_file_ids), np.array(X_test), np.array(test_ids), 
            np.array(test_users))

# Load data
print("\n" + "="*60)
print("LOADING ORIGINAL CSV DATA")
print("="*60)

X_train_raw, y_train, train_users, train_file_ids, X_test_raw, test_ids, test_users = load_csv_data(
    "train/train", "test/test"
)

unique, counts = np.unique(y_train, return_counts=True)
print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")

# ============================================================================
# PER-USER NORMALIZATION
# ============================================================================
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

# ============================================================================
# FEATURE EXTRACTION (373 features)
# ============================================================================
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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

# ============================================================================
# AGGRESSIVE FEATURE SELECTION (373 → 50 features)
# ============================================================================
print("\n" + "="*60)
print("AGGRESSIVE FEATURE SELECTION")
print("="*60)

original_count = X_train_scaled.shape[1]
print(f"Original features: {original_count}")

# Step 1: Variance threshold (remove constant features)
selector_var = VarianceThreshold(threshold=0.01)
X_train_var = selector_var.fit_transform(X_train_scaled)
X_test_var = selector_var.transform(X_test_scaled)
print(f"After variance threshold: {X_train_var.shape[1]}")

# Step 2: Correlation-based elimination (remove highly correlated)
corr_matrix = pd.DataFrame(X_train_var).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]

if high_corr:
    keep_cols = [c for c in range(X_train_var.shape[1]) if c not in high_corr]
    X_train_corr = X_train_var[:, keep_cols]
    X_test_corr = X_test_var[:, keep_cols]
    print(f"After correlation removal: {X_train_corr.shape[1]}")
else:
    X_train_corr = X_train_var
    X_test_corr = X_test_var
    print(f"No highly correlated features found")

# Step 3: Mutual Information selection (keep top 30%)
mi_scores = mutual_info_classif(X_train_corr, y_train, random_state=SEED)
mi_threshold = np.percentile(mi_scores, 70)  # Keep top 30%
mi_mask = mi_scores >= mi_threshold
X_train_mi = X_train_corr[:, mi_mask]
X_test_mi = X_test_corr[:, mi_mask]
print(f"After mutual info (top 30%): {X_train_mi.shape[1]}")

# Step 4: LightGBM importance (keep top 40%)
quick_lgb = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1)
quick_lgb.fit(X_train_mi, y_train)
importances = quick_lgb.feature_importances_
importance_threshold = np.percentile(importances, 60)  # Keep top 40%
importance_mask = importances >= importance_threshold
X_train_lgb = X_train_mi[:, importance_mask]
X_test_lgb = X_test_mi[:, importance_mask]
print(f"After LGBM importance (top 40%): {X_train_lgb.shape[1]}")

# Step 5: RFECV (final selection)
print("Running RFECV (this may take a few minutes)...")
if X_train_lgb.shape[0] > 5000:
    sample_idx = np.random.choice(X_train_lgb.shape[0], 5000, replace=False)
    X_sample = X_train_lgb[sample_idx]
    y_sample = y_train[sample_idx]
else:
    X_sample = X_train_lgb
    y_sample = y_train

rfecv = RFECV(
    estimator=lgb.LGBMClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, verbose=-1),
    step=10,
    cv=3,
    scoring='accuracy',
    min_features_to_select=40,
    n_jobs=-1
)

try:
    rfecv.fit(X_sample, y_sample)
    X_train_final = rfecv.transform(X_train_lgb)
    X_test_final = rfecv.transform(X_test_lgb)
    print(f"After RFECV: {X_train_final.shape[1]} features")
except Exception as e:
    print(f"RFECV failed: {e}")
    print(f"Using LGBM-selected features as final")
    X_train_final = X_train_lgb
    X_test_final = X_test_lgb

print(f"\n✅ Final features: {X_train_final.shape[1]} (from {original_count} original)")
print(f"   Reduction: {(1 - X_train_final.shape[1]/original_count)*100:.1f}%")

# ============================================================================
# LEAVE-USER-OUT CROSS-VALIDATION
# ============================================================================
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
fold_scores = []

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")
    
    probas = []
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            **cfg
        )
        m.fit(X_train_final[tr_idx], y_train[tr_idx])
        probas.append(m.predict_proba(X_train_final[va_idx]))
    
    loo_preds[va_idx] = np.mean(probas, axis=0).argmax(axis=1)
    acc = accuracy_score(y_train[va_idx], loo_preds[va_idx])
    fold_scores.append(acc)
    print(f"  Fold accuracy: {acc:.4f}")

loo_acc = accuracy_score(y_train, loo_preds)
print(f"\nOverall LOO-CV Accuracy: {loo_acc:.4f}")
print(f"Mean fold accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

# Confusion matrix
cm = confusion_matrix(y_train, loo_preds, normalize='true')
print("\nConfusion Matrix (normalized):")
for i in range(6):
    print(f"  Class {i}: " + " ".join([f"{x:.2f}" for x in cm[i]]))

# ============================================================================
# FINAL TRAINING
# ============================================================================
print("\n" + "="*60)
print("FINAL TRAINING")
print("="*60)

final_probas = []
for seed in range(5):
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **cfg
        )
        m.fit(X_train_final, y_train)
        final_probas.append(m.predict_proba(X_test_final))
        print(f"  Trained model {len(final_probas)}/15")

avg_proba = np.mean(final_probas, axis=0)

# Class boost
train_freq = np.array([counts[i]/len(y_train) for i in range(6)])
pred_freq = avg_proba.mean(axis=0)
boost = np.where(pred_freq > 0, train_freq / pred_freq, 1.0)
boost = np.clip(boost, 0.5, 3.0)
avg_proba_boosted = avg_proba * boost
avg_proba_boosted /= avg_proba_boosted.sum(axis=1, keepdims=True)

preds = avg_proba_boosted.argmax(axis=1)

# ============================================================================
# SAVE SUBMISSION
# ============================================================================
submission = pd.DataFrame({"Id": test_ids, "Label": preds})
submission = submission.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run15.csv"
submission.to_csv(out_path, index=False)

print(f"\n✅ Submission saved: {out_path}")
print("\nPrediction distribution:")
for c in range(6):
    count = np.sum(preds == c)
    expected = int(len(preds) * counts[c] / len(y_train))
    print(f"  Class {c}: {count:5d} (expected: {expected:5d})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("RUN15 SUMMARY")
print("="*60)
print(f"Data source: Original CSV files")
print(f"Training samples: {X_train_raw.shape[0]}")
print(f"Test samples: {X_test_raw.shape[0]}")
print(f"Original features: {original_count}")
print(f"Selected features: {X_train_final.shape[1]}")
print(f"Feature reduction: {(1 - X_train_final.shape[1]/original_count)*100:.1f}%")
print(f"\nLOO-CV Accuracy: {loo_acc:.4f}")
print(f"Final features: {X_train_final.shape[1]}")
print(f"\nScore: 0.7565")
print("="*60)