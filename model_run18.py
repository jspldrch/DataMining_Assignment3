"""
model_run18.py — Augmentation + user-contextual features + diverse ensemble

Improvements over run07 (0.7707):
1. Data augmentation for rare classes (noise injection after normalization)
   - Class 2: 358 → ~1790 windows  Class 4: 142 → ~710  Class 5: 526 → ~1052
2. User-contextual features (45 z-score features):
   - For each window, how much do its key stats deviate from that user's median?
   - Walking downstairs is "unusual" for a user who mostly walks flat → high z-score
   - Computed from same-user windows at both train and test time (no label leakage)
3. Diverse ensemble: LightGBM (15) + XGBoost (9) = 24 models
4. Threshold optimization via LOO-CV to maximize macro F1 (not accuracy)
5. Fixed CV bug from run17: validation was double-scaled; now uses raw fold features
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def find_npz(name):
    search_paths = [
        Path("/kaggle/input") / name,
        Path("/kaggle/input/train-data") / name,
        Path("/kaggle/input/test-data") / name,
        Path("/kaggle/input/har-data") / name,
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot find {name} in /kaggle/input/")

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

try:
    tr = np.load(find_npz("train_data.npz"), allow_pickle=True)
    te = np.load(find_npz("test_data.npz"),  allow_pickle=True)
except Exception as e:
    print(f"Kaggle path failed ({e}), trying cwd...")
    tr = np.load("train_data.npz", allow_pickle=True)
    te = np.load("test_data.npz",  allow_pickle=True)

X_tr_raw = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_tr     = tr["y"].astype(np.int32)
users    = tr["users"]
X_te_raw = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
te_ids   = te["file_ids"]
te_users = te["users"]

unique, counts = np.unique(y_tr, return_counts=True)
print(f"Train: {X_tr_raw.shape}  Test: {X_te_raw.shape}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_tr)*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# PER-USER NORMALIZATION (same as run07)
# ──────────────────────────────────────────────────────────────────────────────
def user_normalise(X, user_ids):
    X_out = X.copy()
    for uid in np.unique(user_ids):
        idx = np.where(user_ids == uid)[0]
        data = X[idx]
        mu  = data.mean(axis=(0, 1), keepdims=True)
        sig = data.std(axis=(0, 1), keepdims=True) + 1e-8
        X_out[idx] = (data - mu) / sig
    return X_out

print("\nPer-user normalization...")
X_tr = user_normalise(X_tr_raw, users)
X_te = user_normalise(X_te_raw, te_users)


# ──────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION FOR RARE CLASSES
# Inject Gaussian noise after normalization so noise scale is meaningful (σ≈1).
# Returns augmented X, y, and user_ids (so context features know which user).
# ──────────────────────────────────────────────────────────────────────────────
AUG_CONFIG    = {2: 5, 4: 5, 5: 2, 3: 1}   # copies per original window per class
AUG_NOISE_STD = 0.05

def augment_minority(X, y, user_ids, aug_config, noise_std, seed):
    rng = np.random.RandomState(seed)
    aug_X = [X]; aug_y = [y]; aug_u = [user_ids]
    for cls, n_copies in aug_config.items():
        idx = np.where(y == cls)[0]
        for _ in range(n_copies):
            noise = rng.normal(0, noise_std, X[idx].shape).astype(np.float32)
            aug_X.append(X[idx] + noise)
            aug_y.append(y[idx])
            aug_u.append(user_ids[idx])
    X_aug = np.vstack(aug_X)
    y_aug = np.concatenate(aug_y)
    u_aug = np.concatenate(aug_u)
    perm  = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm], u_aug[perm]

print("\nAugmenting minority classes...")
X_tr_aug, y_tr_aug, users_aug = augment_minority(
    X_tr, y_tr, users, AUG_CONFIG, AUG_NOISE_STD, SEED
)
aug_unique, aug_counts = np.unique(y_tr_aug, return_counts=True)
print(f"  {len(y_tr)} → {len(y_tr_aug)} samples")
for u, c in zip(aug_unique, aug_counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_tr_aug)*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def stats9(s):
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s, 1),
            np.percentile(s, 75, 1)-np.percentile(s, 25, 1),
            np.array([skew(r)     for r in s]),
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
                  psd[(freqs >= 0)   & (freqs < 0.5)].sum(),
                  psd[(freqs >= 0.5) & (freqs < 2.0)].sum(),
                  psd[freqs >= 2.0].sum()]
    return out

def ac(s, lag):
    s1, s2 = s[:, :-lag], s[:, lag:]
    num = ((s1-s1.mean(1, keepdims=True))*(s2-s2.mean(1, keepdims=True))).mean(1)
    return num / (s1.std(1)*s2.std(1)+1e-10)

def seg(s, n_seg):
    N, T = s.shape; sl = T // n_seg; out = []
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1)]
    return out

def xcorr(a, b):
    ca = a - a.mean(1, keepdims=True)
    cb = b - b.mean(1, keepdims=True)
    return (ca*cb).mean(1) / (ca.std(1)*cb.std(1)+1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION — run07's proven 373 features
# ──────────────────────────────────────────────────────────────────────────────
def extract(X):
    N, T, _ = X.shape
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]

    jx = np.diff(mx, axis=1)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)

    mag_mean = np.sqrt(mx**2 + my**2 + mz**2)
    mag_std  = np.sqrt(sx**2 + sy**2 + sz**2)
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
    for ch in [sx, sy, sz]:
        parts += seg(ch, 10)
    for ch in [jx, jy, jz]:
        parts += seg(ch, 10)

    for lag in [1, 2, 5, 10, 20, 30, 60]:
        parts.append(ac(mag_jerk, lag))
    for ch in [sx, sy, sz]:
        for lag in [1, 5, 10, 30]:
            parts.append(ac(ch, lag))

    for sig in [mag_jerk, mag_std, sx, sy, sz]:
        parts.append(spectral5(sig))

    for a, b in [(jx, jy), (jx, jz), (jy, jz)]:
        parts.append(xcorr(a, b))
    for a, b in [(sx, sy), (sx, sz), (sy, sz)]:
        parts.append(xcorr(a, b))

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


# ──────────────────────────────────────────────────────────────────────────────
# USER-CONTEXTUAL FEATURES
# For each window, compute how much its key statistics deviate from that user's
# typical behaviour (z-score relative to user mean/std).
#
# Motivation: walking downstairs is unusual for users who mostly walk flat.
# A Class 2 window will have an abnormally high jerk z-score relative to that
# user's baseline — providing a strong Class 2 signal the base features miss.
#
# Uses the first 45 features (stats9 × 5 channels: sx,sy,sz,mag_std,mag_mean).
# Reference statistics are computed from a reference dataset (default: same data).
# → At train time: reference = original training windows (not augmented copies).
# → At test time:  reference = all test windows for that user.
# ──────────────────────────────────────────────────────────────────────────────
N_CTX = 45   # stats9 × 5 key channels

def add_user_context(X_feat, user_ids, ref_feat=None, ref_user_ids=None):
    """Append N_CTX z-score features relative to per-user statistics."""
    if ref_feat is None:
        ref_feat, ref_user_ids = X_feat, user_ids

    user_mean = {}
    user_std  = {}
    for uid in np.unique(ref_user_ids):
        idx = np.where(ref_user_ids == uid)[0]
        user_mean[uid] = ref_feat[idx, :N_CTX].mean(axis=0)
        user_std[uid]  = ref_feat[idx, :N_CTX].std(axis=0) + 1e-8

    ctx = np.zeros((len(X_feat), N_CTX), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        if uid in user_mean:
            ctx[i] = (X_feat[i, :N_CTX] - user_mean[uid]) / user_std[uid]

    return np.hstack([X_feat, ctx]).astype(np.float32)


print("\nExtracting features...")
X_tr_orig_feat = extract(X_tr)                  # original, un-augmented (373)
X_tr_aug_feat  = extract(X_tr_aug)              # augmented (373)
X_te_feat      = extract(X_te)                  # test (373)
print(f"  Original train: {X_tr_orig_feat.shape}")
print(f"  Augmented train: {X_tr_aug_feat.shape}")
print(f"  Test: {X_te_feat.shape}")

# Add user-contextual features (373 → 418)
# Training: reference stats from original (non-augmented) user windows
X_tr_aug_ctx = add_user_context(
    X_tr_aug_feat, users_aug,
    ref_feat=X_tr_orig_feat, ref_user_ids=users
)
# Test: reference stats from test user windows (all test windows known at inference)
X_te_ctx = add_user_context(X_te_feat, te_users)
print(f"\nWith user-context features:")
print(f"  Augmented train: {X_tr_aug_ctx.shape}")
print(f"  Test: {X_te_ctx.shape}")

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_aug_ctx)
X_te_sc = scaler.transform(X_te_ctx)


# ──────────────────────────────────────────────────────────────────────────────
# LOO-CV WITH PROBABILITY COLLECTION
# Each fold augments only its training portion, computes user context correctly,
# and scales validation with the fold's own scaler (no double-scaling bug).
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOO-CV — collecting probabilities for threshold optimization")
print("="*60)

unique_users = np.unique(users)
user_folds   = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids     = np.array([user_folds[u] for u in users])

CONFIGS = [
    dict(num_leaves=31,  learning_rate=0.05, colsample_bytree=0.7, subsample=0.7),
    dict(num_leaves=63,  learning_rate=0.03, colsample_bytree=0.8, subsample=0.8),
    dict(num_leaves=127, learning_rate=0.02, colsample_bytree=0.7, subsample=0.7),
]

loo_probas = np.zeros((len(y_tr), 6), dtype=np.float64)

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")

    # Augment training portion of this fold
    X_fold_aug, y_fold_aug, users_fold_aug = augment_minority(
        X_tr[tr_idx], y_tr[tr_idx], users[tr_idx],
        AUG_CONFIG, AUG_NOISE_STD, SEED + fold
    )
    # Extract features and add user context
    X_fold_feat = extract(X_fold_aug)
    X_fold_orig = X_tr_orig_feat[tr_idx]          # original fold features for ref
    X_fold_ctx  = add_user_context(
        X_fold_feat, users_fold_aug,
        ref_feat=X_fold_orig, ref_user_ids=users[tr_idx]
    )

    # Validation: original features + user context (reference = val users' own windows)
    X_va_feat = X_tr_orig_feat[va_idx]
    X_va_ctx  = add_user_context(X_va_feat, users[va_idx])

    # Scale: fit on training fold, apply to both
    sc_fold  = StandardScaler()
    X_fold_sc = sc_fold.fit_transform(X_fold_ctx)
    X_va_sc   = sc_fold.transform(X_va_ctx)

    fold_probas = []
    for seed in [42, 7, 13]:
        for cfg in CONFIGS:
            m = lgb.LGBMClassifier(
                n_estimators=500,
                class_weight="balanced",
                min_child_samples=20,
                reg_alpha=0.5, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, verbose=-1, **cfg,
            )
            m.fit(X_fold_sc, y_fold_aug)
            fold_probas.append(m.predict_proba(X_va_sc))

    loo_probas[va_idx] = np.mean(fold_probas, axis=0)
    fold_preds = loo_probas[va_idx].argmax(1)
    print(f"  Macro F1 = {f1_score(y_tr[va_idx], fold_preds, average='macro'):.4f}"
          f"  Acc = {accuracy_score(y_tr[va_idx], fold_preds):.4f}")

baseline_f1    = f1_score(y_tr, loo_probas.argmax(1), average='macro')
per_class_f1   = f1_score(y_tr, loo_probas.argmax(1), average=None)
print(f"\nBaseline CV macro F1: {baseline_f1:.4f}")
print("Per-class F1:", [f"{f:.3f}" for f in per_class_f1])


# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION")
print("="*60)

def neg_macro_f1(log_scales, proba, y_true):
    scales = np.exp(log_scales)
    scaled = proba * scales
    scaled /= scaled.sum(axis=1, keepdims=True)
    return -f1_score(y_true, scaled.argmax(axis=1), average='macro')

best_result, best_f1 = None, -np.inf
for x0_seed in range(8):
    rng = np.random.RandomState(x0_seed * 17)
    x0  = rng.uniform(-0.5, 0.5, 6)
    res = minimize(
        neg_macro_f1, x0=x0, args=(loo_probas, y_tr),
        method='Nelder-Mead',
        options={'maxiter': 100000, 'xatol': 1e-8, 'fatol': 1e-8},
    )
    if -res.fun > best_f1:
        best_f1, best_result = -res.fun, res
    print(f"  Restart {x0_seed}: F1 = {-res.fun:.4f}")

optimal_scales = np.exp(best_result.x)
opt_probas = loo_probas * optimal_scales
opt_probas /= opt_probas.sum(axis=1, keepdims=True)
opt_f1         = f1_score(y_tr, opt_probas.argmax(1), average='macro')
opt_per_class  = f1_score(y_tr, opt_probas.argmax(1), average=None)

print(f"\nOptimal scales:  {np.round(optimal_scales, 3)}")
print(f"CV F1 after opt: {opt_f1:.4f}  (was {baseline_f1:.4f}, +{opt_f1-baseline_f1:.4f})")
print("Per-class F1:", [f"{f:.3f}" for f in opt_per_class])


# ──────────────────────────────────────────────────────────────────────────────
# FINAL TRAINING — LightGBM (15) + XGBoost (9) = 24 models
# XGBoost grows trees level-wise vs LightGBM's leaf-wise → uncorrelated errors.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL TRAINING (15 LightGBM + 9 XGBoost = 24 models)")
print("="*60)

sw_aug = compute_sample_weight('balanced', y_tr_aug)   # for XGBoost

XGB_CONFIGS = [
    dict(max_depth=6, learning_rate=0.05, n_estimators=400,
         subsample=0.8, colsample_bytree=0.7),
    dict(max_depth=8, learning_rate=0.03, n_estimators=500,
         subsample=0.7, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.05, n_estimators=500,
         subsample=0.9, colsample_bytree=0.6),
]

final_probas = []

# LightGBM: 5 seeds × 3 configs = 15 models
for seed in range(5):
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_child_samples=20,
            reg_alpha=0.5, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc, y_tr_aug)
        final_probas.append(m.predict_proba(X_te_sc))
print(f"  LightGBM: {len(final_probas)} models trained")

# XGBoost: 3 seeds × 3 configs = 9 models
xgb_start = len(final_probas)
for seed in range(3):
    for cfg in XGB_CONFIGS:
        xm = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=6,
            tree_method='hist',
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            eval_metric='mlogloss',
            **cfg,
        )
        xm.fit(X_tr_sc, y_tr_aug, sample_weight=sw_aug)
        final_probas.append(xm.predict_proba(X_te_sc))
print(f"  XGBoost:  {len(final_probas) - xgb_start} models trained")
print(f"  Total:    {len(final_probas)} models")

avg_proba = np.mean(final_probas, axis=0)

# Apply optimized thresholds
scaled_test = avg_proba * optimal_scales
scaled_test /= scaled_test.sum(axis=1, keepdims=True)
preds = scaled_test.argmax(1)


# ──────────────────────────────────────────────────────────────────────────────
# SAVE SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run18.csv"
sub.to_csv(out_path, index=False)

print(f"\n✅ Submission saved: {out_path}")
print("\nPrediction distribution (predicted vs expected from train rate):")
for c in range(6):
    cnt = (preds == c).sum()
    exp = int(len(preds) * counts[c] / len(y_tr))
    print(f"  Class {c}: {cnt:5d}  (expected ~{exp})")

print(f"\nOptimal scales:          {np.round(optimal_scales, 3)}")
print(f"CV F1 (threshold-opt):   {opt_f1:.4f}")
print(f"CV F1 (baseline argmax): {baseline_f1:.4f}")
