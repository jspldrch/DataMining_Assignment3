"""
model_run17.py — Threshold-optimized ensemble

Key improvements over run07 (0.7707):
1. Extended feature set (+55 features targeting Class 2 vs 1 confusion):
   - Temporal trend (linear slope) of magnitude signals
   - First/second half comparison (stair descent has directional change)
   - 5-segment breakdown of mag_jerk and mag_mean
   - Second-order jerk (jerk-of-jerk) statistics
   - Additional autocorrelation lags and energy features
2. Soft threshold optimization via LOO-CV probabilities to maximize macro F1
   (run07 uses argmax which maximizes accuracy, not F1)
3. Larger ensemble: 10 seeds × 3 LGB configs = 30 models

Root cause of Class 2 failure: walking downstairs has a consistent downward
trend in magnitude that flat walking lacks. Temporal trend features capture this.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import lightgbm as lgb
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
print("LOADING NPZ DATA")
print("=" * 60)

try:
    train_path = find_npz("train_data.npz")
    test_path  = find_npz("test_data.npz")
    print(f"Train data: {train_path}")
    print(f"Test data:  {test_path}")
    tr = np.load(train_path, allow_pickle=True)
    te = np.load(test_path,  allow_pickle=True)
except Exception as e:
    print(f"Kaggle path search failed ({e}), trying current directory...")
    tr = np.load("train_data.npz", allow_pickle=True)
    te = np.load("test_data.npz",  allow_pickle=True)

X_tr_raw = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_tr     = tr["y"].astype(np.int32)
users    = tr["users"]
X_te_raw = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
te_ids   = te["file_ids"]
te_users = te["users"]

unique, counts = np.unique(y_tr, return_counts=True)
print(f"\nTrain shape: {X_tr_raw.shape}")
print(f"Test shape:  {X_te_raw.shape}")
print(f"Training users: {len(np.unique(users))}")
print("Class distribution:")
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
        mu = data.mean(axis=(0, 1), keepdims=True)
        sig = data.std(axis=(0, 1), keepdims=True) + 1e-8
        X_out[idx] = (data - mu) / sig
    return X_out

print("\nPer-user normalization...")
X_tr = user_normalise(X_tr_raw, users)
X_te = user_normalise(X_te_raw, te_users)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def stats9(s):
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s, 1),
            np.percentile(s, 75, 1)-np.percentile(s, 25, 1),
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
                  psd[(freqs >= 0) & (freqs < 0.5)].sum(),
                  psd[(freqs >= 0.5) & (freqs < 2)].sum(),
                  psd[freqs >= 2].sum()]
    return out

def ac(s, lag):
    s1, s2 = s[:, :-lag], s[:, lag:]
    num = ((s1-s1.mean(1, keepdims=True))*(s2-s2.mean(1, keepdims=True))).mean(1)
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
    ca = a - a.mean(1, keepdims=True)
    cb = b - b.mean(1, keepdims=True)
    return (ca*cb).mean(1) / (ca.std(1)*cb.std(1)+1e-10)

def linear_slope(s):
    """Linear regression slope of signal over time — detects trends like descent."""
    N, T = s.shape
    t = np.arange(T, dtype=np.float32)
    t_c = t - t.mean()
    t_var = (t_c**2).sum()
    return (s * t_c).sum(1) / t_var


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (run07's 373 features + 55 new = 428 total)
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

    # ── run07's proven 373 features (exact same order) ────────────────────────
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

    # ── NEW: Temporal trend features (6) ─────────────────────────────────────
    # Walking downstairs has a distinct downward slope in magnitude over time.
    parts.append(linear_slope(mag_mean))
    parts.append(linear_slope(mag_jerk))
    parts.append(linear_slope(mag_std))
    parts.append(linear_slope(sx))
    parts.append(linear_slope(sy))
    parts.append(linear_slope(sz))

    # ── NEW: First/second half comparison (6) ─────────────────────────────────
    # Stair descent has directional change; flat walking is more stationary.
    for sig in [mag_mean, mag_jerk, mag_std]:
        h = sig.shape[1] // 2
        first, second = sig[:, :h], sig[:, h:]
        parts.append(second.mean(1) - first.mean(1))
        parts.append(second.std(1) / (first.std(1) + 1e-8))

    # ── NEW: 5-segment breakdown (20) ─────────────────────────────────────────
    # Finer temporal resolution than run07's 10/20 segments.
    parts += seg(mag_jerk, 5)
    parts += seg(mag_mean, 5)

    # ── NEW: Second-order jerk stats9 (9) ────────────────────────────────────
    # Captures the rate of change of acceleration change.
    jjx = np.diff(jx, axis=1)
    jjy = np.diff(jy, axis=1)
    jjz = np.diff(jz, axis=1)
    mag_jerk2 = np.sqrt(jjx**2 + jjy**2 + jjz**2)
    parts += stats9(mag_jerk2)

    # ── NEW: Additional autocorrelation lags for mag_jerk (5) ────────────────
    for lag in [3, 7, 15, 45, 90]:
        parts.append(ac(mag_jerk, lag))

    # ── NEW: Cross-correlation mag_mean vs mag_jerk (1) ──────────────────────
    # mag_mean is T=300, mag_jerk is T=299 (diff reduces by 1), trim to match.
    parts.append(xcorr(mag_mean[:, :-1], mag_jerk))

    # ── NEW: Temporal energy distribution (8) ────────────────────────────────
    # Energy in start/mid/end of window reveals directional patterns.
    s20 = T // 5
    for sig in [mag_jerk, mag_std]:
        e_start = (sig[:, :s20]**2).mean(1)
        e_mid   = (sig[:, 2*s20:3*s20]**2).mean(1)
        e_end   = (sig[:, 4*s20:]**2).mean(1)
        parts.append(e_start)
        parts.append(e_mid)
        parts.append(e_end)
        parts.append(e_end / (e_start + 1e-8))

    return np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1) for p in parts
    ]).astype(np.float32)

print("\nExtracting features...")
X_tr_feat = extract(X_tr)
X_te_feat = extract(X_te)
print(f"  Train: {X_tr_feat.shape}, Test: {X_te_feat.shape}")

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_feat)
X_te_sc = scaler.transform(X_te_feat)


# ──────────────────────────────────────────────────────────────────────────────
# LOO-CV WITH PROBABILITY COLLECTION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOO-CV — collecting probabilities for threshold optimization")
print("="*60)

unique_users = np.unique(users)
user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids = np.array([user_folds[u] for u in users])

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

    fold_probas = []
    for seed in [42, 7, 13]:   # 3 seeds in CV (fast but representative)
        for cfg in CONFIGS:
            m = lgb.LGBMClassifier(
                n_estimators=500,
                class_weight="balanced",
                min_child_samples=20,
                reg_alpha=0.5, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, verbose=-1, **cfg,
            )
            m.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
            fold_probas.append(m.predict_proba(X_tr_sc[va_idx]))

    loo_probas[va_idx] = np.mean(fold_probas, axis=0)
    fold_preds = loo_probas[va_idx].argmax(1)
    f1 = f1_score(y_tr[va_idx], fold_preds, average='macro')
    acc = accuracy_score(y_tr[va_idx], fold_preds)
    print(f"  Macro F1 = {f1:.4f}  Acc = {acc:.4f}")

baseline_f1 = f1_score(y_tr, loo_probas.argmax(1), average='macro')
print(f"\nBaseline CV macro F1 (argmax): {baseline_f1:.4f}")

per_class_f1 = f1_score(y_tr, loo_probas.argmax(1), average=None)
print("Per-class F1:", [f"{f:.3f}" for f in per_class_f1])


# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# Finds per-class log-scale factors that maximize macro F1.
# This is the key algorithmic difference vs run07's frequency-ratio boost.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION")
print("="*60)

def neg_macro_f1(log_scales, proba, y_true):
    scales = np.exp(log_scales)
    scaled = proba * scales
    scaled /= scaled.sum(axis=1, keepdims=True)
    preds = scaled.argmax(axis=1)
    return -f1_score(y_true, preds, average='macro')

print("Optimizing scales (Nelder-Mead)...")
best_result = None
best_f1 = -np.inf

# Multiple restarts to avoid local minima
for x0_seed in range(5):
    rng = np.random.RandomState(x0_seed * 17)
    x0 = rng.uniform(-0.5, 0.5, 6)
    result = minimize(
        neg_macro_f1,
        x0=x0,
        args=(loo_probas, y_tr),
        method='Nelder-Mead',
        options={'maxiter': 100000, 'xatol': 1e-8, 'fatol': 1e-8},
    )
    trial_f1 = -result.fun
    if trial_f1 > best_f1:
        best_f1 = trial_f1
        best_result = result
    print(f"  Restart {x0_seed}: F1 = {trial_f1:.4f}")

optimal_scales = np.exp(best_result.x)
print(f"\nOptimal scales: {np.round(optimal_scales, 3)}")

opt_probas = loo_probas * optimal_scales
opt_probas /= opt_probas.sum(axis=1, keepdims=True)
opt_preds = opt_probas.argmax(1)
opt_f1 = f1_score(y_tr, opt_preds, average='macro')
opt_per_class = f1_score(y_tr, opt_preds, average=None)
print(f"CV macro F1 after threshold opt: {opt_f1:.4f} (was {baseline_f1:.4f})")
print("Per-class F1:", [f"{f:.3f}" for f in opt_per_class])
print(f"Improvement: {opt_f1 - baseline_f1:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# FINAL TRAINING — 10 seeds × 3 configs = 30 LightGBM models
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL TRAINING (30 models)")
print("="*60)

final_probas = []
for seed in range(10):
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_child_samples=20,
            reg_alpha=0.5, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc, y_tr)
        final_probas.append(m.predict_proba(X_te_sc))
    if (seed + 1) % 2 == 0:
        print(f"  Trained {(seed+1)*3} / 30 models")

avg_proba = np.mean(final_probas, axis=0)

# Apply optimized scales to test predictions
scaled_test = avg_proba * optimal_scales
scaled_test /= scaled_test.sum(axis=1, keepdims=True)
preds = scaled_test.argmax(1)


# ──────────────────────────────────────────────────────────────────────────────
# SAVE SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run17.csv"
sub.to_csv(out_path, index=False)

print(f"\n✅ Submission saved: {out_path}")
print("\nPrediction distribution (predicted vs expected from train):")
for c in range(6):
    cnt = (preds == c).sum()
    exp = int(len(preds) * counts[c] / len(y_tr))
    print(f"  Class {c}: {cnt:5d}  (train rate suggests ~{exp})")

print(f"\nOptimal class scales: {np.round(optimal_scales, 3)}")
print(f"CV macro F1 (threshold-optimized): {opt_f1:.4f}")
print(f"Baseline CV macro F1 (argmax):     {baseline_f1:.4f}")
