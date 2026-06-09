"""
model_run32.py — run31 + Gaussian noise augmentation (C2×5, C4×5, C5×2, C3×1)

WHY this run:
  The run summary shows that run27–run31 (all 716-feat runs) were submitted WITHOUT
  augmentation. But run23 vs run25 proved that augmentation gives +0.016 on the
  382-feat set (0.7633 → 0.7792). This is the largest untested combination:
  716-feat + augmentation.

  run31 (716 feat, no aug) = 0.7906   ← current best
  run32 (716 feat, + aug)  = ?.????   ← expected ~0.79+ something

Augmentation strategy (matching run23 which gave the best augmented result):
  C2 walk_down : ×5 noisy copies  ← hardest class, main F1 bottleneck
  C4 running   : ×5 noisy copies
  C5 other     : ×2 noisy copies
  C3 walk_up   : ×1 noisy copy
  Noise: Gaussian N(0, 0.02 g) added to raw (T=300, C=6) time series
         BEFORE feature extraction (physically meaningful, not feature-level noise)

CV correctness:
  GroupKFold splits on original indices only.
  Augmented copies of TRAINING-fold users → added to training fold.
  Validation fold → original samples only (no augmented copies evaluated).
  This avoids data leakage while using all augmented data for training.

Everything else identical to run31:
  716 features (run27/30/31 exact set), no normalization, lr=0.01, patience=300
  5 LGB + 2 XGB + 1 RF + Nelder-Mead threshold optimization (8 restarts)

Runtime estimate (Kaggle, CPU accelerator = OFF, n_jobs=-1 uses all 4 cores):
  Feature extraction: ~20–30 min  (original ~8 min + augmented ~18 min)
  CV (5 folds, lr=0.01, patience=300, ~3× data vs run31): ~90–120 min
  Final training (8 models on full augmented set): ~40–60 min
  Total: ~3–4 hours  →  well within Kaggle's 9-hour CPU limit

Accelerator recommendation: CPU (no GPU)
  LightGBM/XGBoost/RF are CPU-parallel (n_jobs=-1).
  The Kaggle GPU instance gives only 2 CPU cores vs 4 on CPU-only — this
  would make tree training ~2× SLOWER. Do NOT enable GPU accelerator.
"""

import numpy as np
import pandas as pd
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/input").exists() \
          else Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

CLASS_NAMES = ["sit/stand", "walk_flat", "walk_down", "walk_up", "running", "other"]

# Augmentation config (matching run23)
AUG_CONFIG = {2: 5, 4: 5, 5: 2, 3: 1}
NOISE_STD  = 0.02   # 0.02 g — appropriate for wrist accelerometer


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
        if path.exists(): return str(path)
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits: return hits[0]
    local = Path(__file__).parent / "outputs" / name
    if local.exists(): return str(local)
    raise FileNotFoundError(f"Cannot find {name}")

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

X_tr   = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_tr   = tr["y"].astype(np.int32)
users  = tr["users"]
X_te   = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
te_ids = te["file_ids"]

unique, counts = np.unique(y_tr, return_counts=True)
print(f"Train: {X_tr.shape}  Test: {X_te.shape}")
for u, c in zip(unique, counts):
    print(f"  Class {u} {CLASS_NAMES[u]:<12}: {c:5d} ({c/len(y_tr)*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# AUGMENTATION  (raw signal level, before feature extraction)
# ──────────────────────────────────────────────────────────────────────────────
def augment_data(X_raw, y, users_arr, aug_config, noise_std=0.02, seed=42):
    """
    Creates augmented copies of hard-class samples.
    Returns ONLY the new copies (not originals) as separate arrays.
    This keeps original and augmented data easy to track during CV.

    Each copy = original window + independent Gaussian noise on every channel/timestep.
    """
    rng = np.random.RandomState(seed)
    X_list, y_list, u_list = [], [], []
    for cls, n_copies in aug_config.items():
        idx = np.where(y == cls)[0]
        for _ in range(n_copies):
            noise = rng.normal(0, noise_std, X_raw[idx].shape).astype(np.float32)
            X_list.append(X_raw[idx] + noise)
            y_list.append(y[idx])
            u_list.append(users_arr[idx])
    return (np.concatenate(X_list),
            np.concatenate(y_list),
            np.concatenate(u_list))

print("\n" + "=" * 60)
print("DATA AUGMENTATION")
print("=" * 60)
X_aug_raw, y_aug, users_aug = augment_data(X_tr, y_tr, users, AUG_CONFIG, NOISE_STD)
print(f"Original train samples:   {len(X_tr)}")
for cls, n in sorted(AUG_CONFIG.items()):
    cnt = (y_tr == cls).sum()
    print(f"  C{cls} ({CLASS_NAMES[cls]:<10}): {cnt:4d} × {n} copies = {cnt*n:5d} new samples")
print(f"Total augmented copies:   {len(X_aug_raw)}")
print(f"Grand total for training: {len(X_tr) + len(X_aug_raw)}")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE HELPERS — identical to run27/30/31
# ──────────────────────────────────────────────────────────────────────────────
def _safe_skew_vec(ch):
    return np.array([skew(r)     if np.std(r) >= 1e-12 else 0.0 for r in ch], dtype=np.float32)

def _safe_kurtosis_vec(ch):
    return np.array([kurtosis(r) if np.std(r) >= 1e-12 else 0.0 for r in ch], dtype=np.float32)

def stat_features(ch, prefix):
    q = np.percentile(ch, [5, 10, 25, 75, 90, 95], axis=1).T
    m = ch.mean(axis=1); e = (ch**2).mean(axis=1)
    f = np.column_stack([
        m, ch.std(axis=1), ch.min(axis=1), ch.max(axis=1),
        ch.max(axis=1)-ch.min(axis=1), np.median(ch, axis=1),
        q[:,0], q[:,1], q[:,2], q[:,3], q[:,4], q[:,5],
        q[:,3]-q[:,2], e, np.sqrt(np.maximum(e, 0)),
        np.abs(ch - m[:,None]).mean(axis=1),
        _safe_skew_vec(ch), _safe_kurtosis_vec(ch),
    ])
    names = [f"{prefix}_{s}" for s in [
        "mean","std","min","max","range","median",
        "q05","q10","q25","q75","q90","q95","iqr",
        "energy","rms","mad","skew","kurtosis",
    ]]
    return f.astype(np.float32), names

def diff_features(ch, prefix):
    d = np.diff(ch, axis=1); a = np.abs(d)
    f = np.column_stack([
        d.mean(1), d.std(1), a.mean(1), a.max(1), (d**2).mean(1),
        ch[:,:60].mean(1), ch[:,-60:].mean(1),
        ch[:,-60:].mean(1) - ch[:,:60].mean(1),
    ])
    names = [f"{prefix}_{s}" for s in [
        "diff_mean","diff_std","diff_abs_mean","diff_abs_max","diff_energy",
        "first_60_mean","last_60_mean","last_minus_first_60",
    ]]
    return f.astype(np.float32), names

def fft_features(ch, prefix):
    N, T  = ch.shape
    freqs = np.fft.rfftfreq(T, d=1.0)
    out   = np.zeros((N, 6), dtype=np.float32)
    for n in range(N):
        v = ch[n] - ch[n].mean()
        if np.std(v) < 1e-12: continue
        p = np.abs(np.fft.rfft(v)) ** 2
        p, f = p[1:], freqs[1:]
        tot = p.sum()
        if tot <= 0: continue
        dom = np.argmax(p); prob = p / tot
        out[n] = [f[dom], p[dom], tot,
                  -np.sum(prob * np.log(prob + 1e-12)),
                  p[f <= 0.10].sum(), p[f > 0.10].sum()]
    names = [f"{prefix}_fft_{s}" for s in [
        "dom_freq","dom_power","total_power","spectral_entropy","low_power","high_power"]]
    return out, names

def peak_features(ch, prefix):
    N, T = ch.shape; out = np.zeros((N, 4), dtype=np.float32)
    for n in range(N):
        v = ch[n]
        if np.std(v) < 1e-12: continue
        pks, props = find_peaks(v, height=v.mean() + 0.5*v.std())
        out[n, 0] = len(pks); out[n, 1] = len(pks) / T
        if len(pks) > 0:
            out[n, 2] = props["peak_heights"].mean()
            out[n, 3] = props["peak_heights"].max()
    names = [f"{prefix}_{s}" for s in [
        "num_peaks","peak_rate","mean_peak_height","max_peak_height"]]
    return out, names

def window_features(ch, prefix, n_windows=10):
    N, T = ch.shape; ws = T // n_windows
    parts, names = [], []
    for w in range(n_windows):
        s = w*ws; e = (w+1)*ws if w < n_windows-1 else T
        win = ch[:, s:e]
        parts.append(np.column_stack([
            win.mean(1), win.std(1), win.min(1), win.max(1), (win**2).mean(1)]))
        names += [f"win{w}_{prefix}_{st}" for st in ["mean","std","min","max","energy"]]
    return np.concatenate(parts, axis=1).astype(np.float32), names


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION — 716 features, identical to run27/30/31
# ──────────────────────────────────────────────────────────────────────────────
def extract(X):
    N, T, _ = X.shape
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]

    acc_mag    = np.sqrt(mx**2 + my**2 + mz**2)
    std_mag    = np.sqrt(sx**2 + sy**2 + sz**2)
    xy_mag     = np.sqrt(mx**2 + my**2)
    xz_mag     = np.sqrt(mx**2 + mz**2)
    yz_mag     = np.sqrt(my**2 + mz**2)
    std_xy_mag = np.sqrt(sx**2 + sy**2)
    std_xz_mag = np.sqrt(sx**2 + sz**2)
    std_yz_mag = np.sqrt(sy**2 + sz**2)
    mean_sum   = mx + my + mz
    std_sum    = sx + sy + sz

    base = [
        ("mean_x", mx), ("mean_y", my), ("mean_z", mz),
        ("std_x",  sx), ("std_y",  sy), ("std_z",  sz),
        ("acc_mag", acc_mag), ("std_mag", std_mag),
        ("xy_mag",  xy_mag),  ("xz_mag",  xz_mag), ("yz_mag", yz_mag),
        ("std_xy_mag", std_xy_mag), ("std_xz_mag", std_xz_mag), ("std_yz_mag", std_yz_mag),
        ("mean_sum", mean_sum), ("std_sum", std_sum),
    ]
    fft_pk = [("acc_mag", acc_mag), ("std_mag", std_mag),
              ("std_x", sx), ("std_y", sy), ("std_z", sz)]
    wins   = [("acc_mag", acc_mag), ("std_mag", std_mag),
              ("std_x", sx), ("std_y", sy), ("std_z", sz)]

    feats_list, feat_names = [], []

    for name, ch in base:
        sf, sn = stat_features(ch, name)
        df, dn = diff_features(ch, name)
        feats_list += [sf, df]; feat_names += sn + dn

    for name, ch in fft_pk:
        ff, fn = fft_features(ch, name)
        pf, pn = peak_features(ch, name)
        feats_list += [ff, pf]; feat_names += fn + pn

    for name, ch in wins:
        wf, wn = window_features(ch, name)
        feats_list.append(wf); feat_names += wn

    return np.concatenate(feats_list, axis=1), feat_names


print("\nExtracting features from original training data...")
X_tr_feat, feat_names = extract(X_tr)
assert X_tr_feat.shape[1] == 716, f"Expected 716, got {X_tr_feat.shape[1]}"
print(f"  Done: {X_tr_feat.shape}")

print("Extracting features from augmented samples...")
X_aug_feat, _ = extract(X_aug_raw)
assert X_aug_feat.shape[1] == 716
print(f"  Done: {X_aug_feat.shape}")

print("Extracting features from test data...")
X_te_feat, _  = extract(X_te)
assert X_te_feat.shape[1] == 716
print(f"  Done: {X_te_feat.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE CLEANING
# Impute NaN/Inf using training medians only (no leakage from aug or test)
# ──────────────────────────────────────────────────────────────────────────────
def clean_all(F_tr, F_aug, F_te):
    F_tr  = np.where(np.isfinite(F_tr),  F_tr,  np.nan)
    F_aug = np.where(np.isfinite(F_aug), F_aug, np.nan)
    F_te  = np.where(np.isfinite(F_te),  F_te,  np.nan)
    meds  = np.nanmedian(F_tr, axis=0)   # computed from originals only
    for F in [F_tr, F_aug, F_te]:
        nans = np.isnan(F)
        F[nans] = np.take(meds, np.where(nans)[1])
    return (F_tr.astype(np.float32),
            F_aug.astype(np.float32),
            F_te.astype(np.float32))

X_tr_feat, X_aug_feat, X_te_feat = clean_all(X_tr_feat, X_aug_feat, X_te_feat)


# ──────────────────────────────────────────────────────────────────────────────
# GROUPKFOLD CV — with augmentation
#
# Split on original indices.  For each fold:
#   - validation: original samples of held-out users (unchanged)
#   - training:   original + augmented copies of training users
#
# This means augmented copies NEVER appear in validation → no leakage.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GROUPKFOLD CV  (lr=0.01, patience=300, + augmentation)")
print("=" * 60)

gkf        = GroupKFold(n_splits=5)
loo_probas = np.zeros((len(y_tr), 6), dtype=np.float64)  # indexed by original samples
best_iters = []
fold_f1s   = []

for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(X_tr_feat, y_tr, groups=users), start=1):

    # ── Identify augmented copies belonging to training-fold users ──────────
    train_users = set(users[tr_idx])
    aug_mask    = np.array([u in train_users for u in users_aug])

    X_f_tr = np.concatenate([X_tr_feat[tr_idx],  X_aug_feat[aug_mask]])
    y_f_tr = np.concatenate([y_tr[tr_idx],        y_aug[aug_mask]])
    X_f_va = X_tr_feat[va_idx]   # ORIGINAL samples only
    y_f_va = y_tr[va_idx]

    print(f"\nFold {fold}/5  "
          f"train_orig={len(tr_idx)}  aug={aug_mask.sum()}  "
          f"total_train={len(X_f_tr)}  val={len(va_idx)}")

    sw = compute_sample_weight('balanced', y_f_tr)

    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=6,
        n_estimators=5000,
        learning_rate=0.01,          # same as run31
        num_leaves=31, max_depth=-1,
        min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED + fold, n_jobs=-1, verbose=-1,
    )
    m.fit(
        X_f_tr, y_f_tr,
        sample_weight=sw,
        eval_set=[(X_f_va, y_f_va)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=300),
            lgb.log_evaluation(period=300),
        ],
    )

    va_proba = m.predict_proba(X_f_va)
    loo_probas[va_idx] = va_proba
    best_iters.append(m.best_iteration_)

    fold_f1 = f1_score(y_f_va, va_proba.argmax(1), average='macro')
    fold_f1s.append(fold_f1)
    print(f"  best_iter={m.best_iteration_:4d}  "
          f"Macro F1={fold_f1:.4f}  Acc={accuracy_score(y_f_va, va_proba.argmax(1)):.4f}")

avg_best_iter = int(np.mean(best_iters) * 1.10)
print(f"\nBest iters per fold: {best_iters}  →  final n_estimators = {avg_best_iter}")
print(f"  run31 (no aug): expected ~400–600")

loo_preds    = loo_probas.argmax(1)
baseline_f1  = f1_score(y_tr, loo_preds, average='macro')
per_class_f1 = f1_score(y_tr, loo_preds, average=None)
print(f"\nOOF macro F1 : {baseline_f1:.4f}")
print(f"Mean fold F1 : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
print("Per-class F1 :", [f"{f:.3f}" for f in per_class_f1])
print(f"  C2 walk_down: {per_class_f1[2]:.3f}  (run31 target: >0.244)")
print("\n" + classification_report(y_tr, loo_preds, target_names=CLASS_NAMES, digits=4))


# ──────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_tr, loo_preds, normalize='true')
print("CV Confusion Matrix (row=true, col=predicted):")
print(f"  {'':16}" + "".join(f"    C{c}" for c in range(6)))
for i in range(6):
    row  = "".join(f"  {cm[i,j]:.2f}" for j in range(6))
    flag = "  ← C2 bottleneck" if i == 2 else ""
    print(f"  C{i} {CLASS_NAMES[i]:<14}{row}{flag}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title(
    f'CV Confusion Matrix — run32 (716 feat, lr=0.01, aug C2×5/C4×5/C5×2/C3×1)\n'
    f'OOF F1 = {baseline_f1:.4f}')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(OUT_DIR / 'run32_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run32_confusion_matrix.png")


# ──────────────────────────────────────────────────────────────────────────────
# SOFT CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────────────
soft_cm = np.zeros((6, 6))
for c in range(6):
    soft_cm[c] = loo_probas[y_tr == c].mean(axis=0)
print("\nSoft confusion (mean predicted probability per true class):")
print(f"  {'':14}" + "".join(f"  {n:>10}" for n in CLASS_NAMES))
for i in range(6):
    print(f"  {CLASS_NAMES[i]:<14}" + "".join(f"  {soft_cm[i,j]:>10.4f}" for j in range(6)))
print(f"\n  C2→C1 confusion: {soft_cm[2,1]:.4f}  (run31 ref: 0.477)")
print(f"  C2 self-proba:   {soft_cm[2,2]:.4f}  (run31 ref: 0.216)")


# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION (Nelder-Mead, 8 restarts)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("THRESHOLD OPTIMIZATION")
print("=" * 60)

def neg_macro_f1(log_scales, proba, y_true):
    scales = np.exp(log_scales)
    scaled = proba * scales
    scaled /= scaled.sum(axis=1, keepdims=True)
    return -f1_score(y_true, scaled.argmax(axis=1), average='macro')

best_result, best_f1_opt = None, -np.inf
for x0_seed in range(8):
    rng = np.random.RandomState(x0_seed * 17)
    x0  = rng.uniform(-0.5, 0.5, 6)
    res = minimize(neg_macro_f1, x0=x0, args=(loo_probas, y_tr),
                   method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-8, 'fatol': 1e-8})
    if -res.fun > best_f1_opt:
        best_f1_opt, best_result = -res.fun, res
    print(f"  Restart {x0_seed}: F1 = {-res.fun:.4f}")

optimal_scales = np.exp(best_result.x)
opt_probas     = loo_probas * optimal_scales
opt_probas    /= opt_probas.sum(axis=1, keepdims=True)
opt_f1         = f1_score(y_tr, opt_probas.argmax(1), average='macro')
opt_per_class  = f1_score(y_tr, opt_probas.argmax(1), average=None)
print(f"\nOptimal scales:   {np.round(optimal_scales, 3)}")
print(f"OOF F1 after opt: {opt_f1:.4f}  (was {baseline_f1:.4f}, +{opt_f1-baseline_f1:.4f})")
print("Per-class F1:", [f"{f:.3f}" for f in opt_per_class])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
colors = ['#d9534f' if i == 2 else '#5bc0de' for i in range(6)]
for ax, f1_vals, title in [
    (axes[0], per_class_f1,  f'Before opt  (OOF={baseline_f1:.4f})'),
    (axes[1], opt_per_class, f'After opt   (OOF={opt_f1:.4f})'),
]:
    bars = ax.bar(range(6), f1_vals, color=colors)
    ax.set_xticks(range(6)); ax.set_ylim(0, 1.05)
    ax.set_xticklabels([f'C{c}\n{CLASS_NAMES[c]}' for c in range(6)], fontsize=8)
    ax.set_ylabel('F1'); ax.set_title(title)
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
fig.suptitle('Per-class CV F1 — run32 (lr=0.01, aug, red=C2)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'run32_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run32_per_class_f1.png")


# ──────────────────────────────────────────────────────────────────────────────
# FINAL TRAINING — full original + all augmented data
# 5 LGB + 2 XGB + 1 RF = 8 models  (identical architecture to run31)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"FINAL TRAINING (5 LGB @ {avg_best_iter} iters + 2 XGB + 1 RF = 8 models)")
print(f"Training on {len(X_tr_feat) + len(X_aug_feat)} samples (orig + all aug)")
print("=" * 60)

# Combine original + all augmented for final training
X_final = np.concatenate([X_tr_feat, X_aug_feat])
y_final  = np.concatenate([y_tr,      y_aug])
sw_all   = compute_sample_weight('balanced', y_final)

final_probas = []
lgb_models   = []

for seed in range(5):
    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=6,
        n_estimators=avg_best_iter,
        learning_rate=0.01,
        num_leaves=31, max_depth=-1,
        min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(X_final, y_final, sample_weight=sw_all)
    final_probas.append(m.predict_proba(X_te_feat))
    lgb_models.append(m)
    print(f"  LGB seed={seed} done")
print(f"  LightGBM: {len(lgb_models)} models (lr=0.01, n_estimators={avg_best_iter})")

xgb_start = len(final_probas)
for seed in range(2):
    xm = xgb.XGBClassifier(
        objective='multi:softprob', num_class=6, tree_method='hist',
        max_depth=6, learning_rate=0.05, n_estimators=400,
        subsample=0.8, colsample_bytree=0.7,
        random_state=seed, n_jobs=-1, verbosity=0, eval_metric='mlogloss',
    )
    xm.fit(X_final, y_final, sample_weight=sw_all)
    final_probas.append(xm.predict_proba(X_te_feat))
    print(f"  XGB seed={seed} done")
print(f"  XGBoost: {len(final_probas) - xgb_start} models")

rf = RandomForestClassifier(
    n_estimators=300, max_features='sqrt', min_samples_leaf=5,
    class_weight='balanced_subsample', random_state=SEED, n_jobs=-1,
)
rf.fit(X_final, y_final)
final_probas.append(rf.predict_proba(X_te_feat))
print(f"  RandomForest done")
print(f"  Total: {len(final_probas)} models")

avg_proba   = np.mean(final_probas, axis=0)
scaled_test = avg_proba * optimal_scales
scaled_test /= scaled_test.sum(axis=1, keepdims=True)
preds = scaled_test.argmax(1)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_GROUPS = [
    (  0,  78, "mean_xyz_axis_stat+diff"),
    ( 78, 156, "std_xyz_axis_stat+diff"),
    (156, 208, "acc_mag+std_mag_stat+diff"),
    (208, 364, "cross_magnitudes_stat+diff"),
    (364, 416, "sum_features_stat+diff"),
    (416, 466, "fft_peak_5ch"),
    (466, 716, "window_10win_5ch"),
]

print("\n" + "=" * 60)
print("FEATURE GROUP IMPORTANCE (LightGBM gain, avg over 5 models)")
print("=" * 60)
imp_matrix = np.zeros((len(lgb_models), X_final.shape[1]))
for i, m in enumerate(lgb_models):
    imp_matrix[i] = m.booster_.feature_importance(importance_type='gain')
avg_imp = imp_matrix.mean(axis=0)
total   = avg_imp.sum()

run31_pcts = {
    "mean_xyz_axis_stat+diff": 10.4, "std_xyz_axis_stat+diff": 28.8,
    "acc_mag+std_mag_stat+diff": 11.2, "cross_magnitudes_stat+diff": 27.5,
    "sum_features_stat+diff": 5.7, "fft_peak_5ch": 2.7, "window_10win_5ch": 13.7,
}
print(f"\n  {'Group':<34} {'N':>5}   {'% total':>8}   run31%  delta")
print(f"  {'-'*65}")
group_rows = []
for start, end, name in FEATURE_GROUPS:
    g   = avg_imp[start:end].sum(); pct = g / total * 100
    group_rows.append((name, end-start, g, pct))
    r31 = run31_pcts.get(name, 0)
    delta = f"({pct-r31:+.1f}pp)" if abs(pct-r31) > 0.3 else ""
    print(f"  {name:<34} {end-start:>5}   {pct:>7.1f}%   {r31:.1f}% {delta}")


# ──────────────────────────────────────────────────────────────────────────────
# SAVE SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run32.csv"
sub.to_csv(out_path, index=False)

print(f"\n✅ Submission saved: {out_path}")
print("\nPrediction distribution (predicted vs expected from train rate):")
for c in range(6):
    cnt = (preds == c).sum()
    exp = int(len(preds) * counts[c] / len(y_tr))
    print(f"  Class {c} {CLASS_NAMES[c]:<12}: {cnt:5d}  (expected ~{exp}, delta {cnt-exp:+d})")

print(f"\n{'─'*60}")
print(f"run32 SUMMARY")
print(f"{'─'*60}")
print(f"  Augmentation:            C2×5, C4×5, C5×2, C3×1  noise_std={NOISE_STD}")
print(f"  Training samples:        {len(X_tr)} orig + {len(X_aug_raw)} aug = {len(X_final)}")
print(f"  Best iters (CV folds):   {best_iters}  → final={avg_best_iter}")
print(f"  OOF F1 (baseline):       {baseline_f1:.4f}")
print(f"  OOF F1 (threshold-opt):  {opt_f1:.4f}")
print(f"  Optimal scales:          {np.round(optimal_scales, 3)}")
print(f"\n  Run comparison (OOF → Kaggle gap consistently ~+0.060):")
print(f"  run31 (716 feat, no aug):  OOF=~0.730  Kaggle=0.7906")
print(f"  run32 (716 feat, + aug):   OOF={baseline_f1:.4f}  Kaggle=?.????")
print(f"  Benchmark: run23 aug effect on 382-feat: +0.016 (0.7633→0.7792)")
print(f"{'─'*60}")