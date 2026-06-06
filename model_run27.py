"""
model_run27.py — Notebook-exact 716 features, no augmentation

Why run25/26 underperform the notebook (~0.77 vs ~0.79 Kaggle):
  1. Missing axis-specific mean_x/y/z features. Gravity direction is consistent
     per activity: walk_up → positive z-axis trend, walk_down → negative z-axis trend,
     running → high-frequency x/y oscillation. These axis-specific raw means encode
     universal physical patterns that transfer directly to unseen test users.
  2. num_leaves=63 (run25/26) vs 31 (notebook): larger trees overfit to training users.
  3. Fixed 500 trees vs early stopping: either under- or over-trains.

Feature set — 716 total (matches notebook full_lightgbm exactly):
  16 base channels × (18 stat + 8 diff)     = 416
    base = mean_x/y/z, std_x/y/z, acc_mag, std_mag,
           xy/xz/yz_mag, std_xy/xz/yz_mag, mean_sum, std_sum
  5 channels × (6 FFT + 4 peak)             =  50
    channels = acc_mag, std_mag, std_x, std_y, std_z
  5 channels × 10 windows × 5 stats         = 250
  Total                                      = 716

Training (notebook-exact):
  GroupKFold(5) on users, LGB n_estimators=2000 + early_stopping(100)
  num_leaves=31, no StandardScaler, class_weight via sample_weight
  Final ensemble: 5 LGB (avg best_iter × 1.1) + 2 XGB

Outputs: confusion matrix, covariance/correlation matrices of OOF probabilities,
         soft confusion matrix, per-class F1, feature group importance.

Paths: auto-detects Kaggle vs laptop (same logic as run26).
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


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING — same NPZ loading as run26
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

X_tr = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_tr = tr["y"].astype(np.int32)
users    = tr["users"]
X_te     = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
te_ids   = te["file_ids"]

unique, counts = np.unique(y_tr, return_counts=True)
print(f"Train: {X_tr.shape}  Test: {X_te.shape}  channels=[mean_x,mean_y,mean_z,std_x,std_y,std_z]")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_tr)*100:.1f}%)")
n_users = len(np.unique(users))
print(f"  Users: {n_users}  → GroupKFold(5): ~{n_users//5} val-users / ~{n_users-n_users//5} train-users per fold")
print(f"  Train/val split: ~{(n_users-n_users//5)/n_users*100:.0f}% / ~{n_users//5/n_users*100:.0f}% by user")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE HELPERS — vectorised, matching notebook exactly
# ──────────────────────────────────────────────────────────────────────────────

def _safe_skew_vec(ch):
    return np.array([skew(row) if np.std(row) >= 1e-12 else 0.0 for row in ch],
                    dtype=np.float32)

def _safe_kurtosis_vec(ch):
    return np.array([kurtosis(row) if np.std(row) >= 1e-12 else 0.0 for row in ch],
                    dtype=np.float32)


def stat_features(ch, prefix):
    """18 statistics per channel. ch: (N,T) → (N,18)"""
    q = np.percentile(ch, [5, 10, 25, 75, 90, 95], axis=1).T  # (N,6)
    m      = ch.mean(axis=1)
    energy = (ch ** 2).mean(axis=1)
    feats  = np.column_stack([
        m,
        ch.std(axis=1),
        ch.min(axis=1),
        ch.max(axis=1),
        ch.max(axis=1) - ch.min(axis=1),
        np.median(ch, axis=1),
        q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4], q[:, 5],
        q[:, 3] - q[:, 2],                              # iqr
        energy,
        np.sqrt(np.maximum(energy, 0)),                 # rms
        np.abs(ch - m[:, None]).mean(axis=1),           # mad
        _safe_skew_vec(ch),
        _safe_kurtosis_vec(ch),
    ])
    names = [f"{prefix}_{s}" for s in [
        "mean", "std", "min", "max", "range", "median",
        "q05", "q10", "q25", "q75", "q90", "q95", "iqr",
        "energy", "rms", "mad", "skew", "kurtosis",
    ]]
    return feats.astype(np.float32), names


def diff_features(ch, prefix):
    """8 temporal diff + trend features. ch: (N,T) → (N,8)"""
    d    = np.diff(ch, axis=1)                          # (N,T-1)
    absd = np.abs(d)
    feats = np.column_stack([
        d.mean(axis=1),
        d.std(axis=1),
        absd.mean(axis=1),
        absd.max(axis=1),
        (d ** 2).mean(axis=1),
        ch[:, :60].mean(axis=1),
        ch[:, -60:].mean(axis=1),
        ch[:, -60:].mean(axis=1) - ch[:, :60].mean(axis=1),
    ])
    names = [f"{prefix}_{s}" for s in [
        "diff_mean", "diff_std", "diff_abs_mean", "diff_abs_max", "diff_energy",
        "first_60_mean", "last_60_mean", "last_minus_first_60",
    ]]
    return feats.astype(np.float32), names


def fft_features(ch, prefix):
    """6 FFT features (rfft, no DC, low/high split at 0.10 Hz). ch: (N,T) → (N,6)"""
    N, T  = ch.shape
    freqs = np.fft.rfftfreq(T, d=1.0)                  # 0 … 0.5 Hz
    out   = np.zeros((N, 6), dtype=np.float32)
    for n in range(N):
        v = ch[n] - ch[n].mean()
        if np.std(v) < 1e-12:
            continue
        pwr   = np.abs(np.fft.rfft(v)) ** 2
        p_ndc = pwr[1:];  f_ndc = freqs[1:]            # drop DC bin
        tot   = p_ndc.sum()
        if tot <= 0:
            continue
        dom  = np.argmax(p_ndc)
        prob = p_ndc / tot
        out[n] = [
            f_ndc[dom],
            p_ndc[dom],
            tot,
            -np.sum(prob * np.log(prob + 1e-12)),
            p_ndc[f_ndc <= 0.10].sum(),
            p_ndc[f_ndc  > 0.10].sum(),
        ]
    names = [f"{prefix}_fft_{s}" for s in [
        "dom_freq", "dom_power", "total_power", "spectral_entropy",
        "low_power", "high_power",
    ]]
    return out, names


def peak_features(ch, prefix):
    """4 peak features (threshold = mean + 0.5·std). ch: (N,T) → (N,4)"""
    N, T = ch.shape
    out  = np.zeros((N, 4), dtype=np.float32)
    for n in range(N):
        v = ch[n]
        if np.std(v) < 1e-12:
            continue
        thr  = v.mean() + 0.5 * v.std()
        pks, props = find_peaks(v, height=thr)
        out[n, 0] = len(pks)
        out[n, 1] = len(pks) / T
        if len(pks) > 0:
            out[n, 2] = props["peak_heights"].mean()
            out[n, 3] = props["peak_heights"].max()
    names = [f"{prefix}_{s}" for s in [
        "num_peaks", "peak_rate", "mean_peak_height", "max_peak_height",
    ]]
    return out, names


def window_features(ch, prefix, n_windows=10):
    """10 windows × 5 stats (mean/std/min/max/energy) = 50 features. ch: (N,T) → (N,50)"""
    N, T = ch.shape
    ws   = T // n_windows
    parts, names = [], []
    for w in range(n_windows):
        s   = w * ws
        e   = (w + 1) * ws if w < n_windows - 1 else T
        win = ch[:, s:e]
        parts.append(np.column_stack([
            win.mean(axis=1),
            win.std(axis=1),
            win.min(axis=1),
            win.max(axis=1),
            (win ** 2).mean(axis=1),
        ]))
        names.extend([f"win{w}_{prefix}_{st}" for st in ["mean", "std", "min", "max", "energy"]])
    return np.concatenate(parts, axis=1).astype(np.float32), names


# ──────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTION — 716 features, notebook full_lightgbm exact match
#
# Feature layout (for FEATURE_GROUPS below):
#   mean_x/y/z stat+diff :  3 × 26 =  78   (idx   0– 77)  ← axis-specific, KEY
#   std_x/y/z  stat+diff :  3 × 26 =  78   (idx  78–155)
#   acc_mag+std_mag s+d  :  2 × 26 =  52   (idx 156–207)
#   cross-magnitudes s+d :  6 × 26 = 156   (idx 208–363)  xy/xz/yz variants
#   mean_sum+std_sum s+d :  2 × 26 =  52   (idx 364–415)
#   FFT + peak (5 ch)    :  5 × 10 =  50   (idx 416–465)
#   windows (5 ch)       :  5 × 50 = 250   (idx 466–715)
#   Total                             716
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

    # 16 base channels: stat(18) + diff(8) = 26 each → 416
    base = [
        ("mean_x", mx), ("mean_y", my), ("mean_z", mz),
        ("std_x",  sx), ("std_y",  sy), ("std_z",  sz),
        ("acc_mag", acc_mag), ("std_mag", std_mag),
        ("xy_mag",  xy_mag),  ("xz_mag",  xz_mag), ("yz_mag", yz_mag),
        ("std_xy_mag", std_xy_mag), ("std_xz_mag", std_xz_mag), ("std_yz_mag", std_yz_mag),
        ("mean_sum", mean_sum), ("std_sum", std_sum),
    ]

    # 5 channels: FFT(6) + peak(4) = 10 each → 50
    fft_pk = [
        ("acc_mag", acc_mag), ("std_mag", std_mag),
        ("std_x",   sx),      ("std_y",   sy), ("std_z", sz),
    ]

    # 5 channels: 10 windows × 5 stats = 50 each → 250
    wins = [
        ("acc_mag", acc_mag), ("std_mag", std_mag),
        ("std_x",   sx),      ("std_y",   sy), ("std_z", sz),
    ]

    feats_list, feat_names = [], []

    for name, ch in base:
        sf, sn = stat_features(ch, name)
        df, dn = diff_features(ch, name)
        feats_list += [sf, df]
        feat_names += sn + dn

    for name, ch in fft_pk:
        ff, fn = fft_features(ch, name)
        pf, pn = peak_features(ch, name)
        feats_list += [ff, pf]
        feat_names += fn + pn

    for name, ch in wins:
        wf, wn = window_features(ch, name)
        feats_list.append(wf)
        feat_names += wn

    return np.concatenate(feats_list, axis=1), feat_names


print("\nExtracting features...")
X_tr_feat, feat_names = extract(X_tr)
X_te_feat, _          = extract(X_te)
assert X_tr_feat.shape[1] == 716, f"Expected 716 features, got {X_tr_feat.shape[1]}"
print(f"  Features: {X_tr_feat.shape[1]}  (notebook full_lightgbm exact match)")
print(f"  Layout: 416 stat+diff  +  50 FFT+peak  +  250 windows")

# Clean inf/NaN: fill with training column median (matches notebook)
def clean(F_tr, F_te):
    F_tr = np.where(np.isfinite(F_tr), F_tr, np.nan)
    F_te = np.where(np.isfinite(F_te), F_te, np.nan)
    meds = np.nanmedian(F_tr, axis=0)
    nan_tr = np.isnan(F_tr); nan_te = np.isnan(F_te)
    F_tr[nan_tr] = np.take(meds, np.where(nan_tr)[1])
    F_te[nan_te] = np.take(meds, np.where(nan_te)[1])
    return F_tr.astype(np.float32), F_te.astype(np.float32)

X_tr_feat, X_te_feat = clean(X_tr_feat, X_te_feat)


# ──────────────────────────────────────────────────────────────────────────────
# GROUPKFOLD CV — notebook-exact training setup
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GROUPKFOLD CV (5 splits, LGB early stopping=100, notebook-exact)")
print("=" * 60)

gkf        = GroupKFold(n_splits=5)
loo_probas = np.zeros((len(y_tr), 6), dtype=np.float64)
fold_te_pr = np.zeros((len(X_te_feat), 6), dtype=np.float64)
best_iters = []
fold_f1s   = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_feat, y_tr, groups=users), start=1):
    X_f_tr, X_f_va = X_tr_feat[tr_idx], X_tr_feat[va_idx]
    y_f_tr, y_f_va = y_tr[tr_idx],      y_tr[va_idx]
    print(f"\nFold {fold}/5  train={len(tr_idx)}  val={len(va_idx)}")

    sw = compute_sample_weight('balanced', y_f_tr)

    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=6,
        n_estimators=2000, learning_rate=0.03,
        num_leaves=31, max_depth=-1, min_child_samples=20,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED + fold, n_jobs=-1, verbose=-1,
    )
    m.fit(
        X_f_tr, y_f_tr,
        sample_weight=sw,
        eval_set=[(X_f_va, y_f_va)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200),
        ],
    )

    va_proba = m.predict_proba(X_f_va)
    loo_probas[va_idx] = va_proba
    fold_te_pr += m.predict_proba(X_te_feat) / 5
    best_iters.append(m.best_iteration_)

    fold_f1 = f1_score(y_f_va, va_proba.argmax(1), average='macro')
    fold_f1s.append(fold_f1)
    print(f"  best_iter={m.best_iteration_:4d}  "
          f"Macro F1={fold_f1:.4f}  Acc={accuracy_score(y_f_va, va_proba.argmax(1)):.4f}")

avg_best_iter = int(np.mean(best_iters) * 1.10)   # +10% for full-data training
print(f"\nBest iters per fold: {best_iters}  →  final n_estimators = {avg_best_iter}")

loo_preds    = loo_probas.argmax(1)
baseline_f1  = f1_score(y_tr, loo_preds, average='macro')
per_class_f1 = f1_score(y_tr, loo_preds, average=None)
print(f"\nOOF macro F1: {baseline_f1:.4f}  (notebook full_lightgbm: 0.7299)")
print(f"Mean fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
print("Per-class F1:", [f"{f:.3f}" for f in per_class_f1])
print("\n" + classification_report(y_tr, loo_preds, target_names=CLASS_NAMES, digits=4))


# ──────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_tr, loo_preds, normalize='true')
print("CV Confusion Matrix (row=true, col=predicted):")
print(f"  {'':16}" + "".join(f"    C{c}" for c in range(6)))
for i in range(6):
    row = "".join(f"  {cm[i,j]:.2f}" for j in range(6))
    print(f"  C{i} {CLASS_NAMES[i]:<14}{row}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title(f'CV Confusion Matrix — run27 (716 notebook features)\nOOF F1 = {baseline_f1:.4f}')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(OUT_DIR / 'run27_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run27_confusion_matrix.png")


# ──────────────────────────────────────────────────────────────────────────────
# COVARIANCE & CORRELATION MATRICES OF OOF PROBABILITIES
# Shows which classes are inherently confused by the model's confidence scores.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OOF PROBABILITY COVARIANCE & CORRELATION MATRICES")
print("=" * 60)

cov_mat  = np.cov(loo_probas.T)                         # (6,6)
std_diag = np.sqrt(np.diag(cov_mat))
corr_mat = cov_mat / (std_diag[:, None] * std_diag[None, :] + 1e-12)

print("\nCovariance matrix (6×6) of OOF predicted probabilities:")
print(f"  {'':14}" + "".join(f"  {n:>10}" for n in CLASS_NAMES))
for i in range(6):
    row = "".join(f"  {cov_mat[i,j]:>10.5f}" for j in range(6))
    print(f"  {CLASS_NAMES[i]:<14}{row}")

print("\nCorrelation matrix (6×6) of OOF predicted probabilities:")
print(f"  {'':14}" + "".join(f"  {n:>10}" for n in CLASS_NAMES))
for i in range(6):
    row = "".join(f"  {corr_mat[i,j]:>10.4f}" for j in range(6))
    print(f"  {CLASS_NAMES[i]:<14}{row}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, mat, title, fmt in [
    (axes[0], cov_mat,  'OOF Probability Covariance Matrix', '.5f'),
    (axes[1], corr_mat, 'OOF Probability Correlation Matrix', '.3f'),
]:
    sns.heatmap(mat, annot=True, fmt=fmt, cmap='RdBu_r', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                center=0, linewidths=0.5)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.suptitle(f'run27 — OOF Probability Matrices  (OOF F1={baseline_f1:.4f})', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'run27_covariance_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run27_covariance_matrices.png")

# Soft confusion: mean predicted probability per true class
soft_cm = np.zeros((6, 6))
for c in range(6):
    idx = np.where(y_tr == c)[0]
    soft_cm[c] = loo_probas[idx].mean(axis=0)

print("\nSoft confusion — mean predicted probability per true class:")
print(f"  {'':14}" + "".join(f"  {n:>10}" for n in CLASS_NAMES))
for i in range(6):
    row = "".join(f"  {soft_cm[i,j]:>10.4f}" for j in range(6))
    print(f"  {CLASS_NAMES[i]:<14}{row}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(soft_cm, annot=True, fmt='.3f', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title('Soft Confusion Matrix (mean predicted probability per true class)\nrun27 — 716 notebook features')
ax.set_xlabel('Predicted class probability'); ax.set_ylabel('True class')
plt.tight_layout()
plt.savefig(OUT_DIR / 'run27_soft_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run27_soft_confusion.png")


# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("THRESHOLD OPTIMIZATION")
print("=" * 60)

def neg_macro_f1(log_scales, proba, y_true):
    scales = np.exp(log_scales)
    scaled = proba * scales
    scaled /= scaled.sum(axis=1, keepdims=True)
    return -f1_score(y_true, scaled.argmax(axis=1), average='macro')

best_result, best_f1 = None, -np.inf
for x0_seed in range(8):
    rng = np.random.RandomState(x0_seed * 17)
    x0  = rng.uniform(-0.5, 0.5, 6)
    res = minimize(neg_macro_f1, x0=x0, args=(loo_probas, y_tr),
                   method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-8, 'fatol': 1e-8})
    if -res.fun > best_f1:
        best_f1, best_result = -res.fun, res
    print(f"  Restart {x0_seed}: F1 = {-res.fun:.4f}")

optimal_scales = np.exp(best_result.x)
opt_probas     = loo_probas * optimal_scales
opt_probas    /= opt_probas.sum(axis=1, keepdims=True)
opt_f1         = f1_score(y_tr, opt_probas.argmax(1), average='macro')
opt_per_class  = f1_score(y_tr, opt_probas.argmax(1), average=None)
print(f"\nOptimal scales:  {np.round(optimal_scales, 3)}")
print(f"CV F1 after opt: {opt_f1:.4f}  (was {baseline_f1:.4f}, +{opt_f1-baseline_f1:.4f})")
print("Per-class F1:", [f"{f:.3f}" for f in opt_per_class])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
colors = ['#d9534f' if i == 2 else '#5bc0de' for i in range(6)]
for ax, f1_vals, title in [
    (axes[0], per_class_f1,  f'Before threshold opt  (OOF={baseline_f1:.4f})'),
    (axes[1], opt_per_class, f'After threshold opt   (OOF={opt_f1:.4f})'),
]:
    bars = ax.bar(range(6), f1_vals, color=colors)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'C{c}\n{CLASS_NAMES[c]}' for c in range(6)], fontsize=8)
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
fig.suptitle('Per-class CV F1 — run27 (716 notebook features, no augmentation)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'run27_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run27_per_class_f1.png")


# ──────────────────────────────────────────────────────────────────────────────
# FINAL TRAINING — 5 LGB (all data, avg best_iter×1.1) + 2 XGB = 7 models
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"FINAL TRAINING (5 LGB @ iter={avg_best_iter} + 2 XGBoost = 7 models)")
print("=" * 60)

sw_all       = compute_sample_weight('balanced', y_tr)
final_probas = []
lgb_models   = []

for seed in range(5):
    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=6,
        n_estimators=avg_best_iter, learning_rate=0.03,
        num_leaves=31, max_depth=-1, min_child_samples=20,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(X_tr_feat, y_tr, sample_weight=sw_all)
    final_probas.append(m.predict_proba(X_te_feat))
    lgb_models.append(m)
print(f"  LightGBM: {len(lgb_models)} models  (n_estimators={avg_best_iter})")

xgb_start = len(final_probas)
for seed in range(2):
    xm = xgb.XGBClassifier(
        objective='multi:softprob', num_class=6, tree_method='hist',
        max_depth=6, learning_rate=0.05, n_estimators=400,
        subsample=0.8, colsample_bytree=0.7,
        random_state=seed, n_jobs=-1, verbosity=0, eval_metric='mlogloss',
    )
    xm.fit(X_tr_feat, y_tr, sample_weight=sw_all)
    final_probas.append(xm.predict_proba(X_te_feat))
print(f"  XGBoost:  {len(final_probas) - xgb_start} models")
print(f"  Total:    {len(final_probas)} models")

avg_proba   = np.mean(final_probas, axis=0)
scaled_test = avg_proba * optimal_scales
scaled_test /= scaled_test.sum(axis=1, keepdims=True)
preds = scaled_test.argmax(1)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_GROUPS = [
    (  0,  78, "mean_xyz_axis_stat+diff"),   # ← KEY: raw mean_x/y/z, gravity not removed
    ( 78, 156, "std_xyz_axis_stat+diff"),
    (156, 208, "acc_mag+std_mag_stat+diff"),
    (208, 364, "cross_magnitudes_stat+diff"),  # xy/xz/yz_mag variants
    (364, 416, "sum_features_stat+diff"),
    (416, 466, "fft_peak_5ch"),
    (466, 716, "window_10win_5ch"),
]

print("\n" + "=" * 60)
print("FEATURE GROUP IMPORTANCE (LightGBM gain, avg over 5 models)")
print("=" * 60)
imp_matrix = np.zeros((len(lgb_models), X_tr_feat.shape[1]))
for i, m in enumerate(lgb_models):
    imp_matrix[i] = m.booster_.feature_importance(importance_type='gain')
avg_imp = imp_matrix.mean(axis=0)
total   = avg_imp.sum()

print(f"\n  {'Group':<32} {'N':>5}   {'% total':>8}")
print(f"  {'-'*50}")
group_rows = []
for start, end, name in FEATURE_GROUPS:
    g = avg_imp[start:end].sum()
    pct = g / total * 100
    group_rows.append((name, end - start, g, pct))
    flag = "  ← axis-specific mean features" if "mean_xyz" in name else ""
    print(f"  {name:<32} {end-start:>5}   {pct:>7.1f}%{flag}")

fig, ax = plt.subplots(figsize=(11, 5))
gr_s = sorted(group_rows, key=lambda x: x[2], reverse=True)
ax.barh(
    [r[0] for r in gr_s],
    [r[3] for r in gr_s],
    color=['#d9534f' if 'mean_xyz' in r[0] else '#5bc0de' for r in gr_s],
)
ax.set_xlabel('% of total LightGBM gain')
ax.set_title('Feature group importance — run27\n(red = axis-specific mean features, new vs run25/26)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / 'run27_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run27_feature_importance.png")

# Top-20 individual features
top20_idx  = np.argsort(avg_imp)[::-1][:20]
print(f"\n  Top-20 individual features:")
for rank, idx in enumerate(top20_idx, 1):
    print(f"  {rank:2d}. {feat_names[idx]:<45}  gain={avg_imp[idx]:.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# SAVE SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run27.csv"
sub.to_csv(out_path, index=False)

print(f"\n✅ Submission saved: {out_path}")
print("\nPrediction distribution (predicted vs expected from train rate):")
for c in range(6):
    cnt = (preds == c).sum()
    exp = int(len(preds) * counts[c] / len(y_tr))
    print(f"  Class {c}: {cnt:5d}  (expected ~{exp})")

print(f"\nOptimal scales:          {np.round(optimal_scales, 3)}")
print(f"OOF F1 (baseline):       {baseline_f1:.4f}")
print(f"OOF F1 (threshold-opt):  {opt_f1:.4f}")
print(f"Notebook full_lightgbm:  0.7299 OOF  (Kaggle ~0.79)")
print(f"run26 (412 features):    0.6981 CV   (Kaggle  0.7703)")
print(f"run25 (382 features):    0.6976 CV   (Kaggle  0.7633)")
