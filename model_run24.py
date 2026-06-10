# run24: 410 features (run23 - 9 mag_mean + 28 extra stats), 15 LGB + 9 XGB, augmentation

import numpy as np
import pandas as pd
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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

CLASS_NAMES = ["sit/stand", "walk_flat", "walk_down", "walk_up", "running", "other"]


# load data
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

print("loading data...")

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
te_users = te["users"]

unique, counts = np.unique(y_tr, return_counts=True)
print(f"Train: {X_tr.shape}  Test: {X_te.shape}  (no per-user normalization)")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_tr)*100:.1f}%)")


# data augmentation (same as run18/run23)
AUG_CONFIG    = {2: 5, 4: 5, 5: 2, 3: 1}
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


# feature helpers
def stats9(s):
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s, 1),
            np.percentile(s, 75, 1)-np.percentile(s, 25, 1),
            np.array([skew(r)     for r in s]),
            np.array([kurtosis(r) for r in s])]

def stats_extra7(s):
    # energy, MAD, RMS, q05, q10, q90, q95 for dominant channels
    energy = (s**2).mean(1)
    return [energy,
            np.abs(s - s.mean(1, keepdims=True)).mean(1),   # MAD
            np.sqrt(energy),                                  # RMS
            np.percentile(s,  5, 1),
            np.percentile(s, 10, 1),
            np.percentile(s, 90, 1),
            np.percentile(s, 95, 1)]

def spectral5(s):
    N, T = s.shape
    out  = np.zeros((N, 5), dtype=np.float32)
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

def trend3(s):
    first = s[:, :60].mean(1)
    last  = s[:, -60:].mean(1)
    return [first, last, last - first]


# feature extraction — 410 features
def extract(X):
    N, T, _ = X.shape
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]

    jx = np.diff(mx, axis=1)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)

    # mag_mean omitted (gravity-dominated without normalization)
    mag_std  = np.sqrt(sx**2 + sy**2 + sz**2)
    mag_jerk = np.sqrt(jx**2 + jy**2 + jz**2)

    parts = []

    for ch in [sx, sy, sz]:       parts += stats9(ch)
    parts += stats9(mag_std)
    for ch in [jx, jy, jz]:       parts += stats9(ch)
    parts += stats9(mag_jerk)
    for sig in [mag_std, mag_jerk]:
        for ns in [10, 20]:
            parts += seg(sig, ns)
    for ch in [sx, sy, sz]:       parts += seg(ch, 10)
    for ch in [jx, jy, jz]:       parts += seg(ch, 10)
    for lag in [1, 2, 5, 10, 20, 30, 60]:
        parts.append(ac(mag_jerk, lag))
    for ch in [sx, sy, sz]:
        for lag in [1, 5, 10, 30]:
            parts.append(ac(ch, lag))
    for sig in [mag_jerk, mag_std, sx, sy, sz]:
        parts.append(spectral5(sig))
    for a, b in [(jx,jy),(jx,jz),(jy,jz)]:
        parts.append(xcorr(a, b))
    for a, b in [(sx,sy),(sx,sz),(sy,sz)]:
        parts.append(xcorr(a, b))
    cj = mag_jerk - mag_jerk.mean(1, keepdims=True)
    parts.append((np.diff(np.sign(cj), axis=1) != 0).sum(1) / T)
    pr = np.zeros(N, dtype=np.float32)
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n], height=mag_jerk[n].mean())[0]) / T
    parts.append(pr)
    for ch in [mx, my, mz, sx, sy, sz]:
        parts += trend3(ch)
    # extra stats for dominant channels (+28)
    for sig in [sx, sy, sz, mag_std]:
        parts += stats_extra7(sig)

    return np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1) for p in parts
    ]).astype(np.float32)


print("\nExtracting features...")
X_tr_orig_feat = extract(X_tr)
X_tr_aug_feat  = extract(X_tr_aug)
X_te_feat      = extract(X_te)
assert X_tr_orig_feat.shape[1] == 410, f"Expected 410 features, got {X_tr_orig_feat.shape[1]}"
print(f"  Features: {X_tr_orig_feat.shape[1]}")

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr_aug_feat)
X_te_sc  = scaler.transform(X_te_feat)


# feature names
_S9    = ['mean','std','min','max','range','median','iqr','skew','kurt']
_EXTRA = ['energy','mad','rms','q05','q10','q90','q95']
FEATURE_NAMES = []
for ch in ['sx','sy','sz']:
    for s in _S9: FEATURE_NAMES.append(f"{ch}_{s}")
for s in _S9: FEATURE_NAMES.append(f"mag_std_{s}")
for ch in ['jx','jy','jz']:
    for s in _S9: FEATURE_NAMES.append(f"{ch}_{s}")
for s in _S9: FEATURE_NAMES.append(f"mag_jerk_{s}")
for sig in ['mag_std','mag_jerk']:
    for ns in [10,20]:
        for i in range(ns):
            FEATURE_NAMES += [f"{sig}_seg{ns}_{i:02d}_mean",
                               f"{sig}_seg{ns}_{i:02d}_std"]
for ch in ['sx','sy','sz']:
    for i in range(10):
        FEATURE_NAMES += [f"{ch}_seg10_{i:02d}_mean", f"{ch}_seg10_{i:02d}_std"]
for ch in ['jx','jy','jz']:
    for i in range(10):
        FEATURE_NAMES += [f"{ch}_seg10_{i:02d}_mean", f"{ch}_seg10_{i:02d}_std"]
for lag in [1,2,5,10,20,30,60]: FEATURE_NAMES.append(f"ac_mag_jerk_lag{lag}")
for ch in ['sx','sy','sz']:
    for lag in [1,5,10,30]: FEATURE_NAMES.append(f"ac_{ch}_lag{lag}")
for sig in ['mag_jerk','mag_std','sx','sy','sz']:
    for f in ['dom_freq','entropy','low_pow','mid_pow','high_pow']:
        FEATURE_NAMES.append(f"spec_{sig}_{f}")
for a,b in [('jx','jy'),('jx','jz'),('jy','jz'),
            ('sx','sy'),('sx','sz'),('sy','sz')]:
    FEATURE_NAMES.append(f"xcorr_{a}_{b}")
FEATURE_NAMES.append("zerocross_mag_jerk")
FEATURE_NAMES.append("peak_rate_mag_jerk")
for ch in ['mx','my','mz','sx','sy','sz']:
    FEATURE_NAMES += [f"{ch}_first60", f"{ch}_last60", f"{ch}_trend"]
for ch in ['sx','sy','sz','mag_std']:
    for s in _EXTRA: FEATURE_NAMES.append(f"{ch}_{s}")
assert len(FEATURE_NAMES) == 410, f"Name mismatch: {len(FEATURE_NAMES)}"

FEATURE_GROUPS = [
    (  0,  27, "std_channels"),
    ( 27,  36, "mag_std"),
    ( 36,  63, "jerk_channels"),
    ( 63,  72, "mag_jerk"),
    ( 72, 192, "seg_mag"),
    (192, 252, "seg_std_ch"),
    (252, 312, "seg_jerk_ch"),
    (312, 319, "ac_jerk"),
    (319, 331, "ac_std_ch"),
    (331, 356, "spectral"),
    (356, 362, "crosscorr"),
    (362, 363, "zerocross"),
    (363, 364, "peaks"),
    (364, 382, "trend_features"),
    (382, 410, "extra_stats_gyro"),
]


# cv
print("\ncv (5 folds x 9 LGB)...")

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

    X_fold_aug, y_fold_aug, _ = augment_minority(
        X_tr[tr_idx], y_tr[tr_idx], users[tr_idx],
        AUG_CONFIG, AUG_NOISE_STD, SEED + fold
    )
    X_fold_feat = extract(X_fold_aug)
    X_va_feat   = X_tr_orig_feat[va_idx]

    sc_fold   = StandardScaler()
    X_fold_sc = sc_fold.fit_transform(X_fold_feat)
    X_va_sc   = sc_fold.transform(X_va_feat)

    fold_probas = []
    for seed in [42, 7, 13]:
        for cfg in CONFIGS:
            m = lgb.LGBMClassifier(
                n_estimators=500, class_weight="balanced",
                min_child_samples=20, reg_alpha=0.5, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, verbose=-1, **cfg,
            )
            m.fit(X_fold_sc, y_fold_aug)
            fold_probas.append(m.predict_proba(X_va_sc))

    loo_probas[va_idx] = np.mean(fold_probas, axis=0)
    fold_preds = loo_probas[va_idx].argmax(1)
    print(f"  Macro F1 = {f1_score(y_tr[va_idx], fold_preds, average='macro'):.4f}"
          f"  Acc = {accuracy_score(y_tr[va_idx], fold_preds):.4f}")

loo_preds    = loo_probas.argmax(1)
baseline_f1  = f1_score(y_tr, loo_preds, average='macro')
per_class_f1 = f1_score(y_tr, loo_preds, average=None)
print(f"\nBaseline CV macro F1: {baseline_f1:.4f}")
print("Per-class F1:", [f"{f:.3f}" for f in per_class_f1])

cm = confusion_matrix(y_tr, loo_preds, normalize='true')
print("\nCV Confusion Matrix (row=true, col=predicted, normalized):")
print(f"  {'':16}" + "".join(f"    C{c}" for c in range(6)))
for i in range(6):
    row = "".join(f"  {cm[i,j]:.2f}" for j in range(6))
    flag = "  <- C2 bottleneck" if i == 2 else ""
    print(f"  C{i} {CLASS_NAMES[i]:<14}{row}{flag}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title(f'CV Confusion Matrix — run24\nBaseline macro F1 = {baseline_f1:.4f}')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(OUT_DIR / 'run24_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: run24_confusion_matrix.png")


# threshold optimization
print("\nthreshold optimization...")

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
opt_probas     = loo_probas * optimal_scales
opt_probas    /= opt_probas.sum(axis=1, keepdims=True)
opt_f1         = f1_score(y_tr, opt_probas.argmax(1), average='macro')
opt_per_class  = f1_score(y_tr, opt_probas.argmax(1), average=None)

print(f"\nOptimal scales:  {np.round(optimal_scales, 3)}")
print(f"CV F1 after opt: {opt_f1:.4f}  (was {baseline_f1:.4f}, +{opt_f1-baseline_f1:.4f})")
print("Per-class F1:", [f"{f:.3f}" for f in opt_per_class])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
colors = ['#d9534f' if i == 2 else '#5bc0de' for i in range(6)]
x = np.arange(6)
for ax, f1_vals, title in [
    (axes[0], per_class_f1,  f'Before threshold opt  (macro={baseline_f1:.4f})'),
    (axes[1], opt_per_class, f'After threshold opt   (macro={opt_f1:.4f})'),
]:
    bars = ax.bar(x, f1_vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}\n{CLASS_NAMES[c]}' for c in range(6)], fontsize=8)
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
fig.suptitle('Per-class CV F1 — run24  (red = Class 2 bottleneck)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'run24_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: run24_per_class_f1.png")


# final training — 15 LGB + 9 XGB
print("\nfinal training (15 LGB + 9 XGB)...")

sw_aug = compute_sample_weight('balanced', y_tr_aug)
XGB_CONFIGS = [
    dict(max_depth=6, learning_rate=0.05, n_estimators=400,
         subsample=0.8, colsample_bytree=0.7),
    dict(max_depth=8, learning_rate=0.03, n_estimators=500,
         subsample=0.7, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.05, n_estimators=500,
         subsample=0.9, colsample_bytree=0.6),
]

final_probas = []
lgb_models   = []

for seed in range(5):
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500, class_weight="balanced",
            min_child_samples=20, reg_alpha=0.5, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc, y_tr_aug)
        final_probas.append(m.predict_proba(X_te_sc))
        lgb_models.append(m)
print(f"  LightGBM: {len(lgb_models)} models trained")

xgb_start = len(final_probas)
for seed in range(3):
    for cfg in XGB_CONFIGS:
        xm = xgb.XGBClassifier(
            objective='multi:softprob', num_class=6, tree_method='hist',
            random_state=seed, n_jobs=-1, verbosity=0,
            eval_metric='mlogloss', **cfg,
        )
        xm.fit(X_tr_sc, y_tr_aug, sample_weight=sw_aug)
        final_probas.append(xm.predict_proba(X_te_sc))
print(f"  XGBoost:  {len(final_probas) - xgb_start} models trained")
print(f"  Total:    {len(final_probas)} models")

avg_proba   = np.mean(final_probas, axis=0)
scaled_test = avg_proba * optimal_scales
scaled_test /= scaled_test.sum(axis=1, keepdims=True)
preds = scaled_test.argmax(1)


# feature group importance
print("\nfeature group importance (LGB gain, avg 15 models):")

n_features = X_tr_sc.shape[1]
imp_matrix  = np.zeros((len(lgb_models), n_features))
for i, m in enumerate(lgb_models):
    imp_matrix[i] = m.booster_.feature_importance(importance_type='gain')
avg_imp = imp_matrix.mean(axis=0)
total   = avg_imp.sum()

print(f"\n  {'Group':<22} {'N':>4}   {'Importance':>12}   {'% total':>8}")
print(f"  {'-'*52}")
group_rows = []
for start, end, name in FEATURE_GROUPS:
    g = avg_imp[start:end].sum()
    group_rows.append((name, end-start, g, g/total*100))
    print(f"  {name:<22} {end-start:>4}   {g:>12.1f}   {g/total*100:>7.1f}%")

group_rows_sorted = sorted(group_rows, key=lambda x: x[2], reverse=True)
fig, ax = plt.subplots(figsize=(10, 6))
names_g = [r[0] for r in group_rows_sorted]
pcts_g  = [r[3] for r in group_rows_sorted]
colors_g = ['#d9534f' if n in ('extra_stats_gyro', 'trend_features')
            else '#5bc0de' for n in names_g]
bars = ax.barh(names_g, pcts_g, color=colors_g)
for bar, v in zip(bars, pcts_g):
    ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=8)
ax.set_xlabel('% of total LightGBM gain importance')
ax.set_title('Feature group importance — run24\n(red = new groups added in run24)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / 'run24_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: run24_feature_importance.png")

print(f"\n  Top 20 individual features:")
print(f"  {'Rank':>4}  {'Feature':<32}  {'Importance':>10}  {'%':>6}")
print(f"  {'-'*58}")
feat_series = pd.Series(avg_imp, index=FEATURE_NAMES).sort_values(ascending=False)
for rank, (fname, fimp) in enumerate(feat_series.head(20).items(), 1):
    print(f"  {rank:>4}  {fname:<32}  {fimp:>10.1f}  {fimp/total*100:>5.2f}%")

print(f"\n  extra_stats_gyro breakdown:")
extra_names = [n for n in FEATURE_NAMES if any(
    n.endswith(s) for s in _EXTRA) and any(
    n.startswith(ch) for ch in ['sx_','sy_','sz_','mag_std_'])]
extra_series = feat_series[extra_names].sort_values(ascending=False)
for fname, fimp in extra_series.head(14).items():
    print(f"    {fname:<28}  {fimp:>10.1f}  {fimp/total*100:>5.2f}%")

df_full = feat_series.reset_index()
df_full.columns = ["feature", "importance"]
df_full["pct"] = df_full["importance"] / total * 100
df_full.to_csv(OUT_DIR / "per_feature_importance_run24.csv", index=False)
print(f"\n  Full importance saved: per_feature_importance_run24.csv")


# save submission
sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run24.csv"
sub.to_csv(out_path, index=False)

print(f"submission saved to {out_path}")
print(f"CV F1: {baseline_f1:.4f}  -> after threshold opt: {opt_f1:.4f}")
print(f"optimal scales: {np.round(optimal_scales, 3)}")
