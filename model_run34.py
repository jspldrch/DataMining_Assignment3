# run34: 722 features (716 + 6 gravity slope), 7 LGB + 4 XGB + 1 RF + 1 ET, lr=0.01, patience=300, no augmentation

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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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



# load data
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

print("loading data...")

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



# feature helpers
def _safe_skew_vec(ch):
    return np.array([skew(r)     if np.std(r) >= 1e-12 else 0.0 for r in ch],
                    dtype=np.float32)

def _safe_kurtosis_vec(ch):
    return np.array([kurtosis(r) if np.std(r) >= 1e-12 else 0.0 for r in ch],
                    dtype=np.float32)

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
        "dom_freq","dom_power","total_power","spectral_entropy",
        "low_power","high_power"]]
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
        names += [f"win{w}_{prefix}_{st}"
                  for st in ["mean","std","min","max","energy"]]
    return np.concatenate(parts, axis=1).astype(np.float32), names



# gravity slope features (6 new features for C2 walk_down)
def gravity_slope_features(X):
    """OLS slope + R² for mean_x/y/z over all 300 timesteps."""
    N, T = X.shape[0], X.shape[1]
    t     = np.arange(T, dtype=np.float64)
    t_c   = t - t.mean()                        # centered time index
    t_var = (t_c ** 2).sum()                    # sum of squared deviations

    feats, names = [], []
    for ch_idx, ax in enumerate(['x', 'y', 'z']):
        ch   = X[:, :, ch_idx].astype(np.float64)          # (N, T)
        ch_c = ch - ch.mean(axis=1, keepdims=True)          # mean-centered

        # OLS slope: β = Σ(t_c * ch_c) / Σ(t_c²)
        slope = (t_c * ch_c).sum(axis=1) / t_var            # (N,) signed

        # R²: proportion of variance explained by the linear trend
        fitted = ch.mean(axis=1, keepdims=True) + slope[:, None] * t_c
        ss_res = ((ch - fitted) ** 2).sum(axis=1)
        ss_tot = (ch_c ** 2).sum(axis=1)
        r2     = 1.0 - ss_res / (ss_tot + 1e-12)            # (N,)

        feats += [slope.astype(np.float32), r2.astype(np.float32)]
        names += [f"mean_{ax}_linslope", f"mean_{ax}_linr2"]

    return np.column_stack(feats), names  # (N, 6)



# feature extraction (716 base + 6 gravity slope = 722)
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
        ("std_xy_mag", std_xy_mag), ("std_xz_mag", std_xz_mag),
        ("std_yz_mag", std_yz_mag),
        ("mean_sum", mean_sum), ("std_sum", std_sum),
    ]
    fft_pk = [("acc_mag", acc_mag), ("std_mag", std_mag),
              ("std_x", sx), ("std_y", sy), ("std_z", sz)]
    wins   = [("acc_mag", acc_mag), ("std_mag", std_mag),
              ("std_x", sx), ("std_y", sy), ("std_z", sz)]

    feats_list, feat_names = [], []

    # 716 base features
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

    # +6 gravity slope features
    gf, gn = gravity_slope_features(X)
    feats_list.append(gf); feat_names += gn

    return np.concatenate(feats_list, axis=1), feat_names


print("\nExtracting features...")
X_tr_feat, feat_names = extract(X_tr)
X_te_feat, _          = extract(X_te)
assert X_tr_feat.shape[1] == 722, f"Expected 722, got {X_tr_feat.shape[1]}"
print(f"  Train: {X_tr_feat.shape}  Test: {X_te_feat.shape}")
print(f"  716 base features (run31 exact) + 6 gravity slope = 722 total")

# Quick sanity check: are the new features actually different across classes?
print("\n  Gravity slope feature means per class (sanity check):")
print(f"  {'Class':<14}  mean_z_linslope  mean_z_linr2")
slope_idx = feat_names.index("mean_z_linslope")
r2_idx    = feat_names.index("mean_z_linr2")
for c in range(6):
    mask = y_tr == c
    print(f"  C{c} {CLASS_NAMES[c]:<12}  "
          f"{X_tr_feat[mask, slope_idx].mean():>+.6f}  "
          f"{X_tr_feat[mask, r2_idx].mean():>8.4f}")
print("  Expected: C2 negative slope, C3 positive slope, C1 near-zero")

def clean(F_tr, F_te):
    F_tr = np.where(np.isfinite(F_tr), F_tr, np.nan)
    F_te = np.where(np.isfinite(F_te), F_te, np.nan)
    meds = np.nanmedian(F_tr, axis=0)
    nan_tr = np.isnan(F_tr); F_tr[nan_tr] = np.take(meds, np.where(nan_tr)[1])
    nan_te = np.isnan(F_te); F_te[nan_te] = np.take(meds, np.where(nan_te)[1])
    return F_tr.astype(np.float32), F_te.astype(np.float32)

X_tr_feat, X_te_feat = clean(X_tr_feat, X_te_feat)



# groupkfold cv (lr=0.01, patience=300, 722 features)
print("\ngroupkfold cv...")

gkf        = GroupKFold(n_splits=5)
loo_probas = np.zeros((len(y_tr), 6), dtype=np.float64)
best_iters = []
fold_f1s   = []

for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(X_tr_feat, y_tr, groups=users), start=1):
    X_f_tr, X_f_va = X_tr_feat[tr_idx], X_tr_feat[va_idx]
    y_f_tr, y_f_va = y_tr[tr_idx],      y_tr[va_idx]
    print(f"\nFold {fold}/5  train={len(tr_idx)}  val={len(va_idx)}")

    sw = compute_sample_weight('balanced', y_f_tr)

    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=6,
        n_estimators=5000,
        learning_rate=0.01,
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
          f"Macro F1={fold_f1:.4f}  "
          f"Acc={accuracy_score(y_f_va, va_proba.argmax(1)):.4f}")

avg_best_iter = int(np.mean(best_iters) * 1.10)
print(f"\nBest iters per fold: {best_iters}  ->  final n_estimators = {avg_best_iter}")

loo_preds    = loo_probas.argmax(1)
baseline_f1  = f1_score(y_tr, loo_preds, average='macro')
per_class_f1 = f1_score(y_tr, loo_preds, average=None)
print(f"\nOOF macro F1 : {baseline_f1:.4f}")
print(f"Mean fold F1 : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
print("Per-class F1 :", [f"{f:.3f}" for f in per_class_f1])
print(f"  C2 walk_down: {per_class_f1[2]:.3f}")
print("\n" + classification_report(y_tr, loo_preds, target_names=CLASS_NAMES, digits=4))



# confusion matrix
cm = confusion_matrix(y_tr, loo_preds, normalize='true')
print("CV Confusion Matrix (row=true, col=predicted):")
print(f"  {'':16}" + "".join(f"    C{c}" for c in range(6)))
for i in range(6):
    row  = "".join(f"  {cm[i,j]:.2f}" for j in range(6))
    flag = "  <- C2 bottleneck" if i == 2 else ""
    print(f"  C{i} {CLASS_NAMES[i]:<14}{row}{flag}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title(
    f'CV Confusion Matrix — run34\n'
    f'722 feat (716+6 gravity slope), 13-model ensemble\n'
    f'OOF F1 = {baseline_f1:.4f}')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(OUT_DIR / 'run34_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run34_confusion_matrix.png")



# soft confusion (mean predicted prob per true class)
soft_cm = np.zeros((6, 6))
for c in range(6):
    soft_cm[c] = loo_probas[y_tr == c].mean(axis=0)

print("\nSoft confusion (mean predicted probability per true class):")
print(f"  {'':14}" + "".join(f"  {n:>10}" for n in CLASS_NAMES))
for i in range(6):
    print(f"  {CLASS_NAMES[i]:<14}"
          + "".join(f"  {soft_cm[i,j]:>10.4f}" for j in range(6)))
print(f"\n  C2->C1 confusion: {soft_cm[2,1]:.4f}")
print(f"  C2 self-proba:    {soft_cm[2,2]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(soft_cm, annot=True, fmt='.3f', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title(
    f'Soft Confusion — run34 (722 feat)\n'
    f'C2->C1: {soft_cm[2,1]:.4f}  C2 self: {soft_cm[2,2]:.4f}')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('True class')
plt.tight_layout()
plt.savefig(OUT_DIR / 'run34_soft_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run34_soft_confusion.png")



# threshold optimization (nelder-mead, 8 restarts)
print("\nthreshold optimization...")

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
print(f"  C2 walk_down after opt: {opt_per_class[2]:.3f}")

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
fig.suptitle('Per-class CV F1 — run34 (722 feat, 13 models, red=C2)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'run34_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: run34_per_class_f1.png")



# feature groups (includes new +6 as separate group)
FEATURE_GROUPS = [
    (  0,  78, "mean_xyz_axis_stat+diff"),
    ( 78, 156, "std_xyz_axis_stat+diff"),
    (156, 208, "acc_mag+std_mag_stat+diff"),
    (208, 364, "cross_magnitudes_stat+diff"),
    (364, 416, "sum_features_stat+diff"),
    (416, 466, "fft_peak_5ch"),
    (466, 716, "window_10win_5ch"),
    (716, 722, "gravity_slope_NEW"),        # 6 new features
]

run31_pcts = {
    "mean_xyz_axis_stat+diff": 10.4, "std_xyz_axis_stat+diff": 28.8,
    "acc_mag+std_mag_stat+diff": 11.2, "cross_magnitudes_stat+diff": 27.5,
    "sum_features_stat+diff": 5.7, "fft_peak_5ch": 2.7,
    "window_10win_5ch": 13.7, "gravity_slope_NEW": 0.0,
}



# final training (7 LGB + 4 XGB + 1 RF + 1 ET = 13 models)
print(f"\nfinal training (13 models, n_estimators={avg_best_iter})...")

sw_all       = compute_sample_weight('balanced', y_tr)
final_probas = []
lgb_models   = []

for seed in range(7):
    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=6,
        n_estimators=avg_best_iter,
        learning_rate=0.01,
        num_leaves=31, max_depth=-1,
        min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(X_tr_feat, y_tr, sample_weight=sw_all)
    final_probas.append(m.predict_proba(X_te_feat))
    lgb_models.append(m)
    print(f"  LGB seed={seed} done")
print(f"  → LightGBM: {len(lgb_models)} models")

# 4 XGB (2 configs x 2 seeds)
xgb_configs = [
    dict(max_depth=6, learning_rate=0.05, n_estimators=400,
         subsample=0.8, colsample_bytree=0.7),
    dict(max_depth=5, learning_rate=0.04, n_estimators=500,
         subsample=0.75, colsample_bytree=0.75),
]
xgb_start = len(final_probas)
for cfg_idx, cfg in enumerate(xgb_configs):
    for seed in range(2):
        xm = xgb.XGBClassifier(
            objective='multi:softprob', num_class=6, tree_method='hist',
            random_state=seed + cfg_idx * 10, n_jobs=-1,
            verbosity=0, eval_metric='mlogloss',
            **cfg,
        )
        xm.fit(X_tr_feat, y_tr, sample_weight=sw_all)
        final_probas.append(xm.predict_proba(X_te_feat))
        print(f"  XGB cfg={cfg_idx} seed={seed} done  "
              f"(depth={cfg['max_depth']}, lr={cfg['learning_rate']}, "
              f"n_est={cfg['n_estimators']})")
print(f"  → XGBoost: {len(final_probas) - xgb_start} models (2 configs × 2 seeds)")

rf = RandomForestClassifier(
    n_estimators=300, max_features='sqrt', min_samples_leaf=5,
    class_weight='balanced_subsample', random_state=SEED, n_jobs=-1,
)
rf.fit(X_tr_feat, y_tr)
final_probas.append(rf.predict_proba(X_te_feat))
print(f"  RandomForest done")

et = ExtraTreesClassifier(
    n_estimators=300, max_features='sqrt', min_samples_leaf=5,
    class_weight='balanced_subsample', random_state=SEED + 1, n_jobs=-1,
)
et.fit(X_tr_feat, y_tr)
final_probas.append(et.predict_proba(X_te_feat))
print(f"  ExtraTrees done")
print(f"  → Total: {len(final_probas)} models")

avg_proba   = np.mean(final_probas, axis=0)
scaled_test = avg_proba * optimal_scales
scaled_test /= scaled_test.sum(axis=1, keepdims=True)
preds = scaled_test.argmax(1)



# feature group importance
print("\nfeature group importance...")
imp_matrix = np.zeros((len(lgb_models), X_tr_feat.shape[1]))
for i, m in enumerate(lgb_models):
    imp_matrix[i] = m.booster_.feature_importance(importance_type='gain')
avg_imp = imp_matrix.mean(axis=0)
total   = avg_imp.sum()

print(f"\n  {'Group':<34} {'N':>5}   {'% total':>8}   run31%  delta")
print(f"  {'-'*65}")
group_rows = []
for start, end, name in FEATURE_GROUPS:
    g   = avg_imp[start:end].sum(); pct = g / total * 100
    group_rows.append((name, end-start, g, pct))
    r31 = run31_pcts.get(name, 0)
    delta = f"({pct-r31:+.1f}pp)" if r31 > 0 else "(new)"
    print(f"  {name:<34} {end-start:>5}   {pct:>7.1f}%   {r31:.1f}% {delta}")

# Individual gravity feature importances
print(f"\n  Gravity slope feature importances (run34 new features):")
for i, name in enumerate(["mean_x_linslope","mean_x_linr2",
                           "mean_y_linslope","mean_y_linr2",
                           "mean_z_linslope","mean_z_linr2"]):
    feat_idx = 716 + i
    pct = avg_imp[feat_idx] / total * 100
    # rank among all 722 features
    rank = int((avg_imp > avg_imp[feat_idx]).sum()) + 1
    print(f"    {name:<22}  gain={avg_imp[feat_idx]:8.1f}  ({pct:.2f}%)  "
          f"rank {rank}/722")

print(f"\n  Top-15 individual features (LGB gain):")
top15 = np.argsort(avg_imp)[::-1][:15]
for rank, idx in enumerate(top15, 1):
    new_tag = " [NEW]" if idx >= 716 else ""
    print(f"  {rank:2d}. {feat_names[idx]:<50}  "
          f"gain={avg_imp[idx]:.1f}{new_tag}")



# save submission
sub      = pd.DataFrame({"Id": te_ids, "Label": preds})
sub      = sub.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run34.csv"
sub.to_csv(out_path, index=False)

print(f"submission saved to {out_path}")
print("\nPrediction distribution:")
for c in range(6):
    cnt = (preds == c).sum()
    exp = int(len(preds) * counts[c] / len(y_tr))
    print(f"  Class {c} {CLASS_NAMES[c]:<12}: {cnt:5d}  (expected ~{exp}, delta {cnt-exp:+d})")

print(f"\nOOF F1 (raw): {baseline_f1:.4f}  OOF F1 (opt): {opt_f1:.4f}")
print(f"best_iters: {best_iters}  avg={int(np.mean(best_iters))}")
print(f"C2 walk_down F1: {per_class_f1[2]:.3f} raw / {opt_per_class[2]:.3f} opt")
print(f"C2->C1 confusion: {soft_cm[2,1]:.4f}  C2 self-proba: {soft_cm[2,2]:.4f}")
