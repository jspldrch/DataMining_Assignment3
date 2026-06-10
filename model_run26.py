# run26: 412 features (382 base + 30 energy per segment), 5 LGB + 2 XGB, lr=0.03, no augmentation

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

X_tr = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_tr = tr["y"].astype(np.int32)
users    = tr["users"]
X_te     = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
te_ids   = te["file_ids"]
te_users = te["users"]

unique, counts = np.unique(y_tr, return_counts=True)
print(f"Train: {X_tr.shape}  Test: {X_te.shape}  (no normalization, no augmentation)")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_tr)*100:.1f}%)")
print(f"  class_weight='balanced' handles imbalance")


# feature helpers
def stats9(s):
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s, 1),
            np.percentile(s, 75, 1)-np.percentile(s, 25, 1),
            np.array([skew(r)     for r in s]),
            np.array([kurtosis(r) for r in s])]

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
    # mean + std per segment
    N, T = s.shape; sl = T // n_seg; out = []
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1)]
    return out

def seg_with_energy(s, n_seg):
    # mean + std + energy per segment (energy = mean(x²))
    N, T = s.shape; sl = T // n_seg; out = []
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1), (w**2).mean(1)]
    return out

def xcorr(a, b):
    ca = a - a.mean(1, keepdims=True)
    cb = b - b.mean(1, keepdims=True)
    return (ca*cb).mean(1) / (ca.std(1)*cb.std(1)+1e-10)

def trend3(s):
    first = s[:, :60].mean(1)
    last  = s[:, -60:].mean(1)
    return [first, last, last - first]


# feature extraction — 412 features (382 base + 30 energy for sx/sy/sz segments)
def extract(X):
    N, T, _ = X.shape
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]

    jx = np.diff(mx, axis=1)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)

    mag_std  = np.sqrt(sx**2 + sy**2 + sz**2)
    mag_jerk = np.sqrt(jx**2 + jy**2 + jz**2)

    parts = []
    for ch in [sx, sy, sz]:       parts += stats9(ch)          # std_channels 27
    parts += stats9(mag_std)                                    # mag_std 9
    for ch in [jx, jy, jz]:       parts += stats9(ch)          # jerk_channels 27
    parts += stats9(mag_jerk)                                   # mag_jerk 9
    for sig in [mag_std, mag_jerk]:                             # seg_mag 120
        for ns in [10, 20]: parts += seg(sig, ns)
    for ch in [sx, sy, sz]:       parts += seg_with_energy(ch, 10)  # seg_std_ch 90 (NEW)
    for ch in [jx, jy, jz]:       parts += seg(ch, 10)         # seg_jerk_ch 60
    for lag in [1, 2, 5, 10, 20, 30, 60]:                       # ac_jerk 7
        parts.append(ac(mag_jerk, lag))
    for ch in [sx, sy, sz]:                                     # ac_std_ch 12
        for lag in [1, 5, 10, 30]: parts.append(ac(ch, lag))
    for sig in [mag_jerk, mag_std, sx, sy, sz]:                 # spectral 25
        parts.append(spectral5(sig))
    for a, b in [(jx,jy),(jx,jz),(jy,jz)]:                     # crosscorr 6
        parts.append(xcorr(a, b))
    for a, b in [(sx,sy),(sx,sz),(sy,sz)]:
        parts.append(xcorr(a, b))
    cj = mag_jerk - mag_jerk.mean(1, keepdims=True)             # zerocross 1
    parts.append((np.diff(np.sign(cj), axis=1) != 0).sum(1) / T)
    pr = np.zeros(N, dtype=np.float32)                          # peaks 1
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n], height=mag_jerk[n].mean())[0]) / T
    parts.append(pr)
    for ch in [mx, my, mz, sx, sy, sz]: parts += trend3(ch)    # trend 18

    return np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1) for p in parts
    ]).astype(np.float32)


print("\nExtracting features...")
X_tr_feat = extract(X_tr)
X_te_feat = extract(X_te)
assert X_tr_feat.shape[1] == 412, f"Expected 412, got {X_tr_feat.shape[1]}"
print(f"  Features: {X_tr_feat.shape[1]}  (382 run23 base + 30 segment energy)")

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr_feat)
X_te_sc  = scaler.transform(X_te_feat)


# cv — 5 folds
print("\ncv (5 folds x 1 LGB)...")

unique_users = np.unique(users)
user_folds   = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids     = np.array([user_folds[u] for u in users])

loo_probas = np.zeros((len(y_tr), 6), dtype=np.float64)

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")

    sc_fold   = StandardScaler()
    X_fold_sc = sc_fold.fit_transform(X_tr_feat[tr_idx])
    X_va_sc   = sc_fold.transform(X_tr_feat[va_idx])

    m = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=63,
        learning_rate=0.03, colsample_bytree=0.8, subsample=0.8,
        class_weight="balanced", min_child_samples=20,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    m.fit(X_fold_sc, y_tr[tr_idx])
    loo_probas[va_idx] = m.predict_proba(X_va_sc)
    fold_preds = loo_probas[va_idx].argmax(1)
    print(f"  Macro F1 = {f1_score(y_tr[va_idx], fold_preds, average='macro'):.4f}"
          f"  Acc = {accuracy_score(y_tr[va_idx], fold_preds):.4f}")

loo_preds    = loo_probas.argmax(1)
baseline_f1  = f1_score(y_tr, loo_preds, average='macro')
per_class_f1 = f1_score(y_tr, loo_preds, average=None)
print(f"\nBaseline CV macro F1: {baseline_f1:.4f}")
print("Per-class F1:", [f"{f:.3f}" for f in per_class_f1])

cm = confusion_matrix(y_tr, loo_preds, normalize='true')
print("\nCV Confusion Matrix (row=true, col=predicted):")
print(f"  {'':16}" + "".join(f"    C{c}" for c in range(6)))
for i in range(6):
    row = "".join(f"  {cm[i,j]:.2f}" for j in range(6))
    flag = "  ← C2 bottleneck" if i == 2 else ""
    print(f"  C{i} {CLASS_NAMES[i]:<14}{row}{flag}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title(f'CV Confusion Matrix — run26 (energy per segment)\nCV F1 = {baseline_f1:.4f}')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(OUT_DIR / 'run26_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: run26_confusion_matrix.png")


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
    (axes[0], per_class_f1,  f'Before threshold opt  (macro={baseline_f1:.4f})'),
    (axes[1], opt_per_class, f'After threshold opt   (macro={opt_f1:.4f})'),
]:
    bars = ax.bar(range(6), f1_vals, color=colors)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'C{c}\n{CLASS_NAMES[c]}' for c in range(6)], fontsize=8)
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
fig.suptitle('Per-class CV F1 — run26 (energy per segment)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'run26_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: run26_per_class_f1.png")


# final training — 5 LGB + 2 XGB
print("\nfinal training (5 LGB + 2 XGB)...")

sw = compute_sample_weight('balanced', y_tr)
final_probas = []
lgb_models   = []

for seed in range(5):
    m = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=63,
        learning_rate=0.03, colsample_bytree=0.8, subsample=0.8,
        class_weight="balanced", min_child_samples=20,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(X_tr_sc, y_tr)
    final_probas.append(m.predict_proba(X_te_sc))
    lgb_models.append(m)
print(f"  LightGBM: {len(lgb_models)} models")

xgb_start = len(final_probas)
for seed in range(2):
    xm = xgb.XGBClassifier(
        objective='multi:softprob', num_class=6, tree_method='hist',
        max_depth=6, learning_rate=0.05, n_estimators=400,
        subsample=0.8, colsample_bytree=0.7,
        random_state=seed, n_jobs=-1, verbosity=0, eval_metric='mlogloss',
    )
    xm.fit(X_tr_sc, y_tr, sample_weight=sw)
    final_probas.append(xm.predict_proba(X_te_sc))
print(f"  XGBoost:  {len(final_probas) - xgb_start} models")
print(f"  Total:    {len(final_probas)} models")

avg_proba   = np.mean(final_probas, axis=0)
scaled_test = avg_proba * optimal_scales
scaled_test /= scaled_test.sum(axis=1, keepdims=True)
preds = scaled_test.argmax(1)


# feature group importance
FEATURE_GROUPS = [
    (  0,  27, "std_channels"),
    ( 27,  36, "mag_std"),
    ( 36,  63, "jerk_channels"),
    ( 63,  72, "mag_jerk"),
    ( 72, 192, "seg_mag"),
    (192, 282, "seg_std_ch+energy"),   # 90: mean+std+energy for sx/sy/sz
    (282, 342, "seg_jerk_ch"),
    (342, 349, "ac_jerk"),
    (349, 361, "ac_std_ch"),
    (361, 386, "spectral"),
    (386, 392, "crosscorr"),
    (392, 393, "zerocross"),
    (393, 394, "peaks"),
    (394, 412, "trend_features"),
]

print("\nfeature group importance (LGB gain, avg 5 models):")
imp_matrix = np.zeros((len(lgb_models), X_tr_sc.shape[1]))
for i, m in enumerate(lgb_models):
    imp_matrix[i] = m.booster_.feature_importance(importance_type='gain')
avg_imp = imp_matrix.mean(axis=0)
total   = avg_imp.sum()

print(f"\n  {'Group':<26} {'N':>4}   {'% total':>8}")
print(f"  {'-'*42}")
group_rows = []
for start, end, name in FEATURE_GROUPS:
    g = avg_imp[start:end].sum()
    group_rows.append((name, end-start, g, g/total*100))
    flag = " [NEW]" if name == "seg_std_ch+energy" else ""
    print(f"  {name:<26} {end-start:>4}   {g/total*100:>7.1f}%{flag}")

fig, ax = plt.subplots(figsize=(10, 5))
gr_sorted = sorted(group_rows, key=lambda x: x[2], reverse=True)
colors_g = ['#d9534f' if r[0] == 'seg_std_ch+energy' else '#5bc0de'
            for r in gr_sorted]
ax.barh([r[0] for r in gr_sorted], [r[3] for r in gr_sorted], color=colors_g)
ax.set_xlabel('% of total LightGBM gain')
ax.set_title('Feature group importance — run26 (red = modified group)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / 'run26_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: run26_feature_importance.png")


# save submission
sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run26.csv"
sub.to_csv(out_path, index=False)

print(f"submission saved to {out_path}")
print(f"CV F1: {baseline_f1:.4f}  -> after threshold opt: {opt_f1:.4f}")
print(f"optimal scales: {np.round(optimal_scales, 3)}")
