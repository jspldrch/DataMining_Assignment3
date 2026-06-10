# run19: ablation study on mag_mean (373 features), 15 LGB + 9 XGB, augmentation, per-user norm

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

# feature groups (start, end, name)
FEATURE_GROUPS = [
    (  0,  27, "std_channels"),
    ( 27,  36, "mag_std"),
    ( 36,  45, "mag_mean"),        # test removing this
    ( 45,  72, "jerk_channels"),
    ( 72,  81, "mag_jerk"),
    ( 81, 201, "seg_mag"),
    (201, 261, "seg_std_ch"),
    (261, 321, "seg_jerk_ch"),
    (321, 328, "ac_jerk"),
    (328, 340, "ac_std_ch"),
    (340, 365, "spectral"),
    (365, 371, "crosscorr"),
    (371, 372, "zerocross"),
    (372, 373, "peaks"),
]

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


# per-user normalization
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


# data augmentation (same as run18)
AUG_CONFIG    = {2: 5, 4: 5, 5: 2, 3: 1}
AUG_NOISE_STD = 0.03

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
print(f"  {len(y_tr)} -> {len(y_tr_aug)} samples")


# feature helpers
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


# feature extraction — 373 features (same as run07/run18)
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

    for ch in [sx, sy, sz]:          parts += stats9(ch)   # [0:27]
    parts += stats9(mag_std)                                # [27:36]
    parts += stats9(mag_mean)                               # [36:45] mag_mean
    for ch in [jx, jy, jz]:          parts += stats9(ch)   # [45:72]
    parts += stats9(mag_jerk)                               # [72:81]

    for sig in [mag_std, mag_jerk]:
        for ns in [10, 20]:           parts += seg(sig, ns) # [81:201]
    for ch in [sx, sy, sz]:           parts += seg(ch, 10)  # [201:261]
    for ch in [jx, jy, jz]:           parts += seg(ch, 10)  # [261:321]

    for lag in [1, 2, 5, 10, 20, 30, 60]:
        parts.append(ac(mag_jerk, lag))                     # [321:328]
    for ch in [sx, sy, sz]:
        for lag in [1, 5, 10, 30]:    parts.append(ac(ch, lag)) # [328:340]

    for sig in [mag_jerk, mag_std, sx, sy, sz]:
        parts.append(spectral5(sig))                        # [340:365]

    for a, b in [(jx,jy),(jx,jz),(jy,jz)]:
        parts.append(xcorr(a, b))                           # [365:368]
    for a, b in [(sx,sy),(sx,sz),(sy,sz)]:
        parts.append(xcorr(a, b))                           # [368:371]

    cj = mag_jerk - mag_jerk.mean(1, keepdims=True)
    parts.append((np.diff(np.sign(cj), axis=1) != 0).sum(1) / T) # [371]

    pr = np.zeros(N, dtype=np.float32)
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n], height=mag_jerk[n].mean())[0]) / T
    parts.append(pr)                                        # [372]

    return np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1) for p in parts
    ]).astype(np.float32)


# user-contextual features (same as run18)
N_CTX = 45

def add_user_context(X_feat, user_ids, ref_feat=None, ref_user_ids=None):
    if ref_feat is None:
        ref_feat, ref_user_ids = X_feat, user_ids
    user_mean = {}; user_std = {}
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
X_tr_orig_feat = extract(X_tr)
X_tr_aug_feat  = extract(X_tr_aug)
X_te_feat      = extract(X_te)
print(f"  Features: {X_tr_orig_feat.shape[1]} base")


# ablation: test removing mag_mean (features 36:45), quick 2-fold CV
print("\nablation: mag_mean removal (2-fold check)")

unique_users = np.unique(users)
user_folds   = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids     = np.array([user_folds[u] for u in users])

MAG_MEAN_IDX = np.array(list(range(36, 45)))   # indices of mag_mean features
ALL_IDX      = np.arange(373)
NO_MAG_IDX   = np.array([i for i in ALL_IDX if i not in MAG_MEAN_IDX])

ablation_results = {}

for variant_name, feat_idx in [("full_373", ALL_IDX), ("no_mag_mean_364", NO_MAG_IDX)]:
    fold_f1s = []
    for fold in [0, 1]:   # quick 2-fold only
        tr_idx = np.where(fold_ids != fold)[0]
        va_idx = np.where(fold_ids == fold)[0]

        X_fold_aug, y_fold_aug, u_fold_aug = augment_minority(
            X_tr[tr_idx], y_tr[tr_idx], users[tr_idx],
            AUG_CONFIG, AUG_NOISE_STD, SEED + fold
        )
        X_fold_feat = extract(X_fold_aug)[:, feat_idx]
        X_fold_orig = X_tr_orig_feat[tr_idx][:, feat_idx]
        X_va_feat   = X_tr_orig_feat[va_idx][:, feat_idx]

        # User context on selected features only (first min(45, n_feat) features)
        n_ctx = min(45, len(feat_idx))
        def ctx_subset(Xf, uids, ref_f=None, ref_u=None):
            if ref_f is None: ref_f, ref_u = Xf, uids
            um = {}; us = {}
            for uid in np.unique(ref_u):
                idx2 = np.where(ref_u == uid)[0]
                um[uid] = ref_f[idx2, :n_ctx].mean(0)
                us[uid] = ref_f[idx2, :n_ctx].std(0) + 1e-8
            c = np.zeros((len(Xf), n_ctx), dtype=np.float32)
            for i, uid in enumerate(uids):
                if uid in um: c[i] = (Xf[i, :n_ctx] - um[uid]) / us[uid]
            return np.hstack([Xf, c]).astype(np.float32)

        X_fold_ctx = ctx_subset(X_fold_feat, u_fold_aug, X_fold_orig, users[tr_idx])
        X_va_ctx   = ctx_subset(X_va_feat,   users[va_idx])

        sc = StandardScaler()
        Xf_sc = sc.fit_transform(X_fold_ctx)
        Xv_sc = sc.transform(X_va_ctx)

        m = lgb.LGBMClassifier(
            n_estimators=500, class_weight="balanced",
            min_child_samples=20, reg_alpha=0.5, reg_lambda=1.0,
            num_leaves=63, learning_rate=0.03,
            colsample_bytree=0.8, subsample=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        m.fit(Xf_sc, y_fold_aug)
        preds = m.predict(Xv_sc)
        fold_f1s.append(f1_score(y_tr[va_idx], preds, average='macro'))

    ablation_results[variant_name] = np.mean(fold_f1s)
    print(f"  {variant_name:20s}: CV macro F1 = {np.mean(fold_f1s):.4f}"
          f"  (folds: {[f'{f:.4f}' for f in fold_f1s]})")

remove_mag_mean = ablation_results["no_mag_mean_364"] > ablation_results["full_373"]
print(f"\nremoving mag_mean {'HELPS' if remove_mag_mean else 'HURTS'}"
      f"  (delta = {ablation_results['no_mag_mean_364'] - ablation_results['full_373']:+.4f})")

FEAT_IDX = NO_MAG_IDX if remove_mag_mean else ALL_IDX
N_BASE   = len(FEAT_IDX)
print(f"using {N_BASE} base features for full model")


# build final feature matrices with chosen feature set
X_tr_aug_sel  = X_tr_aug_feat[:, FEAT_IDX]
X_tr_orig_sel = X_tr_orig_feat[:, FEAT_IDX]
X_te_sel      = X_te_feat[:, FEAT_IDX]

X_tr_aug_ctx = add_user_context(
    X_tr_aug_sel, users_aug,
    ref_feat=X_tr_orig_sel, ref_user_ids=users
)
X_te_ctx = add_user_context(X_te_sel, te_users)
print(f"\nFinal feature dimensions: train {X_tr_aug_ctx.shape}, test {X_te_ctx.shape}")

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_aug_ctx)
X_te_sc = scaler.transform(X_te_ctx)


# 5-fold LOO-CV with probability collection
print("\nLOO-CV — threshold optimization")

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

    X_fold_aug, y_fold_aug, u_fold_aug = augment_minority(
        X_tr[tr_idx], y_tr[tr_idx], users[tr_idx],
        AUG_CONFIG, AUG_NOISE_STD, SEED + fold
    )
    X_fold_feat = extract(X_fold_aug)[:, FEAT_IDX]
    X_fold_orig = X_tr_orig_feat[tr_idx][:, FEAT_IDX]
    X_fold_ctx  = add_user_context(
        X_fold_feat, u_fold_aug,
        ref_feat=X_fold_orig, ref_user_ids=users[tr_idx]
    )
    X_va_feat = X_tr_orig_feat[va_idx][:, FEAT_IDX]
    X_va_ctx  = add_user_context(X_va_feat, users[va_idx])

    sc_fold   = StandardScaler()
    X_fold_sc = sc_fold.fit_transform(X_fold_ctx)
    X_va_sc   = sc_fold.transform(X_va_ctx)

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

baseline_f1  = f1_score(y_tr, loo_probas.argmax(1), average='macro')
per_class_f1 = f1_score(y_tr, loo_probas.argmax(1), average=None)
print(f"\nBaseline CV macro F1: {baseline_f1:.4f}")
print("Per-class F1:", [f"{f:.3f}" for f in per_class_f1])


# threshold optimization
print("\nthreshold optimization")

def neg_macro_f1(log_scales, proba, y_true):
    scales = np.exp(log_scales)
    scaled = proba * scales
    scaled /= scaled.sum(axis=1, keepdims=True)
    return -f1_score(y_true, scaled.argmax(axis=1), average='macro')

best_result, best_f1_val = None, -np.inf
for x0_seed in range(8):
    rng = np.random.RandomState(x0_seed * 17)
    x0  = rng.uniform(-0.5, 0.5, 6)
    res = minimize(
        neg_macro_f1, x0=x0, args=(loo_probas, y_tr),
        method='Nelder-Mead',
        options={'maxiter': 100000, 'xatol': 1e-8, 'fatol': 1e-8},
    )
    if -res.fun > best_f1_val:
        best_f1_val, best_result = -res.fun, res
    print(f"  Restart {x0_seed}: F1 = {-res.fun:.4f}")

optimal_scales = np.exp(best_result.x)
opt_probas = loo_probas * optimal_scales
opt_probas /= opt_probas.sum(axis=1, keepdims=True)
opt_f1        = f1_score(y_tr, opt_probas.argmax(1), average='macro')
opt_per_class = f1_score(y_tr, opt_probas.argmax(1), average=None)
print(f"\nOptimal scales:  {np.round(optimal_scales, 3)}")
print(f"CV F1 after opt: {opt_f1:.4f}  (was {baseline_f1:.4f}, +{opt_f1-baseline_f1:.4f})")
print("Per-class F1:", [f"{f:.3f}" for f in opt_per_class])


# final training — 15 LGB + 9 XGB
print("\nfinal training (24 models)")

sw_aug = compute_sample_weight('balanced', y_tr_aug)

XGB_CONFIGS = [
    dict(max_depth=6, learning_rate=0.05, n_estimators=400,
         subsample=0.8, colsample_bytree=0.7),
    dict(max_depth=8, learning_rate=0.03, n_estimators=500,
         subsample=0.7, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.05, n_estimators=500,
         subsample=0.9, colsample_bytree=0.6),
]

final_probas  = []
lgb_importances = []

for seed in range(5):
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500, class_weight="balanced",
            min_child_samples=20, reg_alpha=0.5, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc, y_tr_aug)
        final_probas.append(m.predict_proba(X_te_sc))
        lgb_importances.append(m.feature_importances_)
print(f"  LightGBM: {len(final_probas)} models trained")

xgb_start = len(final_probas)
for seed in range(3):
    for cfg in XGB_CONFIGS:
        xm = xgb.XGBClassifier(
            objective='multi:softprob', num_class=6,
            tree_method='hist', random_state=seed,
            n_jobs=-1, verbosity=0, eval_metric='mlogloss', **cfg,
        )
        xm.fit(X_tr_sc, y_tr_aug, sample_weight=sw_aug)
        final_probas.append(xm.predict_proba(X_te_sc))
print(f"  XGBoost:  {len(final_probas) - xgb_start} models trained")

avg_proba   = np.mean(final_probas, axis=0)
scaled_test = avg_proba * optimal_scales
scaled_test /= scaled_test.sum(axis=1, keepdims=True)
preds = scaled_test.argmax(1)


# feature group importance (LGB gain, averaged over 15 models)
print("\nfeature group importance")

avg_imp = np.mean(lgb_importances, axis=0)
total_imp = avg_imp.sum()

selected_groups = []
for start, end, name in FEATURE_GROUPS:
    group_mask = (FEAT_IDX >= start) & (FEAT_IDX < end)
    n_in_group = group_mask.sum()
    if n_in_group == 0:
        selected_groups.append((name, 0, 0))
        continue
    selected_pos = np.where(group_mask)[0]
    group_imp = avg_imp[selected_pos].sum()
    selected_groups.append((name, n_in_group, group_imp))

ctx_imp = avg_imp[N_BASE:].sum()
selected_groups.append(("user_context", N_CTX, ctx_imp))

print(f"\n{'Group':<20} {'N feats':>8} {'Importance':>12} {'% of total':>12}")
print("-" * 56)
for name, n, imp in sorted(selected_groups, key=lambda x: -x[2]):
    pct = 100 * imp / total_imp if total_imp > 0 else 0
    removed = " (REMOVED)" if n == 0 else ""
    print(f"  {name:<18} {n:>8}   {imp:>12.1f}   {pct:>10.1f}%{removed}")

imp_df = pd.DataFrame(selected_groups, columns=["group", "n_features", "importance"])
imp_df["pct_total"] = 100 * imp_df["importance"] / total_imp
imp_df = imp_df.sort_values("importance", ascending=False)
imp_df.to_csv(OUT_DIR / "feature_group_importance_run19.csv", index=False)
print(f"importance table saved: {OUT_DIR / 'feature_group_importance_run19.csv'}")

n_base_feats = X_tr_aug_ctx.shape[1]
feat_names = []
for i, fi in enumerate(FEAT_IDX):
    group = next((n for s,e,n in FEATURE_GROUPS if s <= fi < e), "unknown")
    feat_names.append(f"{group}_{fi}")
for j in range(N_CTX):
    feat_names.append(f"user_ctx_{j}")
per_feat_df = pd.DataFrame({
    "feature": feat_names[:len(avg_imp)],
    "importance": avg_imp,
})
per_feat_df = per_feat_df.sort_values("importance", ascending=False)
per_feat_df.to_csv(OUT_DIR / "per_feature_importance_run19.csv", index=False)


# save submission
sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run19.csv"
sub.to_csv(out_path, index=False)

print(f"submission saved to {out_path}")
print(f"ablation: full_373={ablation_results['full_373']:.4f}  no_mag_mean_364={ablation_results['no_mag_mean_364']:.4f}  -> used {'no_mag_mean_364' if remove_mag_mean else 'full_373'}")
print(f"CV F1: {baseline_f1:.4f}  -> after threshold opt: {opt_f1:.4f}")
print(f"optimal scales: {np.round(optimal_scales, 3)}")
