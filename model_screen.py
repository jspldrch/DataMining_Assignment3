"""
model_screen.py — Fast feature screening. NOT for final submission.

Goal: test whether a new feature group helps BEFORE committing to a 2-3hr run.
Target runtime: 15-25 minutes on Kaggle.

Key difference from full runs:
  - NO per-user normalization (raw axis values keep gravity direction info)
  - No augmentation (class_weight='balanced' compensates)
  - 3-fold CV instead of 5-fold
  - 1 LGB per fold with early stopping
  - No XGBoost, no threshold optimization, no submission file

Why no normalization:
  The notebook achieved 0.7923 (vs run18's 0.7738) without normalization.
  Per-user normalization removes the gravity direction from each axis,
  which is exactly the signal that trend/diff features need to distinguish
  walking upstairs vs downstairs.
  Note: jerk = np.diff(mx) naturally removes gravity (high-pass filter),
  so jerk features are unaffected by this choice.

How to use:
  1. Run with all toggles False first → establishes your no-normalization baseline.
  2. Flip ONE toggle at a time and compare macro F1 + Class 2 F1.
  3. Only do a full run if improvement > 0.005 OOF macro F1.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE TOGGLES — flip ONE at a time to test its contribution
#
#  Start: all False  → baseline (run18 features, no normalization)
#  Then enable one at a time and compare OOF macro F1 and Class 2 F1
# ══════════════════════════════════════════════════════════════════════════════

USE_USER_CONTEXT   = True    # +45  z-score of features vs user's own mean
USE_JERK_ASYMMETRY = False   # +6   pos/neg jerk mean per axis (run22 — didn't help)
USE_TREND_FEATURES = False   # +18  first_60_mean, last_60_mean, last-first per channel
USE_DIFF_FEATURES  = False   # +27  diff_mean, diff_std, diff_abs_mean per signal
USE_EXTRA_STATS    = False   # +35  energy, MAD, RMS, q05, q10, q90, q95 per signal
USE_BETTER_PEAKS   = False   # +6   num_peaks, mean_peak_height, max_peak_height

# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (raw, no normalization)
# ──────────────────────────────────────────────────────────────────────────────
def find_npz(name):
    search_paths = [
        Path("/outoutkaggle/input") / name,
        Path("/kaggle/input/train-data") / name,
        Path("/kaggle/input/test-data") / name,
        Path("/kaggle/input/har-data") / name,
    ]
    for p in search_paths:
        if p.exists(): return str(p)
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits: return hits[0]
    raise FileNotFoundError(f"Cannot find {name}")

print("Loading data...")
try:
    tr = np.load(find_npz("train_data.npz"), allow_pickle=True)
except:
    tr = np.load("train_data.npz", allow_pickle=True)

X_tr   = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)   # raw, not normalized
y_tr   = tr["y"].astype(np.int32)
users  = tr["users"]
unique_users = np.unique(users)
print(f"  Train: {X_tr.shape}   Users: {len(unique_users)}")
print(f"  NOTE: no per-user normalization applied to raw signal")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def stats9(s):
    """9 base stats — same as all full runs."""
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1) - s.min(1), np.median(s, 1),
            np.percentile(s, 75, 1) - np.percentile(s, 25, 1),
            np.array([skew(r)     for r in s]),
            np.array([kurtosis(r) for r in s])]

def stats_extra7(s):
    """energy, MAD, RMS, q05, q10, q90, q95 — from notebook approach."""
    energy = (s ** 2).mean(1)
    mad    = np.abs(s - s.mean(1, keepdims=True)).mean(1)
    rms    = np.sqrt(energy)
    return [energy, mad, rms,
            np.percentile(s,  5, 1),
            np.percentile(s, 10, 1),
            np.percentile(s, 90, 1),
            np.percentile(s, 95, 1)]

def trend3(s):
    """first_60_mean, last_60_mean, last_minus_first — temporal trend."""
    first = s[:, :60].mean(1)
    last  = s[:, -60:].mean(1)
    return [first, last, last - first]

def diff_stats3(s):
    """diff_mean, diff_std, diff_abs_mean — rate of change."""
    d = np.diff(s, axis=1)
    return [d.mean(1), d.std(1), np.abs(d).mean(1)]

def peak_stats3(s):
    """num_peaks, mean_peak_height, max_peak_height — better than peak_rate."""
    N = s.shape[0]
    n_p = np.zeros(N, dtype=np.float32)
    m_h = np.zeros(N, dtype=np.float32)
    x_h = np.zeros(N, dtype=np.float32)
    for i in range(N):
        sig  = s[i]
        thr  = sig.mean() + 0.5 * sig.std()
        pks, props = find_peaks(sig, height=thr)
        n_p[i] = len(pks)
        if len(pks) > 0:
            h = props['peak_heights']
            m_h[i] = h.mean()
            x_h[i] = h.max()
    return [n_p, m_h, x_h]

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
    num = ((s1 - s1.mean(1, keepdims=True)) *
           (s2 - s2.mean(1, keepdims=True))).mean(1)
    return num / (s1.std(1) * s2.std(1) + 1e-10)

def seg(s, n_seg):
    N, T = s.shape; sl = T // n_seg; out = []
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1)]
    return out

def xcorr(a, b):
    ca = a - a.mean(1, keepdims=True)
    cb = b - b.mean(1, keepdims=True)
    return (ca * cb).mean(1) / (ca.std(1) * cb.std(1) + 1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def extract(X,
            use_jerk_asymmetry=False,
            use_trend_features=False,
            use_diff_features=False,
            use_extra_stats=False,
            use_better_peaks=False):
    N, T, _ = X.shape
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]

    jx = np.diff(mx, axis=1)   # jerk naturally removes gravity (high-pass)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)

    mag_mean = np.sqrt(mx**2 + my**2 + mz**2)
    mag_std  = np.sqrt(sx**2 + sy**2 + sz**2)
    mag_jerk = np.sqrt(jx**2 + jy**2 + jz**2)

    parts = []
    feature_log = {}

    # ── Baseline features identical to run18 (373) ───────────────────────────
    n0 = 0
    for ch in [sx, sy, sz]:       parts += stats9(ch)    # std_channels  27
    parts += stats9(mag_std)                              # mag_std        9
    parts += stats9(mag_mean)                             # mag_mean       9
    for ch in [jx, jy, jz]:       parts += stats9(ch)    # jerk_channels 27
    parts += stats9(mag_jerk)                             # mag_jerk       9

    for sig in [mag_std, mag_jerk]:                       # seg_mag       120
        for ns in [10, 20]:
            parts += seg(sig, ns)
    for ch in [sx, sy, sz]:       parts += seg(ch, 10)   # seg_std_ch     60
    for ch in [jx, jy, jz]:       parts += seg(ch, 10)   # seg_jerk_ch    60

    for lag in [1, 2, 5, 10, 20, 30, 60]:                # ac_jerk         7
        parts.append(ac(mag_jerk, lag))
    for ch in [sx, sy, sz]:                               # ac_std_ch      12
        for lag in [1, 5, 10, 30]:
            parts.append(ac(ch, lag))

    for sig in [mag_jerk, mag_std, sx, sy, sz]:           # spectral       25
        parts.append(spectral5(sig))

    for a, b in [(jx,jy),(jx,jz),(jy,jz)]:               # crosscorr       6
        parts.append(xcorr(a, b))
    for a, b in [(sx,sy),(sx,sz),(sy,sz)]:
        parts.append(xcorr(a, b))

    cj = mag_jerk - mag_jerk.mean(1, keepdims=True)       # zerocross       1
    parts.append((np.diff(np.sign(cj), axis=1) != 0).sum(1) / T)

    pr = np.zeros(N, dtype=np.float32)                    # peak_rate       1
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n],
                               height=mag_jerk[n].mean())[0]) / T
    parts.append(pr)
    feature_log['baseline_373'] = 373
    # ── end baseline ─────────────────────────────────────────────────────────

    # ── Optional: jerk asymmetry (+6) ────────────────────────────────────────
    if use_jerk_asymmetry:
        for ch in [jx, jy, jz]:
            parts.append(np.where(ch > 0, ch, 0).mean(1))
            parts.append(np.where(ch < 0, ch, 0).mean(1))
        feature_log['jerk_asymmetry'] = 6

    # ── Optional: temporal trend features (+18) ───────────────────────────────
    # Applied to raw channels: gravity direction makes first/last meaningful.
    # first_60_mean, last_60_mean, last-first for mx, my, mz, sx, sy, sz.
    if use_trend_features:
        for ch in [mx, my, mz, sx, sy, sz]:
            parts += trend3(ch)
        feature_log['trend_features'] = 18

    # ── Optional: diff features (+27) ────────────────────────────────────────
    # diff_mean, diff_std, diff_abs_mean for 9 key signals.
    if use_diff_features:
        for sig in [mx, my, mz, sx, sy, sz, mag_mean, mag_std, mag_jerk]:
            parts += diff_stats3(sig)
        feature_log['diff_features'] = 27

    # ── Optional: extra stats (+35) ──────────────────────────────────────────
    # energy, MAD, RMS, q05, q10, q90, q95 for 5 key signals.
    if use_extra_stats:
        for sig in [sx, sy, sz, mag_std, mag_mean]:
            parts += stats_extra7(sig)
        feature_log['extra_stats'] = 35

    # ── Optional: better peak features (+6) ──────────────────────────────────
    # num_peaks, mean_peak_height, max_peak_height for mag_jerk and mag_std.
    if use_better_peaks:
        for sig in [mag_jerk, mag_std]:
            parts += peak_stats3(sig)
        feature_log['better_peaks'] = 6

    X_out = np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1) for p in parts
    ]).astype(np.float32)

    return X_out, feature_log


# ──────────────────────────────────────────────────────────────────────────────
# USER-CONTEXTUAL FEATURES
# Z-scores relative to each user's own feature distribution.
# Works on raw features too — captures "how unusual is this window for this user?"
# ──────────────────────────────────────────────────────────────────────────────
N_CTX = 45

def add_user_context(X_feat, user_ids):
    user_mean, user_std = {}, {}
    for uid in np.unique(user_ids):
        idx = np.where(user_ids == uid)[0]
        user_mean[uid] = X_feat[idx, :N_CTX].mean(0)
        user_std[uid]  = X_feat[idx, :N_CTX].std(0) + 1e-8
    ctx = np.zeros((len(X_feat), N_CTX), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        if uid in user_mean:
            ctx[i] = (X_feat[i, :N_CTX] - user_mean[uid]) / user_std[uid]
    return np.hstack([X_feat, ctx]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACT FEATURES
# ──────────────────────────────────────────────────────────────────────────────
print("\nExtracting features...")
X_feat, feature_log = extract(
    X_tr,
    use_jerk_asymmetry = USE_JERK_ASYMMETRY,
    use_trend_features = USE_TREND_FEATURES,
    use_diff_features  = USE_DIFF_FEATURES,
    use_extra_stats    = USE_EXTRA_STATS,
    use_better_peaks   = USE_BETTER_PEAKS,
)

if USE_USER_CONTEXT:
    X_feat = add_user_context(X_feat, users)
    feature_log['user_context'] = 45

total_features = X_feat.shape[1]
print(f"  Feature breakdown:")
for name, n in feature_log.items():
    print(f"    {name:<22} +{n}")
print(f"  Total: {total_features} features")


# ──────────────────────────────────────────────────────────────────────────────
# FAST 3-FOLD LOO-CV
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FAST 3-FOLD LOO-CV  (no augmentation, early stopping)")
print("="*60)
print("Run with all toggles False first to get your no-normalization baseline.\n")

user_folds = {u: i % 3 for i, u in enumerate(unique_users)}
fold_ids   = np.array([user_folds[u] for u in users])

all_preds  = np.zeros(len(y_tr), dtype=int)
fold_results = []

for fold in range(3):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]

    sc        = StandardScaler()
    X_fold_sc = sc.fit_transform(X_feat[tr_idx])
    X_va_sc   = sc.transform(X_feat[va_idx])

    m = lgb.LGBMClassifier(
        n_estimators     = 1000,
        learning_rate    = 0.05,
        num_leaves       = 63,
        class_weight     = 'balanced',
        min_child_samples= 20,
        reg_alpha        = 0.5,
        reg_lambda       = 1.0,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = SEED,
        n_jobs           = -1,
        verbose          = -1,
    )
    m.fit(
        X_fold_sc, y_tr[tr_idx],
        eval_set=[(X_va_sc, y_tr[va_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    preds  = m.predict_proba(X_va_sc).argmax(1)
    all_preds[va_idx] = preds

    macro  = f1_score(y_tr[va_idx], preds, average='macro')
    per_cl = f1_score(y_tr[va_idx], preds, average=None)
    fold_results.append({'fold': fold+1, 'n_val': len(va_idx),
                         'n_trees': m.best_iteration_,
                         'macro': macro, 'per_class': per_cl})

    print(f"Fold {fold+1}/3  val={len(va_idx):4d}  trees={m.best_iteration_:3d}  "
          f"macro F1={macro:.4f}")
    print(f"  per-class: " +
          "  ".join(f"C{c}={per_cl[c]:.3f}" for c in range(6)))


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
oof_macro  = f1_score(y_tr, all_preds, average='macro')
oof_per_cl = f1_score(y_tr, all_preds, average=None)
avg_trees  = int(np.mean([r['n_trees'] for r in fold_results]))

print("\n" + "="*60)
print("SCREENING RESULTS")
print("="*60)

enabled = [k for k in feature_log if k != 'baseline_373']
print(f"\n  Active toggles:    {enabled if enabled else ['none — baseline only']}")
print(f"  Total features:    {total_features}")
print(f"  Avg trees (early stop): {avg_trees} of max 1000")
print(f"  No per-user normalization applied")

print(f"\n  OOF macro F1:  {oof_macro:.4f}")
print(f"\n  Per-class OOF F1:")
CLASS_NAMES = ["sit/stand", "walk flat", "walk down", "walk up", "running", "other"]
for c in range(6):
    bar  = "█" * int(oof_per_cl[c] * 20)
    flag = "  ← main bottleneck" if c == 2 else ""
    print(f"    C{c} {CLASS_NAMES[c]:<14} {oof_per_cl[c]:.3f}  {bar}{flag}")

print(f"\n  Fold breakdown:")
print(f"  {'Fold':>4}  {'Macro':>6}  {'C0':>5}  {'C1':>5}  "
      f"{'C2':>5}  {'C3':>5}  {'C4':>5}  {'C5':>5}  {'Trees':>5}")
print(f"  {'-'*56}")
for r in fold_results:
    pc = r['per_class']
    print(f"  {r['fold']:>4}  {r['macro']:>6.4f}  " +
          "  ".join(f"{pc[c]:>5.3f}" for c in range(6)) +
          f"  {r['n_trees']:>5}")

print(f"\n  Decision rule:")
print(f"    Improvement > 0.005 macro F1 vs your baseline → run full model")
print(f"    C2 F1 improvement → most important signal for this competition")
