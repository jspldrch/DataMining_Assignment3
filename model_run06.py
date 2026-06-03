"""
model_run06.py — Physics-informed user-invariant features + LightGBM ensemble

Root cause of LOO-CV vs Kaggle gap:
  mean_x, mean_y, mean_z contain the GRAVITY COMPONENT, which depends on
  phone placement (orientation). Different users → different phone placement
  → model memorises phone orientation instead of activity.

Fix: build ONLY user-invariant features:
  1. JERK (d/dt of mean channels): gravity is constant → derivative = 0
     So jerk captures pure motion, not phone orientation.
  2. MAGNITUDE |mean| = sqrt(mean_x²+mean_y²+mean_z²): invariant to rotation
  3. STD channels (std_x/y/z): variability already removes DC offset
  4. MAGNITUDE |std| = sqrt(std_x²+std_y²+std_z²): total activity intensity

No within-window normalisation (it removes discriminative signal).
Output: submission_run06.csv → saved directly to /kaggle/working/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb

# ── Output ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() \
          else Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

# ── Load npz ───────────────────────────────────────────────────────────────────
def find_npz(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits: return hits[0]
    local = Path(__file__).parent / "outputs" / name
    if local.exists(): return str(local)
    raise FileNotFoundError(f"{name} not found")

print("Loading npz …")
tr = np.load(find_npz("train_data.npz"), allow_pickle=True)
te = np.load(find_npz("test_data.npz"),  allow_pickle=True)

X_tr  = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)  # (N,300,6)
y_tr  = tr["y"].astype(np.int32)
users = tr["users"]
X_te  = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
te_ids = te["file_ids"]

unique_users = np.unique(users)
unique, counts = np.unique(y_tr, return_counts=True)
print(f"Train {X_tr.shape}  Test {X_te.shape}  Users {len(unique_users)}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_tr)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — user-invariant only
# ══════════════════════════════════════════════════════════════════════════════

def stats9(s):
    """9 statistics for a (N, T) signal array."""
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s,1),
            np.percentile(s,75,1)-np.percentile(s,25,1),
            np.array([skew(r)     for r in s]),
            np.array([kurtosis(r) for r in s])]

def spectral5(s):
    """5 spectral features for a (N, T) signal array."""
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

def autocorr(s, lag):
    s1, s2 = s[:, :-lag], s[:, lag:]
    num = ((s1-s1.mean(1,keepdims=True))*(s2-s2.mean(1,keepdims=True))).mean(1)
    return num / (s1.std(1)*s2.std(1)+1e-10)

def seg_stats(s, n_seg):
    N, T = s.shape
    sl = T // n_seg
    parts = []
    for i in range(n_seg):
        seg = s[:, i*sl:(i+1)*sl]
        parts += [seg.mean(1), seg.std(1)]
    return parts

def extract(X: np.ndarray) -> np.ndarray:
    N, T, _ = X.shape

    # ── Source signals ─────────────────────────────────────────────────────────
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]   # mean channels
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]   # std channels

    # 1. JERK: derivative of mean channels (removes gravity constant)
    jx = np.diff(mx, axis=1)   # (N, 299)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)

    # 2. MAGNITUDES (rotation-invariant)
    mag_mean  = np.sqrt(mx**2 + my**2 + mz**2)        # |mean| ≈ 1g for static
    mag_std   = np.sqrt(sx**2 + sy**2 + sz**2)        # |std|  activity intensity
    mag_jerk  = np.sqrt(jx**2 + jy**2 + jz**2)       # |jerk| motion intensity

    parts = []

    # ── A. Statistics of each std channel (9 × 3 = 27) ──────────────────────
    for ch in [sx, sy, sz]:
        parts += stats9(ch)

    # ── B. Magnitude of std vector (9) ───────────────────────────────────────
    parts += stats9(mag_std)

    # ── C. Mean magnitude |mean| statistics (9) ──────────────────────────────
    parts += stats9(mag_mean)

    # ── D. Jerk per axis — statistics (9 × 3 = 27) ───────────────────────────
    for ch in [jx, jy, jz]:
        parts += stats9(ch)

    # ── E. Jerk magnitude statistics (9) ─────────────────────────────────────
    parts += stats9(mag_jerk)

    # ── F. Temporal segments of mag_std and mag_jerk (20 seg × 2 × 2 = 80) ──
    for sig in [mag_std, mag_jerk]:
        for n_seg in [10, 20]:
            parts += seg_stats(sig, n_seg)

    # ── G. Temporal segments of each std channel (10 × 2 × 3 = 60) ──────────
    for ch in [sx, sy, sz]:
        parts += seg_stats(ch, 10)

    # ── H. Autocorrelation of mag_jerk at multiple lags (7) ──────────────────
    # Captures rhythmic motion (walking cadence etc.)
    for lag in [1, 2, 5, 10, 20, 30, 60]:
        parts.append(autocorr(mag_jerk, lag))

    # ── I. Autocorrelation of each std channel (7 × 3 = 21) ─────────────────
    for ch in [sx, sy, sz]:
        for lag in [1, 5, 10, 30]:
            parts.append(autocorr(ch, lag))

    # ── J. Spectral features of mag_jerk and mag_std (5 × 2 = 10) ───────────
    parts.append(spectral5(mag_jerk))
    parts.append(spectral5(mag_std))

    # ── K. Spectral features of each std channel (5 × 3 = 15) ───────────────
    for ch in [sx, sy, sz]:
        parts.append(spectral5(ch))

    # ── L. Cross-axis correlations of jerk (3 pairs) ─────────────────────────
    for a, b in [(jx,jy), (jx,jz), (jy,jz)]:
        ca = a - a.mean(1,keepdims=True)
        cb = b - b.mean(1,keepdims=True)
        parts.append((ca*cb).mean(1)/(ca.std(1)*cb.std(1)+1e-10))

    # ── M. Cross-axis correlations of std channels (3 pairs) ─────────────────
    for a, b in [(sx,sy), (sx,sz), (sy,sz)]:
        ca = a - a.mean(1,keepdims=True)
        cb = b - b.mean(1,keepdims=True)
        parts.append((ca*cb).mean(1)/(ca.std(1)*cb.std(1)+1e-10))

    # ── N. Zero-crossing rate of mag_jerk (1) ────────────────────────────────
    cj = mag_jerk - mag_jerk.mean(1,keepdims=True)
    parts.append((np.diff(np.sign(cj),axis=1)!=0).sum(1)/T)

    # ── O. Peak features of mag_jerk (2) ─────────────────────────────────────
    peak_rates = np.zeros(N, dtype=np.float32)
    peak_ints  = np.zeros(N, dtype=np.float32)
    for n in range(N):
        sig = mag_jerk[n]
        pks, props = find_peaks(sig, height=sig.mean())
        peak_rates[n] = len(pks) / T
        peak_ints[n]  = props["peak_heights"].mean() if len(pks) > 0 else 0
    parts += [peak_rates, peak_ints]

    return np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1)
        for p in parts
    ]).astype(np.float32)


print("\nExtracting user-invariant features (train) …")
X_tr_feat = extract(X_tr)
print(f"  Train features: {X_tr_feat.shape}")

print("Extracting user-invariant features (test) …")
X_te_feat = extract(X_te)

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr_feat)
X_te_sc  = scaler.transform(X_te_feat)


# ══════════════════════════════════════════════════════════════════════════════
# LEAVE-USER-OUT CV
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LEAVE-USER-OUT CV  (5 user-folds)")
print("="*60)

user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids   = np.array([user_folds[u] for u in users])
loo_preds  = np.zeros(len(y_tr), dtype=int)

CONFIGS = [
    dict(num_leaves=63,  learning_rate=0.05, colsample_bytree=0.8, subsample=0.8),
    dict(num_leaves=127, learning_rate=0.03, colsample_bytree=0.7, subsample=0.9),
    dict(num_leaves=255, learning_rate=0.02, colsample_bytree=0.7, subsample=0.8),
    dict(num_leaves=31,  learning_rate=0.1,  colsample_bytree=0.9, subsample=0.7),
]

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")

    probas = []
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=2000, class_weight="balanced",
            min_child_samples=10, reg_alpha=0.05, reg_lambda=0.05,
            random_state=42, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
        probas.append(m.predict_proba(X_tr_sc[va_idx]))

    loo_preds[va_idx] = np.mean(probas, axis=0).argmax(axis=1)
    print(f"  Fold acc = {accuracy_score(y_tr[va_idx], loo_preds[va_idx]):.4f}")

loo_acc = accuracy_score(y_tr, loo_preds)
print(f"\nOverall LOO-CV accuracy: {loo_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL TRAINING + PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("FINAL TRAINING ON ALL DATA")
print("="*60)

final_probas = []
for seed in range(5):
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=2000, class_weight="balanced",
            min_child_samples=10, reg_alpha=0.05, reg_lambda=0.05,
            random_state=seed, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc, y_tr)
        final_probas.append(m.predict_proba(X_te_sc))

preds = np.mean(final_probas, axis=0).argmax(axis=1)
print(f"  Trained {len(final_probas)} models")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT DIRECTLY TO /kaggle/working
# ══════════════════════════════════════════════════════════════════════════════

sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out = OUT_DIR / "submission_run06.csv"
sub.to_csv(out, index=False)
print(f"\nSubmission saved: {out}")
print(sub["Label"].value_counts().sort_index().to_string())

# Confusion matrix
cm = confusion_matrix(y_tr, loo_preds, normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            xticklabels=range(6), yticklabels=range(6), linewidths=0.5)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"LOO-CV Confusion Matrix — User-invariant features\nAcc = {loo_acc:.4f}")
plt.tight_layout()
plt.savefig(OUT_DIR / "run06_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Done. Submit: {out}")
