"""
model_run07.py — Best-of-both feature set + regularized LightGBM

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

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() \
          else Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

def find_npz(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits: return hits[0]
    local = Path(__file__).parent / "outputs" / name
    if local.exists(): return str(local)
    raise FileNotFoundError(f"{name} not found")

print("Loading npz …")
tr = np.load(find_npz("train_data.npz"), allow_pickle=True)
te = np.load(find_npz("test_data.npz"),  allow_pickle=True)

X_tr   = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_tr   = tr["y"].astype(np.int32)
users  = tr["users"]
X_te   = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
te_ids = te["file_ids"]
te_users = te["users"]   # test user IDs — available for normalization

unique_users = np.unique(users)
unique, counts = np.unique(y_tr, return_counts=True)
print(f"Train {X_tr.shape}  Test {X_te.shape}  Users {len(unique_users)}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_tr)*100:.1f}%)")

# PER-USER NORMALISATION

def user_normalise(X, user_ids):
    X_out = X.copy()
    for uid in np.unique(user_ids):
        idx = np.where(user_ids == uid)[0]
        data = X[idx]                                   # (n_windows, 300, 6)
        mu   = data.mean(axis=(0, 1), keepdims=True)    # (1, 1, 6)
        sig  = data.std(axis=(0, 1), keepdims=True) + 1e-8
        X_out[idx] = (data - mu) / sig
    return X_out

print("\nApplying per-user normalisation …")
X_tr = user_normalise(X_tr, users)
X_te = user_normalise(X_te, te_users)
print("  Done.")

#features

def stats9(s):
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s,1),
            np.percentile(s,75,1)-np.percentile(s,25,1),
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
    num = ((s1-s1.mean(1,keepdims=True))*(s2-s2.mean(1,keepdims=True))).mean(1)
    return num / (s1.std(1)*s2.std(1)+1e-10)

def seg(s, n_seg):
    N, T = s.shape; sl = T // n_seg; out = []
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1)]
    return out

def xcorr(a, b):
    ca = a - a.mean(1,keepdims=True)
    cb = b - b.mean(1,keepdims=True)
    return (ca*cb).mean(1) / (ca.std(1)*cb.std(1)+1e-10)

def extract(X):
    N, T, _ = X.shape
    mx,my,mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx,sy,sz = X[:,:,3], X[:,:,4], X[:,:,5]

    # ── Gravity-free mean-channel signals ─────────────────────────────────────
    # Jerk removes DC (gravity) at 1Hz: np.diff IS the correct high-pass filter.
    jx = np.diff(mx, axis=1)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)

    # ── Rotation-invariant magnitudes ─────────────────────────────────────────
    mag_mean = np.sqrt(mx**2 + my**2 + mz**2)   # ≈ 1g static, deviates for motion
    mag_std  = np.sqrt(sx**2 + sy**2 + sz**2)   # total activity intensity
    mag_jerk = np.sqrt(jx**2 + jy**2 + jz**2)  # motion change intensity

    parts = []

    # A. Std channels — all stats (9×3=27) ────────────────────────────────────
    for ch in [sx, sy, sz]:
        parts += stats9(ch)

    # B. Std magnitude (9) ────────────────────────────────────────────────────
    parts += stats9(mag_std)

    # C. Mean magnitude — keeps orientation-invariant gravity info (9) ─────────
    parts += stats9(mag_mean)

    # D. Jerk per axis (9×3=27) ───────────────────────────────────────────────
    for ch in [jx, jy, jz]:
        parts += stats9(ch)

    # E. Jerk magnitude (9) ───────────────────────────────────────────────────
    parts += stats9(mag_jerk)

    # F. Segments of mag_std, mag_jerk at 10 and 20 scales (80) ───────────────
    for sig in [mag_std, mag_jerk]:
        for ns in [10, 20]:
            parts += seg(sig, ns)

    # G. Segments of each std channel at scale 10 (60) ────────────────────────
    for ch in [sx, sy, sz]:
        parts += seg(ch, 10)

    # H. Segments of each jerk channel at scale 10 (60) ──────────────────────
    for ch in [jx, jy, jz]:
        parts += seg(ch, 10)

    # I. Autocorrelation: mag_jerk (7 lags) ───────────────────────────────────
    for lag in [1, 2, 5, 10, 20, 30, 60]:
        parts.append(ac(mag_jerk, lag))

    # J. Autocorrelation: std channels (4 lags × 3 = 12) ─────────────────────
    for ch in [sx, sy, sz]:
        for lag in [1, 5, 10, 30]:
            parts.append(ac(ch, lag))

    # K. Spectral: mag_jerk, mag_std, each std channel (5×5=25) ───────────────
    for sig in [mag_jerk, mag_std, sx, sy, sz]:
        parts.append(spectral5(sig))

    # L. Cross-correlations: jerk axes + std axes (6 pairs) ───────────────────
    for a,b in [(jx,jy),(jx,jz),(jy,jz)]:
        parts.append(xcorr(a,b))
    for a,b in [(sx,sy),(sx,sz),(sy,sz)]:
        parts.append(xcorr(a,b))

    # M. Zero-crossing rate of mag_jerk (1) ───────────────────────────────────
    cj = mag_jerk - mag_jerk.mean(1,keepdims=True)
    parts.append((np.diff(np.sign(cj),axis=1)!=0).sum(1)/T)

    # N. Peak rate of mag_jerk (1) ────────────────────────────────────────────
    pr = np.zeros(N, dtype=np.float32)
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n], height=mag_jerk[n].mean())[0]) / T
    parts.append(pr)

    return np.column_stack([
        np.asarray(p).reshape(N,-1) if np.asarray(p).ndim>1
        else np.asarray(p).reshape(N,1) for p in parts
    ]).astype(np.float32)


print("\nExtracting features (train) …")
X_tr_feat = extract(X_tr)
print(f"  Train: {X_tr_feat.shape}")

print("Extracting features (test) …")
X_te_feat = extract(X_te)

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_feat)
X_te_sc = scaler.transform(X_te_feat)


# LEAVE-USER-OUT CV 

print("\n" + "="*60)
print("LEAVE-USER-OUT CV")
print("="*60)

user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids   = np.array([user_folds[u] for u in users])
loo_preds  = np.zeros(len(y_tr), dtype=int)

# Regularized LightGBM configs (reduced complexity vs run06)
CONFIGS = [
    dict(num_leaves=31,  learning_rate=0.05, colsample_bytree=0.7, subsample=0.7),
    dict(num_leaves=63,  learning_rate=0.03, colsample_bytree=0.8, subsample=0.8),
    dict(num_leaves=127, learning_rate=0.02, colsample_bytree=0.7, subsample=0.7),
]

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")

    probas = []
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500,           # reduced from 2000 — less memorisation
            class_weight="balanced",
            min_child_samples=20,       # increased — prevents leaf-level overfitting
            reg_alpha=0.5,              # stronger L1
            reg_lambda=1.0,             # stronger L2
            random_state=42, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
        probas.append(m.predict_proba(X_tr_sc[va_idx]))

    loo_preds[va_idx] = np.mean(probas, axis=0).argmax(axis=1)
    print(f"  Fold acc = {accuracy_score(y_tr[va_idx], loo_preds[va_idx]):.4f}")

loo_acc = accuracy_score(y_tr, loo_preds)
print(f"\nOverall LOO-CV accuracy: {loo_acc:.4f}")

# FINAL TRAINING

print("\n" + "="*60)
print("FINAL TRAINING ON ALL DATA")
print("="*60)

final_probas = []
for seed in range(5):
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

avg_proba = np.mean(final_probas, axis=0)   # (N_test, 6)

# Boost minority classes 2 and 4 whose predicted counts are far below expected.
# Scale factor = expected_fraction / mean_predicted_prob so the model
# predicts them at a rate closer to their training prevalence.
train_freq = np.array([counts[i]/len(y_tr) for i in range(6)])
pred_freq  = avg_proba.mean(axis=0)
boost = np.where(pred_freq > 0, train_freq / pred_freq, 1.0)
boost = np.clip(boost, 0.5, 3.0)   # don't over-correct
avg_proba_boosted = avg_proba * boost
avg_proba_boosted /= avg_proba_boosted.sum(axis=1, keepdims=True)

preds = avg_proba_boosted.argmax(axis=1)
print(f"  Trained {len(final_probas)} models")
print(f"  Class freq boost factors: {np.round(boost, 2)}")

# ── Save ───────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out = OUT_DIR / "submission_run07.csv"
sub.to_csv(out, index=False)
print(f"\nSubmission saved: {out}")
print(sub["Label"].value_counts().sort_index().to_string())

cm = confusion_matrix(y_tr, loo_preds, normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            xticklabels=range(6), yticklabels=range(6), linewidths=0.5)
ax.set_title(f"LOO-CV Confusion Matrix — run07\nAcc = {loo_acc:.4f}")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(OUT_DIR / "run07_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Done. Submit: {out}")
