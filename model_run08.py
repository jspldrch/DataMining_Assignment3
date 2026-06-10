"""
model_run08.py — run07 + TTA + minority oversampling

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

X_tr_raw = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_tr     = tr["y"].astype(np.int32)
users    = tr["users"]
X_te_raw = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
te_ids   = te["file_ids"]
te_users = te["users"]

unique_users = np.unique(users)
unique, counts = np.unique(y_tr, return_counts=True)
print(f"Train {X_tr_raw.shape}  Test {X_te_raw.shape}  Users {len(unique_users)}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_tr)*100:.1f}%)")

# 1. PER-USER NORMALISATION

def user_normalise(X, user_ids):
    X_out = X.copy()
    for uid in np.unique(user_ids):
        idx  = np.where(user_ids == uid)[0]
        data = X[idx]
        mu   = data.mean(axis=(0, 1), keepdims=True)
        sig  = data.std(axis=(0, 1),  keepdims=True) + 1e-8
        X_out[idx] = (data - mu) / sig
    return X_out

print("\nPer-user normalisation …")
X_tr = user_normalise(X_tr_raw, users)
X_te = user_normalise(X_te_raw, te_users)


# 2. MINORITY CLASS OVERSAMPLING


def oversample(X, y, user_ids, targets={2: 700, 4: 500}):
    rng = np.random.default_rng(42)
    X_aug, y_aug, u_aug = [X], [y], [user_ids]
    for cls, target in targets.items():
        idx = np.where(y == cls)[0]
        needed = max(0, target - len(idx))
        if needed == 0:
            continue
        chosen = rng.choice(idx, needed, replace=True)
        noise  = rng.normal(0, 0.02, X[chosen].shape).astype(np.float32)
        X_aug.append(X[chosen] + noise)
        y_aug.append(np.full(needed, cls, dtype=np.int32))
        u_aug.append(user_ids[chosen])
        print(f"  Class {cls}: {len(idx)} → {len(idx)+needed} samples")
    return (np.vstack(X_aug), np.concatenate(y_aug),
            np.concatenate(u_aug))

# Oversampling disabled — run09 showed it hurts generalization.
# TTA alone is the new element being tested in run08.
X_tr_os, y_tr_os, users_os = X_tr, y_tr, users
print(f"  No oversampling (using original {X_tr_os.shape[0]} windows)")

# 3. FEATURE EXTRACTION 

def stats9(s):
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s,1),
            np.percentile(s,75,1)-np.percentile(s,25,1),
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
                  -np.sum(pn*np.log(pn+1e-10)),
                  psd[(freqs>=0)&(freqs<0.5)].sum(),
                  psd[(freqs>=0.5)&(freqs<2)].sum(),
                  psd[freqs>=2].sum()]
    return out

def ac(s, lag):
    s1, s2 = s[:,:-lag], s[:,lag:]
    num = ((s1-s1.mean(1,keepdims=True))*(s2-s2.mean(1,keepdims=True))).mean(1)
    return num/(s1.std(1)*s2.std(1)+1e-10)

def seg(s, n_seg):
    N, T = s.shape; sl = T//n_seg; out = []
    for i in range(n_seg):
        w = s[:,i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1)]
    return out

def xcorr(a, b):
    ca = a-a.mean(1,keepdims=True); cb = b-b.mean(1,keepdims=True)
    return (ca*cb).mean(1)/(ca.std(1)*cb.std(1)+1e-10)

def perm_entropy(s, order=3):
    """Permutation entropy — measures signal regularity per window."""
    N, T = s.shape
    out  = np.zeros(N, dtype=np.float32)
    for n in range(N):
        x = s[n]
        patterns = np.array([np.argsort(x[i:i+order]) for i in range(T-order)])
        _, cnts = np.unique(patterns, axis=0, return_counts=True)
        p = cnts / cnts.sum()
        out[n] = -np.sum(p * np.log2(p + 1e-10))
    return out

def wavelet_energy(s):
    """Wavelet energy per decomposition level (3 levels = 6 coeffs per signal)."""
    try:
        import pywt
    except ImportError:
        return np.zeros((s.shape[0], 6), dtype=np.float32)
    N, T = s.shape
    out  = np.zeros((N, 6), dtype=np.float32)
    for n in range(N):
        coeffs = pywt.wavedec(s[n], 'db4', level=3)
        for i, c in enumerate(coeffs):
            out[n, i] = np.sum(c**2) / len(c)
    return out

def seg_slopes(s, n_seg=20):
    """Linear slope within each segment — captures local trends."""
    N, T = s.shape; sl = T // n_seg
    t = np.arange(sl, dtype=np.float32) - sl/2
    denom = (t**2).sum()
    out = np.zeros((N, n_seg), dtype=np.float32)
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out[:, i] = (w * t).sum(1) / denom
    return out

def interseg_var(s, n_seg=10):
    """Std of segment means — how much does the signal vary across segments."""
    N, T = s.shape; sl = T // n_seg
    seg_means = np.stack([s[:, i*sl:(i+1)*sl].mean(1) for i in range(n_seg)], axis=1)
    return seg_means.std(1)

def extract(X):
    N, T, _ = X.shape
    mx,my,mz = X[:,:,0],X[:,:,1],X[:,:,2]
    sx,sy,sz = X[:,:,3],X[:,:,4],X[:,:,5]
    jx,jy,jz = np.diff(mx,axis=1),np.diff(my,axis=1),np.diff(mz,axis=1)
    # Second-order jerk (snap): captures sudden movement changes
    snx,sny,snz = np.diff(jx,axis=1),np.diff(jy,axis=1),np.diff(jz,axis=1)

    mag_mean = np.sqrt(mx**2+my**2+mz**2)
    mag_std  = np.sqrt(sx**2+sy**2+sz**2)
    mag_jerk = np.sqrt(jx**2+jy**2+jz**2)
    mag_snap = np.sqrt(snx**2+sny**2+snz**2)

    parts = []

    # ── Existing run07 features ───────────────────────────────────────────────
    for ch in [sx,sy,sz]:              parts += stats9(ch)
    parts += stats9(mag_std)
    parts += stats9(mag_mean)
    for ch in [jx,jy,jz]:             parts += stats9(ch)
    parts += stats9(mag_jerk)
    for sig in [mag_std,mag_jerk]:
        for ns in [10,20]:             parts += seg(sig, ns)
    for ch in [sx,sy,sz]:             parts += seg(ch, 10)
    for ch in [jx,jy,jz]:            parts += seg(ch, 10)
    for lag in [1,2,5,10,20,30,60]:   parts.append(ac(mag_jerk, lag))
    for ch in [sx,sy,sz]:
        for lag in [1,5,10,30]:        parts.append(ac(ch, lag))
    for sig in [mag_jerk,mag_std,sx,sy,sz]: parts.append(spectral5(sig))
    for a,b in [(jx,jy),(jx,jz),(jy,jz)]:  parts.append(xcorr(a,b))
    for a,b in [(sx,sy),(sx,sz),(sy,sz)]:   parts.append(xcorr(a,b))
    cj = mag_jerk-mag_jerk.mean(1,keepdims=True)
    parts.append((np.diff(np.sign(cj),axis=1)!=0).sum(1)/T)
    pr = np.zeros(N,dtype=np.float32)
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n],height=mag_jerk[n].mean())[0])/T
    parts.append(pr)

    # ── NEW: per-user-normalised mean channels (deviation from user baseline) ─
    # After user normalisation, these encode "how different is this window
    # from this user's typical position" → activity-specific.
    for ch in [mx,my,mz]:             parts += stats9(ch)

    # ── NEW: second-order jerk (snap) ────────────────────────────────────────
    parts += stats9(mag_snap)
    for ch in [snx,sny,snz]:          parts += stats9(ch)

    # ── NEW: permutation entropy (regularity) ─────────────────────────────────
    # Low entropy = regular (walking), high = irregular (random motion)
    for sig in [mag_jerk, mag_std, mx, my, mz]:
        parts.append(perm_entropy(sig))

    # ── NEW: wavelet energy per level ─────────────────────────────────────────
    for sig in [mag_jerk, mag_std]:
        parts.append(wavelet_energy(sig))

    # ── NEW: segment slopes (local trends within each 15s sub-window) ────────
    for sig in [mag_jerk, mag_std]:
        parts.append(seg_slopes(sig, n_seg=20))

    # ── NEW: inter-segment variability ────────────────────────────────────────
    for sig in [mag_jerk, mag_std, sx, sy, sz]:
        parts.append(interseg_var(sig, n_seg=10))

    # ── NEW: additional autocorrelation lags for mag_std ─────────────────────
    for lag in [2, 5, 15, 20, 45]:
        parts.append(ac(mag_std, lag))

    # ── NEW: percentile features ──────────────────────────────────────────────
    for sig in [mag_jerk, mag_std]:
        for pct in [10, 25, 75, 90]:
            parts.append(np.percentile(sig, pct, axis=1))

    return np.column_stack([
        np.asarray(p).reshape(N,-1) if np.asarray(p).ndim>1
        else np.asarray(p).reshape(N,1) for p in parts
    ]).astype(np.float32)


print("\nExtracting features …")
X_tr_feat = extract(X_tr)
X_te_feat = extract(X_te)
print(f"  Train: {X_tr_feat.shape}  Test: {X_te_feat.shape}")

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_feat)
X_te_sc = scaler.transform(X_te_feat)

# 4. LEAVE-USER-OUT CV 

print("\n" + "="*60)
print("LEAVE-USER-OUT CV (original data, no oversampling)")
print("="*60)

X_tr_orig_feat = extract(X_tr)
X_tr_orig_sc   = scaler.transform(X_tr_orig_feat)

user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids   = np.array([user_folds[u] for u in users])
loo_preds  = np.zeros(len(y_tr), dtype=int)

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
            n_estimators=500, class_weight="balanced",
            min_child_samples=20, reg_alpha=0.5, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_orig_sc[tr_idx], y_tr[tr_idx])
        probas.append(m.predict_proba(X_tr_orig_sc[va_idx]))

    loo_preds[va_idx] = np.mean(probas, axis=0).argmax(axis=1)
    print(f"  Fold acc = {accuracy_score(y_tr[va_idx], loo_preds[va_idx]):.4f}")

loo_acc = accuracy_score(y_tr, loo_preds)
print(f"\nOverall LOO-CV accuracy: {loo_acc:.4f}")

# 5. FINAL TRAINING + TEST-TIME AUGMENTATION

print("\n" + "="*60)
print("FINAL TRAINING + TTA")
print("="*60)

# Train final models on ALL oversampled training data
final_models = []
for seed in range(5):
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500, class_weight="balanced",
            min_child_samples=20, reg_alpha=0.5, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc, y_tr_os)
        final_models.append(m)

print(f"  Trained {len(final_models)} models")

# TTA: 5 time shifts → average probabilities
TTA_SHIFTS = [-20, -10, 0, 10, 20]
tta_probas = []

for shift in TTA_SHIFTS:
    if shift == 0:
        X_tta = X_te
    else:
        X_tta = np.roll(X_te_raw, shift, axis=1)
        X_tta = user_normalise(X_tta, te_users)   # re-normalise shifted data
    feat = extract(X_tta)
    sc   = scaler.transform(feat)
    shift_proba = np.mean([m.predict_proba(sc) for m in final_models], axis=0)
    tta_probas.append(shift_proba)
    print(f"  TTA shift={shift:+d}s  done")

avg_proba = np.mean(tta_probas, axis=0)

# Minority class probability boost
train_freq = np.array([counts[i]/len(y_tr) for i in range(6)])
pred_freq  = avg_proba.mean(axis=0)
boost = np.where(pred_freq > 0, train_freq / pred_freq, 1.0)
boost = np.clip(boost, 0.5, 5.0)    # raised cap to 5x for minority classes
avg_proba_boosted  = avg_proba * boost
avg_proba_boosted /= avg_proba_boosted.sum(axis=1, keepdims=True)

preds = avg_proba_boosted.argmax(axis=1)
print(f"\n  Class freq boost factors: {np.round(boost, 2)}")

# 6. SAVE

sub = pd.DataFrame({"Id": te_ids, "Label": preds})
sub = sub.sort_values("Id").reset_index(drop=True)
out = OUT_DIR / "submission_run08.csv"
sub.to_csv(out, index=False)
print(f"\nSubmission saved: {out}")
print(sub["Label"].value_counts().sort_index().to_string())

cm = confusion_matrix(y_tr, loo_preds, normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            xticklabels=range(6), yticklabels=range(6), linewidths=0.5)
ax.set_title(f"LOO-CV — run08\nAcc = {loo_acc:.4f}")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(OUT_DIR / "run08_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Done. Submit: {out}")
