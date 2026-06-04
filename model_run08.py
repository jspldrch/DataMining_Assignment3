"""
model_run08.py — run07 + TTA + minority oversampling

Progress so far (LOO-CV gap shrinking = less overfitting to training users):
  run05: LOO-CV 0.8874  Kaggle 0.7337  gap 0.154
  run06: LOO-CV 0.8824  Kaggle 0.7501  gap 0.132
  run07: LOO-CV 0.8698  Kaggle 0.7707  gap 0.099  ← best

New in run08:
  1. Test-Time Augmentation (TTA): predict from 5 time-shifted versions of
     each test window and average → reduces prediction variance.
  2. Minority class oversampling: classes 2 and 4 duplicated with small noise
     to address severe class imbalance (2%→target 8%, 1%→target 6%).
  3. Slightly more aggressive regularization.
  4. Stronger minority class probability boost (cap raised to 5x).

Output: submission_run08.csv → /kaggle/working/
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


# ══════════════════════════════════════════════════════════════════════════════
# 1. PER-USER NORMALISATION (same as run07)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 2. MINORITY CLASS OVERSAMPLING
#    Classes 2 and 4 are tiny (3.2% and 1.3%). Duplicate their windows with
#    tiny noise so the model sees more examples during training.
# ══════════════════════════════════════════════════════════════════════════════

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

print("\nOversampling minority classes …")
X_tr_os, y_tr_os, users_os = oversample(X_tr, y_tr, users)
unique_os, counts_os = np.unique(y_tr_os, return_counts=True)
print(f"  After oversampling: {X_tr_os.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE EXTRACTION (same as run07)
# ══════════════════════════════════════════════════════════════════════════════

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

def extract(X):
    N, T, _ = X.shape
    mx,my,mz = X[:,:,0],X[:,:,1],X[:,:,2]
    sx,sy,sz = X[:,:,3],X[:,:,4],X[:,:,5]
    jx,jy,jz = np.diff(mx,axis=1),np.diff(my,axis=1),np.diff(mz,axis=1)
    mag_mean = np.sqrt(mx**2+my**2+mz**2)
    mag_std  = np.sqrt(sx**2+sy**2+sz**2)
    mag_jerk = np.sqrt(jx**2+jy**2+jz**2)
    parts = []
    for ch in [sx,sy,sz]:           parts += stats9(ch)
    parts += stats9(mag_std)
    parts += stats9(mag_mean)
    for ch in [jx,jy,jz]:          parts += stats9(ch)
    parts += stats9(mag_jerk)
    for sig in [mag_std,mag_jerk]:
        for ns in [10,20]:          parts += seg(sig, ns)
    for ch in [sx,sy,sz]:          parts += seg(ch, 10)
    for ch in [jx,jy,jz]:         parts += seg(ch, 10)
    for lag in [1,2,5,10,20,30,60]: parts.append(ac(mag_jerk, lag))
    for ch in [sx,sy,sz]:
        for lag in [1,5,10,30]:     parts.append(ac(ch, lag))
    for sig in [mag_jerk,mag_std,sx,sy,sz]: parts.append(spectral5(sig))
    for a,b in [(jx,jy),(jx,jz),(jy,jz)]:  parts.append(xcorr(a,b))
    for a,b in [(sx,sy),(sx,sz),(sy,sz)]:   parts.append(xcorr(a,b))
    cj = mag_jerk-mag_jerk.mean(1,keepdims=True)
    parts.append((np.diff(np.sign(cj),axis=1)!=0).sum(1)/T)
    pr = np.zeros(N,dtype=np.float32)
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n],height=mag_jerk[n].mean())[0])/T
    parts.append(pr)
    return np.column_stack([
        np.asarray(p).reshape(N,-1) if np.asarray(p).ndim>1
        else np.asarray(p).reshape(N,1) for p in parts
    ]).astype(np.float32)


print("\nExtracting features …")
X_tr_feat = extract(X_tr_os)   # oversampled training data
X_te_feat = extract(X_te)
print(f"  Train: {X_tr_feat.shape}  Test: {X_te_feat.shape}")

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_feat)
X_te_sc = scaler.transform(X_te_feat)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LEAVE-USER-OUT CV (on original data — no oversampling in CV eval)
# ══════════════════════════════════════════════════════════════════════════════

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

    # Oversample only the training fold
    X_fold, y_fold, _ = oversample(X_tr[tr_idx], y_tr[tr_idx], users[tr_idx])
    X_fold_feat = extract(X_fold)
    X_fold_sc   = scaler.transform(X_fold_feat)

    probas = []
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=500, class_weight="balanced",
            min_child_samples=20, reg_alpha=0.5, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_fold_sc, y_fold)
        probas.append(m.predict_proba(X_tr_orig_sc[va_idx]))

    loo_preds[va_idx] = np.mean(probas, axis=0).argmax(axis=1)
    print(f"  Fold acc = {accuracy_score(y_tr[va_idx], loo_preds[va_idx]):.4f}")

loo_acc = accuracy_score(y_tr, loo_preds)
print(f"\nOverall LOO-CV accuracy: {loo_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FINAL TRAINING + TEST-TIME AUGMENTATION (TTA)
#    Predict from 5 time-shifted versions of the test windows.
#    At 1Hz, rolling 10s shifts the signal but preserves all activity info.
#    Averaging predictions reduces variance → more stable results.
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════

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
