"""
model_run05.py — LightGBM ensemble, no CNN
Why no CNN: CNN scored 0.8012 LOO-CV but hurt the Kaggle score when ensembled.
LightGBM alone scored 0.8722 LOO-CV → target ~0.82 on Kaggle.

Key improvements:
  1. LightGBM only (no CNN dragging down the ensemble)
  2. 10 LightGBM models with different seeds/params → averaged probabilities
  3. Both raw and within-window normalised features combined
  4. Saves directly to /kaggle/working/ automatically
Output: submission_run05.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb

# ── Output path: always /kaggle/working if on Kaggle, else local outputs/ ─────
if Path("/kaggle/working").exists():
    OUT_DIR = Path("/kaggle/working")
else:
    OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD NPZ  — searches /kaggle/input recursively, then local outputs/
# ══════════════════════════════════════════════════════════════════════════════

def find_npz(name: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return hits[0]
    local = Path(__file__).parent / "outputs" / name
    if local.exists():
        return str(local)
    raise FileNotFoundError(f"{name} not found in /kaggle/input or local outputs/")

print("Loading npz files …")
train_data = np.load(find_npz("train_data.npz"), allow_pickle=True)
test_data  = np.load(find_npz("test_data.npz"),  allow_pickle=True)

X_train_raw = np.nan_to_num(train_data["X"].astype(np.float32), nan=0.0)
y_train     = train_data["y"].astype(np.int32)
train_users = train_data["users"]
test_ids    = test_data["file_ids"]
X_test_raw  = np.nan_to_num(test_data["X"].astype(np.float32),  nan=0.0)

unique_users = np.unique(train_users)
unique, counts = np.unique(y_train, return_counts=True)
print(f"Train: {X_train_raw.shape}  Test: {X_test_raw.shape}  Users: {len(unique_users)}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def window_norm(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

X_train_norm = window_norm(X_train_raw)
X_test_norm  = window_norm(X_test_raw)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract(X: np.ndarray) -> np.ndarray:
    N, T, C = X.shape
    parts = []

    # Global statistics per channel
    for c in range(C):
        s = X[:, :, c]
        parts += [s.mean(1), s.std(1), s.min(1), s.max(1),
                  s.max(1)-s.min(1), np.median(s, 1),
                  np.percentile(s, 75, 1)-np.percentile(s, 25, 1),
                  np.array([skew(r)     for r in s]),
                  np.array([kurtosis(r) for r in s])]

    # Vector magnitude
    mag = np.sqrt((X[:, :, :3]**2).sum(2))
    parts += [mag.mean(1), mag.std(1), mag.max(1)-mag.min(1),
              np.percentile(mag, 25, 1), np.percentile(mag, 75, 1)]

    # Temporal segments — 5, 10, 20 resolutions
    for n_seg in [5, 10, 20]:
        sl = T // n_seg
        for i in range(n_seg):
            seg = X[:, i*sl:(i+1)*sl, :]
            parts += [seg.mean(1), seg.std(1)]

    # Autocorrelation
    for lag in [1, 2, 5, 10, 20, 30, 60]:
        ac = np.zeros((N, C), dtype=np.float32)
        for c in range(C):
            s = X[:,:,c]; s1,s2 = s[:,:-lag],s[:,lag:]
            num = ((s1-s1.mean(1,keepdims=True))*(s2-s2.mean(1,keepdims=True))).mean(1)
            ac[:,c] = num/(s1.std(1)*s2.std(1)+1e-10)
        parts.append(ac)

    # Linear trend
    t = np.arange(T, dtype=np.float32)-T/2
    sl_arr = np.zeros((N,C), dtype=np.float32)
    for c in range(C):
        sl_arr[:,c] = (X[:,:,c]*t).sum(1)/(t**2).sum()
    parts.append(sl_arr)

    # Cross-channel correlations
    for a,b in [(0,1),(0,2),(1,2)]:
        sa = X[:,:,a]-X[:,:,a].mean(1,keepdims=True)
        sb = X[:,:,b]-X[:,:,b].mean(1,keepdims=True)
        parts.append(((sa*sb).mean(1)/(sa.std(1)*sb.std(1)+1e-10)).reshape(-1,1))

    # Zero-crossing rate
    zcr = np.zeros((N,C), dtype=np.float32)
    for c in range(C):
        s = X[:,:,c]-X[:,:,c].mean(1,keepdims=True)
        zcr[:,c] = (np.diff(np.sign(s), axis=1)!=0).sum(1)/T
    parts.append(zcr)

    # Spectral features
    spec = np.zeros((N, 5*C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            sig = X[n,:,c]
            freqs, psd = welch(sig, fs=1.0, nperseg=min(64,T))
            pn = psd/(psd.sum()+1e-10)
            spec[n, c*5:(c+1)*5] = [
                freqs[np.argmax(psd)],
                -np.sum(pn*np.log(pn+1e-10)),
                psd[(freqs>=0)&(freqs<0.5)].sum(),
                psd[(freqs>=0.5)&(freqs<2)].sum(),
                psd[freqs>=2].sum(),
            ]
    parts.append(spec)

    return np.hstack([np.asarray(p).reshape(N,-1) for p in parts]).astype(np.float32)


print("\nExtracting features from normalised data …")
X_tr_norm_feat = extract(X_train_norm)
X_te_norm_feat = extract(X_test_norm)

print("Extracting features from raw data …")
X_tr_raw_feat  = extract(X_train_raw)
X_te_raw_feat  = extract(X_test_raw)

# Combine both normalised and raw features
X_tr_combined = np.hstack([X_tr_norm_feat, X_tr_raw_feat])
X_te_combined = np.hstack([X_te_norm_feat, X_te_raw_feat])

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_combined)
X_te_sc = scaler.transform(X_te_combined)
print(f"Feature matrix: {X_tr_sc.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LEAVE-USER-OUT CV
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LEAVE-USER-OUT CV  (5 user-folds)")
print("="*60)

user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids   = np.array([user_folds[u] for u in train_users])
loo_preds  = np.zeros(len(y_train), dtype=int)

LGB_CONFIGS = [
    dict(num_leaves=63,  learning_rate=0.05, colsample_bytree=0.8, subsample=0.8),
    dict(num_leaves=127, learning_rate=0.03, colsample_bytree=0.7, subsample=0.9),
    dict(num_leaves=31,  learning_rate=0.1,  colsample_bytree=0.9, subsample=0.7),
]

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")

    fold_probas = []
    for cfg in LGB_CONFIGS:
        m = lgb.LGBMClassifier(
            n_estimators=1000, class_weight="balanced",
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, n_jobs=-1, verbose=-1, **cfg,
        )
        m.fit(X_tr_sc[tr_idx], y_train[tr_idx])
        fold_probas.append(m.predict_proba(X_tr_sc[va_idx]))

    loo_preds[va_idx] = np.mean(fold_probas, axis=0).argmax(axis=1)
    fold_acc = accuracy_score(y_train[va_idx], loo_preds[va_idx])
    print(f"  Fold {fold+1} acc = {fold_acc:.4f}")

loo_acc = accuracy_score(y_train, loo_preds)
print(f"\nOverall LOO-CV accuracy: {loo_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FINAL TRAINING ON ALL DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("FINAL TRAINING ON ALL DATA")
print("="*60)

final_probas = []
for seed, cfg in enumerate(LGB_CONFIGS * 3):   # 9 models total
    m = lgb.LGBMClassifier(
        n_estimators=1000, class_weight="balanced",
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=seed, n_jobs=-1, verbose=-1, **cfg,
    )
    m.fit(X_tr_sc, y_train)
    final_probas.append(m.predict_proba(X_te_sc))
    print(f"  Model {seed+1}/9 trained")

preds = np.mean(final_probas, axis=0).argmax(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE SUBMISSION DIRECTLY TO /kaggle/working
# ══════════════════════════════════════════════════════════════════════════════

submission = pd.DataFrame({"Id": test_ids, "Label": preds})
submission = submission.sort_values("Id").reset_index(drop=True)
out_path   = OUT_DIR / "submission_run05.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission saved: {out_path}")
print(submission["Label"].value_counts().sort_index().to_string())

# Confusion matrix (LOO-CV)
sns.set_style("whitegrid")
cm = confusion_matrix(y_train, loo_preds, normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            xticklabels=range(6), yticklabels=range(6), linewidths=0.5)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"LightGBM Ensemble — LOO-CV Confusion Matrix\nAcc = {loo_acc:.4f}")
plt.tight_layout()
plt.savefig(OUT_DIR / "run05_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'run05_confusion_matrix.png'}")
print(f"\nDone. Submit {out_path} to Kaggle.")
