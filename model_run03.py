"""
model_run03.py — 1D-CNN + LightGBM Ensemble with user-invariant features
Key improvements over run02 (0.7531):
  1. Within-window normalization: removes user-specific DC offset so the model
     generalises to unseen test users instead of memorising training users.
  2. 1D-CNN (PyTorch): learns temporal patterns from raw sequences directly.
  3. Leave-user-out CV: gives a realistic accuracy estimate matching the test condition.
  4. Ensemble: average CNN + LightGBM probabilities.
Output: submission_run03.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not found — running CNN only. pip install lightgbm")

# ── Paths ──────────────────────────────────────────────────────────────────────
def _find_base_dir():
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for comp_dir in kaggle_input.iterdir():
            if (comp_dir / "train" / "train").exists():
                return comp_dir, Path("/kaggle/working")
    try:
        import google.colab
        p = Path("/content/DataMining_Assignment3")
        return p, p / "outputs"
    except ImportError:
        pass
    p = Path(__file__).parent
    return p, p / "outputs"

BASE_DIR, OUT_DIR = _find_base_dir()
TRAIN_DIR = BASE_DIR / "train" / "train"
TEST_DIR  = BASE_DIR / "test"  / "test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"BASE_DIR : {BASE_DIR}")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
print(f"Device   : {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(root_dir: Path):
    sequences, labels, file_ids, users = [], [], [], []
    for user_dir in sorted(root_dir.iterdir()):
        for csv_path in sorted(user_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            sequences.append(df[FEAT_COLS].values.astype(np.float32))
            if "label" in df.columns:
                labels.append(int(df["label"].iloc[0]))
            file_ids.append(int(df["file_id"].iloc[0]))
            users.append(user_dir.name)
    X = np.array(sequences)                          # (N, 300, 6)
    X = np.nan_to_num(X, nan=0.0)
    return X, np.array(labels) if labels else None, np.array(file_ids), np.array(users)

print("Loading data …")
X_train_raw, y_train, train_ids, train_users = load_dataset(TRAIN_DIR)
X_test_raw,  _,       test_ids,  _           = load_dataset(TEST_DIR)
print(f"  Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")

unique_users = np.unique(train_users)
print(f"  Training users: {len(unique_users)}")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. WITHIN-WINDOW NORMALISATION
#    Key fix: each 5-minute window is z-scored using its own mean and std.
#    This removes user-specific DC offset (phone placement / body size)
#    so the model sees activity shape, not absolute acceleration values.
# ══════════════════════════════════════════════════════════════════════════════

def window_normalise(X: np.ndarray) -> np.ndarray:
    mu  = X.mean(axis=1, keepdims=True)          # (N, 1, 6)
    sig = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sig

X_train_norm = window_normalise(X_train_raw)     # (N, 300, 6)
X_test_norm  = window_normalise(X_test_raw)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 1D-CNN
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.MaxPool1d(pool),
        )
    def forward(self, x):
        return self.net(x)


class HARNet(nn.Module):
    """
    Input: (batch, 6, 300)
    Architecture:
      Conv 6→64  k=11 → pool/2 → 150
      Conv 64→128 k=7 → pool/2 → 75
      Conv 128→256 k=5 → pool/3 → 25
      Conv 256→256 k=3 → GlobalAvgPool → 256-d vector
      FC 256→128 → Dropout → FC 128→6
    """
    def __init__(self, n_classes=6):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(6,   64,  11, pool=2),
            ConvBlock(64,  128,  7, pool=2),
            ConvBlock(128, 256,  5, pool=3),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return self.head(x)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random noise + random time-shift."""
    x = x + torch.randn_like(x) * 0.02
    shift = np.random.randint(-30, 30)
    x = torch.roll(x, shift, dims=-1)
    return x


def train_cnn(X_tr, y_tr, X_va=None, y_va=None,
              epochs=30, batch=64, lr=1e-3):
    class_weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # (N, 6, 300) — channels first for Conv1D
    Xt = torch.tensor(X_tr.transpose(0, 2, 1), dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)

    model = HARNet().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best_val_acc, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = augment(xb).to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()

        if X_va is not None and ep % 5 == 0:
            val_acc = predict_cnn(model, X_va).eq(torch.tensor(y_va)).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {ep:3d}/{epochs}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_cnn(model, X: np.ndarray) -> torch.Tensor:
    model.eval()
    Xt = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32).to(DEVICE)
    return model(Xt).argmax(dim=1).cpu()


@torch.no_grad()
def predict_proba_cnn(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32).to(DEVICE)
    return torch.softmax(model(Xt), dim=1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 4. LIGHTGBM FEATURES (from within-window-normalised data)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(X: np.ndarray) -> np.ndarray:
    """237-dim feature vector from within-window-normalised sequences."""
    N, T, C = X.shape
    parts = []

    for c in range(C):
        s = X[:, :, c]
        parts += [s.mean(axis=1), s.std(axis=1), s.min(axis=1), s.max(axis=1),
                  s.max(axis=1)-s.min(axis=1), np.median(s,axis=1),
                  np.percentile(s,75,axis=1)-np.percentile(s,25,axis=1),
                  np.array([skew(r) for r in s]),
                  np.array([kurtosis(r) for r in s])]

    mag = np.sqrt((X[:,:,:3]**2).sum(axis=2))
    parts += [mag.mean(axis=1), mag.std(axis=1), mag.max(axis=1)-mag.min(axis=1)]

    for n_seg in [10, 20]:
        seg_len = T // n_seg
        for i in range(n_seg):
            seg = X[:, i*seg_len:(i+1)*seg_len, :]
            parts += [seg.mean(axis=1), seg.std(axis=1)]

    for lag in [1, 2, 5, 10, 20, 30, 60]:
        ac = np.zeros((N, C), dtype=np.float32)
        for c in range(C):
            s = X[:,:,c]; s1,s2 = s[:,:-lag],s[:,lag:]
            num = ((s1-s1.mean(1,keepdims=True))*(s2-s2.mean(1,keepdims=True))).mean(1)
            ac[:,c] = num / (s1.std(1)*s2.std(1)+1e-10)
        parts.append(ac)

    t = np.arange(T,dtype=np.float32)-T/2
    slopes = np.zeros((N,C),dtype=np.float32)
    for c in range(C):
        slopes[:,c] = (X[:,:,c]*t).sum(1)/(t**2).sum()
    parts.append(slopes)

    for a,b in [(0,1),(0,2),(1,2)]:
        sa = X[:,:,a]-X[:,:,a].mean(1,keepdims=True)
        sb = X[:,:,b]-X[:,:,b].mean(1,keepdims=True)
        cross = (sa*sb).mean(1)/((sa.std(1)*sb.std(1))+1e-10)
        parts.append(cross.reshape(-1,1))

    zcr = np.zeros((N,C),dtype=np.float32)
    for c in range(C):
        s = X[:,:,c]-X[:,:,c].mean(1,keepdims=True)
        zcr[:,c] = (np.diff(np.sign(s),axis=1)!=0).sum(1)/T
    parts.append(zcr)

    spec = np.zeros((N,5*C),dtype=np.float32)
    for n in range(N):
        for c in range(C):
            sig = X[n,:,c]
            freqs,psd = welch(sig, fs=1.0, nperseg=min(64,T))
            pn = psd/(psd.sum()+1e-10)
            spec[n,c*5:(c+1)*5] = [freqs[np.argmax(psd)],
                                    -np.sum(pn*np.log(pn+1e-10)),
                                    psd[(freqs>=0)&(freqs<0.5)].sum(),
                                    psd[(freqs>=0.5)&(freqs<2)].sum(),
                                    psd[freqs>=2].sum()]
    parts.append(spec)

    flat = []
    for p in parts:
        arr = np.asarray(p)
        flat.append(arr.reshape(N,-1))
    return np.hstack(flat).astype(np.float32)


print("\nExtracting LightGBM features …")
X_tr_feat = extract_features(X_train_norm)
X_te_feat = extract_features(X_test_norm)
scaler    = StandardScaler()
X_tr_sc   = scaler.fit_transform(X_tr_feat)
X_te_sc   = scaler.transform(X_te_feat)
print(f"  Feature matrix: {X_tr_sc.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LEAVE-USER-OUT CV  (realistic evaluation)
#    Matches the actual test condition: model trained on some users,
#    evaluated on held-out users it has never seen.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LEAVE-USER-OUT CV (realistic cross-user evaluation)")
print("="*60)

cnn_loo_preds = np.zeros(len(y_train), dtype=int)
lgb_loo_preds = np.zeros(len(y_train), dtype=int)

# Group users into 5 folds
user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids   = np.array([user_folds[u] for u in train_users])

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    print(f"\nFold {fold+1}/5  train={len(tr_idx)}  val={len(va_idx)}")

    # CNN fold
    print("  Training CNN …")
    cnn_model = train_cnn(X_train_norm[tr_idx], y_train[tr_idx],
                          X_va=X_train_norm[va_idx], y_va=y_train[va_idx],
                          epochs=60, batch=64)
    cnn_loo_preds[va_idx] = predict_cnn(cnn_model, X_train_norm[va_idx]).numpy()

    # LightGBM fold
    if HAS_LGB:
        print("  Training LightGBM …")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
        lgb_model.fit(X_tr_sc[tr_idx], y_train[tr_idx])
        lgb_loo_preds[va_idx] = lgb_model.predict(X_tr_sc[va_idx])

cnn_acc = accuracy_score(y_train, cnn_loo_preds)
lgb_acc = accuracy_score(y_train, lgb_loo_preds) if HAS_LGB else 0
print(f"\nLeave-User-Out CV results:")
print(f"  CNN      : {cnn_acc:.4f}")
if HAS_LGB:
    print(f"  LightGBM : {lgb_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FINAL TRAINING ON ALL DATA + ENSEMBLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("FINAL TRAINING ON ALL DATA")
print("="*60)

print("Training final CNN …")
final_cnn = train_cnn(X_train_norm, y_train, epochs=40, batch=64)
cnn_proba = predict_proba_cnn(final_cnn, X_test_norm)  # (N_test, 6)

if HAS_LGB:
    print("Training final LightGBM …")
    final_lgb = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    final_lgb.fit(X_tr_sc, y_train)
    lgb_proba = final_lgb.predict_proba(X_te_sc)        # (N_test, 6)
    final_proba = (cnn_proba + lgb_proba) / 2
else:
    final_proba = cnn_proba

preds = final_proba.argmax(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

submission = pd.DataFrame({"Id": test_ids, "Label": preds})
submission = submission.sort_values("Id").reset_index(drop=True)
submission_path = OUT_DIR / "submission_run03.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved: {submission_path}")
print("Prediction distribution:")
print(submission["Label"].value_counts().sort_index().to_string())

# Confusion matrix (LOO CV)
sns.set_style("whitegrid")
cm_norm = confusion_matrix(y_train, cnn_loo_preds, normalize="true")
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            xticklabels=range(6), yticklabels=range(6), linewidths=0.5)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"CNN Leave-User-Out CV — Normalised Confusion Matrix\nAcc = {cnn_acc:.4f}")
plt.tight_layout()
plt.savefig(OUT_DIR / "run03_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: run03_confusion_matrix.png")

# CV summary bar
fig, ax = plt.subplots(figsize=(6,4))
bars_data = {"CNN (LOO-CV)": cnn_acc}
if HAS_LGB: bars_data["LightGBM (LOO-CV)"] = lgb_acc
bars_data["run02 Kaggle"] = 0.7531
colors = ["#3498db","#2ecc71","#e74c3c"]
ax.bar(bars_data.keys(), bars_data.values(), color=colors[:len(bars_data)], edgecolor="white")
for i,(k,v) in enumerate(bars_data.items()):
    ax.text(i, v+0.005, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_title("model_run03 — Leave-User-Out CV vs Previous Score")
ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig(OUT_DIR / "run03_cv_summary.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: run03_cv_summary.png")

print(f"\nDone. Submit {submission_path} to Kaggle.")
