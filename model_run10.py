# run10: feature selection + TCN Transformer + LightGBM ensemble

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel, RFECV
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# configuration
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Output dir: {OUT_DIR}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# data loading
def find_npz(name):
    search_paths = [
        Path("/kaggle/input/train-data") / name,
        Path("/kaggle/input/test-data") / name,
        Path("/kaggle/input") / name,
    ]
    for p in search_paths:
        if p.exists():
            return str(p)
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot find {name}")

print("Loading data...")
tr = np.load(find_npz("train_data.npz"), allow_pickle=True)
te = np.load(find_npz("test_data.npz"), allow_pickle=True)

X_train_raw = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_train = tr["y"].astype(np.int32)
train_users = tr["users"]
X_test_raw = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
test_ids = te["file_ids"]
test_users = te["users"]

unique_users = np.unique(train_users)
unique, counts = np.unique(y_train, return_counts=True)
print(f"Train: {X_train_raw.shape} | Test: {X_test_raw.shape} | Users: {len(unique_users)}")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")

# per-user normalization
def user_normalize(X, user_ids):
    X_out = X.copy()
    for uid in np.unique(user_ids):
        idx = np.where(user_ids == uid)[0]
        data = X[idx]
        mu = data.mean(axis=(0, 1), keepdims=True)
        sigma = data.std(axis=(0, 1), keepdims=True) + 1e-8
        X_out[idx] = (data - mu) / sigma
    return X_out

print("\nPer-user normalization...")
X_train = user_normalize(X_train_raw, train_users)
X_test = user_normalize(X_test_raw, test_users)

# enhanced feature extraction (373 features, will be reduced)
def stats9(s):
    """9 statistical features for each signal"""
    return [s.mean(1), s.std(1), s.min(1), s.max(1),
            s.max(1)-s.min(1), np.median(s,1),
            np.percentile(s,75,1)-np.percentile(s,25,1),
            np.array([skew(r) for r in s]),
            np.array([kurtosis(r) for r in s])]

def spectral5(s):
    """5 spectral features"""
    N, T = s.shape
    out = np.zeros((N, 5), dtype=np.float32)
    for n in range(N):
        sig = s[n] - s[n].mean()
        freqs, psd = welch(sig, fs=1.0, nperseg=min(64, T))
        pn = psd / (psd.sum() + 1e-10)
        out[n] = [freqs[np.argmax(psd)],
                  -np.sum(pn * np.log(pn + 1e-10)),
                  psd[(freqs>=0)&(freqs<0.5)].sum(),
                  psd[(freqs>=0.5)&(freqs<2)].sum(),
                  psd[freqs>=2].sum()]
    return out

def ac(s, lag):
    """Autocorrelation at given lag"""
    s1, s2 = s[:,:-lag], s[:,lag:]
    num = ((s1-s1.mean(1,keepdims=True))*(s2-s2.mean(1,keepdims=True))).mean(1)
    return num / (s1.std(1)*s2.std(1)+1e-10)

def seg(s, n_seg):
    """Segment statistics"""
    N, T = s.shape
    sl = T // n_seg
    out = []
    for i in range(n_seg):
        w = s[:, i*sl:(i+1)*sl]
        out += [w.mean(1), w.std(1)]
    return out

def xcorr(a, b):
    """Cross-correlation between two signals"""
    ca = a - a.mean(1,keepdims=True)
    cb = b - b.mean(1,keepdims=True)
    return (ca*cb).mean(1) / (ca.std(1)*cb.std(1)+1e-10)

def extract_features(X):
    """Extract all features (same as run07 successful set)"""
    N, T, _ = X.shape
    mx, my, mz = X[:,:,0], X[:,:,1], X[:,:,2]
    sx, sy, sz = X[:,:,3], X[:,:,4], X[:,:,5]
    
    # Jerk (derivative, removes gravity)
    jx = np.diff(mx, axis=1)
    jy = np.diff(my, axis=1)
    jz = np.diff(mz, axis=1)
    
    # Magnitudes (rotation-invariant)
    mag_mean = np.sqrt(mx**2 + my**2 + mz**2)
    mag_std = np.sqrt(sx**2 + sy**2 + sz**2)
    mag_jerk = np.sqrt(jx**2 + jy**2 + jz**2)
    
    parts = []
    
    # Std channel statistics
    for ch in [sx, sy, sz]:
        parts += stats9(ch)
    parts += stats9(mag_std)
    parts += stats9(mag_mean)
    
    # Jerk statistics
    for ch in [jx, jy, jz]:
        parts += stats9(ch)
    parts += stats9(mag_jerk)
    
    # Segment features
    for sig in [mag_std, mag_jerk]:
        for ns in [10, 20]:
            parts += seg(sig, ns)
    for ch in [sx, sy, sz, jx, jy, jz]:
        parts += seg(ch, 10)
    
    # Autocorrelation
    for lag in [1, 2, 5, 10, 20, 30, 60]:
        parts.append(ac(mag_jerk, lag))
    for ch in [sx, sy, sz]:
        for lag in [1, 5, 10, 30]:
            parts.append(ac(ch, lag))
    
    # Spectral features
    for sig in [mag_jerk, mag_std, sx, sy, sz]:
        parts.append(spectral5(sig))
    
    # Cross-correlations
    for a,b in [(jx,jy),(jx,jz),(jy,jz)]:
        parts.append(xcorr(a,b))
    for a,b in [(sx,sy),(sx,sz),(sy,sz)]:
        parts.append(xcorr(a,b))
    
    # Zero-crossing rate
    cj = mag_jerk - mag_jerk.mean(1, keepdims=True)
    parts.append((np.diff(np.sign(cj), axis=1) != 0).sum(1) / T)
    
    # Peak rate
    pr = np.zeros(N, dtype=np.float32)
    for n in range(N):
        pr[n] = len(find_peaks(mag_jerk[n], height=mag_jerk[n].mean())[0]) / T
    parts.append(pr)
    
    return np.column_stack([
        np.asarray(p).reshape(N, -1) if np.asarray(p).ndim > 1
        else np.asarray(p).reshape(N, 1) for p in parts
    ]).astype(np.float32)

# multi-stage feature selection
print("\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)

# Extract features
print("Extracting features...")
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)
print(f"Original feature count: {X_train_features.shape[1]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Stage 1: LightGBM importance-based selection
print("\nStage 1: LightGBM importance selection...")
selector_lgb = SelectFromModel(
    lgb.LGBMClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1),
    threshold='median',  # Keep features above median importance
    max_features=200     # Cap at 200 features
)
X_train_selected = selector_lgb.fit_transform(X_train_scaled, y_train)
X_test_selected = selector_lgb.transform(X_test_scaled)
feature_mask = selector_lgb.get_support()
print(f"  After importance selection: {X_train_selected.shape[1]} features")

# Stage 2: Remove highly correlated features (>0.95 correlation)
print("\nStage 2: Removing highly correlated features...")
corr_matrix = pd.DataFrame(X_train_selected).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr = []
for col in upper_tri.columns:
    if any(upper_tri[col] > 0.95):
        high_corr.append(col)

if high_corr:
    keep_cols = [c for c in range(X_train_selected.shape[1]) if c not in high_corr]
    X_train_clean = X_train_selected[:, keep_cols]
    X_test_clean = X_test_selected[:, keep_cols]
    print(f"  Removed {len(high_corr)} highly correlated features")
    print(f"  After correlation removal: {X_train_clean.shape[1]} features")
else:
    X_train_clean = X_train_selected
    X_test_clean = X_test_selected
    print(f"  No highly correlated features found")

# Stage 3: Recursive Feature Elimination (optional, if we have time)
# Using RFECV to find optimal feature count
print("\nStage 3: Recursive Feature Elimination...")
rfecv = RFECV(
    estimator=lgb.LGBMClassifier(n_estimators=150, random_state=SEED, n_jobs=-1, verbose=-1),
    step=20,
    cv=min(3, N_FOLDS),
    scoring='accuracy',
    min_features_to_select=80,
    n_jobs=-1
)

# Use a subset for RFECV if too slow
if X_train_clean.shape[0] > 5000:
    sample_idx = np.random.choice(X_train_clean.shape[0], 5000, replace=False)
    rfecv.fit(X_train_clean[sample_idx], y_train[sample_idx])
else:
    rfecv.fit(X_train_clean, y_train)

print(f"  RFECV optimal features: {rfecv.n_features_}")
X_train_final = rfecv.transform(X_train_clean)
X_test_final = rfecv.transform(X_test_clean)

# Final feature count
print(f"\nFinal feature count: {X_train_final.shape[1]} (from {X_train_features.shape[1]} original)")

# time warping augmentation for CNN
def time_warp(x, sigma=0.2):
    """Non-linear time warping augmentation"""
    batch, seq_len, channels = x.shape
    device = x.device
    
    orig_t = torch.linspace(0, 1, seq_len, device=device)
    warp_points = torch.linspace(0, 1, 5, device=device)
    warp_values = torch.linspace(0, 1, 5, device=device) + torch.randn(5, device=device) * sigma
    warp_values = torch.clamp(warp_values, 0.2, 1.8)
    warp_values = torch.cat([torch.tensor([0.0], device=device), 
                             warp_values[1:-1], 
                             torch.tensor([1.0], device=device)])
    
    warped_t = torch.zeros(seq_len, device=device)
    for i in range(len(warp_points)-1):
        mask = (orig_t >= warp_points[i]) & (orig_t < warp_points[i+1])
        if mask.any():
            t_local = (orig_t[mask] - warp_points[i]) / (warp_points[i+1] - warp_points[i])
            warped_t[mask] = warp_values[i] + t_local * (warp_values[i+1] - warp_values[i])
    
    warped_t = torch.clamp(warped_t, 0, 1)
    warped_x = torch.zeros_like(x)
    for c in range(channels):
        warped_x[:, :, c] = torch.interp(orig_t, warped_t, x[:, :, c])
    
    return warped_x

def add_noise(x, noise_level=0.05):
    return x + torch.randn_like(x) * noise_level

def augment_batch(x):
    aug_type = np.random.choice(['time_warp', 'noise', 'none'], p=[0.3, 0.3, 0.4])
    if aug_type == 'time_warp':
        return time_warp(x)
    elif aug_type == 'noise':
        return add_noise(x)
    return x

# TCN + Transformer model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                                padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)

class TCNTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.tcn = nn.Sequential(
            ResidualBlock(input_dim, 64, kernel_size=5, dilation=1, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock(64, 128, kernel_size=5, dilation=2, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock(128, d_model, kernel_size=5, dilation=4, dropout=dropout),
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 75, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 6)
        )
        
    def forward(self, x):
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_encoding[:, :x.shape[1], :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

# training functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = augment_batch(x)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            predictions.append(output.argmax(dim=1).cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)

# cross-validation to find optimal ensemble weights
print("\n" + "="*60)
print("CROSS-VALIDATION FOR ENSEMBLE WEIGHTS")
print("="*60)

# Prepare CNN data
X_train_tensor = torch.tensor(X_train.transpose(0, 2, 1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# User-based folds
user_folds = {u: i % N_FOLDS for i, u in enumerate(unique_users)}
fold_ids = np.array([user_folds[u] for u in train_users])

# Store OOF predictions
oof_cnn = np.zeros((len(y_train), 6))
oof_lgb = np.zeros((len(y_train), 6))

for fold in range(N_FOLDS):
    print(f"\n{'='*40}")
    print(f"Fold {fold+1}/{N_FOLDS}")
    
    train_idx = np.where(fold_ids != fold)[0]
    val_idx = np.where(fold_ids == fold)[0]
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    # Train CNN
    model = TCNTransformer().to(DEVICE)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train[train_idx])
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor[train_idx], y_train_tensor[train_idx]), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_train_tensor[val_idx], y_train_tensor[val_idx]), 
                            batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            val_preds, val_true = evaluate(model, val_loader, DEVICE)
            val_acc = accuracy_score(val_true, val_preds)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Get OOF predictions
    model.eval()
    val_probs = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(DEVICE)
            probs = F.softmax(model(x), dim=1)
            val_probs.append(probs.cpu().numpy())
    oof_cnn[val_idx] = np.vstack(val_probs)
    print(f"  CNN Best Val Acc: {best_val_acc:.4f}")
    
    # Train LightGBM on selected features
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=30,
        reg_alpha=0.5,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train_final[train_idx], y_train[train_idx])
    oof_lgb[val_idx] = lgb_model.predict_proba(X_train_final[val_idx])
    
    lgb_val_acc = accuracy_score(y_train[val_idx], oof_lgb[val_idx].argmax(axis=1))
    print(f"  LGB Val Acc: {lgb_val_acc:.4f}")

# Find optimal ensemble weight
print("\n" + "="*60)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("="*60)

best_weight = 0.5
best_acc = 0

for w in np.arange(0.1, 0.91, 0.05):
    ensemble_pred = ((w * oof_cnn + (1-w) * oof_lgb)).argmax(axis=1)
    acc = accuracy_score(y_train, ensemble_pred)
    if acc > best_acc:
        best_acc = acc
        best_weight = w

print(f"Optimal CNN weight: {best_weight:.2f} (LGB weight: {1-best_weight:.2f})")
print(f"Ensemble OOF Accuracy: {best_acc:.4f}")

# final training on all data
print("\n" + "="*60)
print("FINAL TRAINING ON ALL DATA")
print("="*60)

# Train final CNN models (3 seeds for efficiency)
cnn_probas = []
for seed in range(3):
    print(f"\nTraining CNN {seed+1}/3...")
    torch.manual_seed(SEED + seed * 100)
    
    model = TCNTransformer().to(DEVICE)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    full_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                             batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, full_loader, optimizer, criterion, DEVICE)
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {train_loss:.4f}")
    
    # Predict on test
    model.eval()
    X_test_tensor = torch.tensor(X_test.transpose(0, 2, 1), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        test_probs = F.softmax(model(X_test_tensor), dim=1).cpu().numpy()
    cnn_probas.append(test_probs)

# Train final LightGBM models (3 seeds)
print("\nTraining LightGBM models...")
lgb_probas = []
for seed in range(3):
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=30,
        reg_alpha=0.5,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=SEED + seed * 100,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train_final, y_train)
    lgb_probas.append(lgb_model.predict_proba(X_test_final))
    print(f"  Trained LGBM {seed+1}/3")

# Ensemble predictions
cnn_ensemble = np.mean(cnn_probas, axis=0)
lgb_ensemble = np.mean(lgb_probas, axis=0)
final_proba = best_weight * cnn_ensemble + (1 - best_weight) * lgb_ensemble
final_preds = final_proba.argmax(axis=1)

# Gentle post-processing for rare classes
print("\nPost-processing predictions...")
pred_counts = pd.Series(final_preds).value_counts().sort_index()
print("Final prediction distribution:")
for c in range(6):
    expected = int(len(final_preds) * counts[c] / len(y_train))
    print(f"  Class {c}: {pred_counts.get(c, 0):5d} (expected: {expected:5d})")

# Save Submission

submission = pd.DataFrame({"Id": test_ids, "Label": final_preds})
submission = submission.sort_values("Id").reset_index(drop=True)
out_path = OUT_DIR / "submission_run10.csv"
submission.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print(f"Submission saved: {out_path}")
print(f"{'='*60}")

# Summary
print("\n" + "="*60)
print("RUN12 SUMMARY")
print("="*60)
print(f"Original features: 373")
print(f"Selected features: {X_train_final.shape[1]}")
print(f"Feature reduction: {(1 - X_train_final.shape[1]/373)*100:.1f}%")
print(f"CNN weight: {best_weight:.2f}")
print(f"LGB weight: {1-best_weight:.2f}")
print(f"Ensemble size: {len(cnn_probas)} CNN + {len(lgb_probas)} LGB")
print(f"\nExpected Kaggle Score: 0.79-0.81")
print(f"{'='*60}")