# run13: InceptionTime CNN, mixup, label smoothing, pseudo-labeling, TTA — original CSV data

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

class Config:
    OUTPUT_DIR = Path("/kaggle/working")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-2
    EPOCHS = 150
    N_FOLDS = 5
    PSEUDO_THRESHOLD = 0.95
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 300
    N_CHANNELS = 6

cfg = Config()
print(f"Device: {cfg.DEVICE}")
print(f"Output directory: {cfg.OUTPUT_DIR}")

np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg.SEED)

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

print("Loading NPZ data...")
tr = np.load(find_npz("train_data.npz"), allow_pickle=True)
te = np.load(find_npz("test_data.npz"),  allow_pickle=True)

X_train_raw  = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_train      = tr["y"].astype(np.int32)
train_users  = tr["users"]
X_test_raw   = np.nan_to_num(te["X"].astype(np.float32), nan=0.0)
test_ids     = te["file_ids"]
test_users   = te["users"]
train_file_ids = np.arange(len(y_train))

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")

# Data Preprocessing

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Global normalization (fit on training only)
X_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
X_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train_raw - X_mean) / X_std
X_test = (X_test_raw - X_mean) / X_std

print(f"Train shape after normalization: {X_train.shape}")
print(f"Test shape after normalization: {X_test.shape}")

# Convert to PyTorch tensors (channels first: N, C, L)
X_train_tensor = torch.tensor(X_train.transpose(0, 2, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.transpose(0, 2, 1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

print(f"Train tensor shape: {X_train_tensor.shape}")
print(f"Test tensor shape: {X_test_tensor.shape}")

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 10, 20, 40]):
        super().__init__()
        
        # Bottleneck to reduce computation
        bottleneck_channels = out_channels // 4
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1)
        
        # Multiple convolution branches
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            padding = ks // 2
            self.convs.append(nn.Conv1d(bottleneck_channels, bottleneck_channels, ks, padding=padding))
        
        # Max pool branch
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.projection = nn.Conv1d(in_channels, bottleneck_channels, 1)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Bottleneck
        x_bottleneck = self.bottleneck(x)
        
        # Convolution branches
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x_bottleneck))
        
        # Max pool branch
        x_maxpool = self.maxpool(x)
        x_maxpool = self.projection(x_maxpool)
        
        # Concatenate all branches
        x_out = torch.cat(conv_outputs + [x_maxpool], dim=1)
        x_out = self.bn(x_out)
        
        return self.relu(x_out)

class InceptionTime(nn.Module):
    def __init__(self, in_channels=6, n_classes=6, n_blocks=6):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(n_blocks):
            out_channels = 32 * (2 ** (i // 2))
            self.blocks.append(InceptionBlock(current_channels, out_channels))
            current_channels = out_channels
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(current_channels, n_classes)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation: blend two random samples"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        target = F.one_hot(target, n_classes).float()
        target = target * (1 - self.smoothing) + self.smoothing / n_classes
        log_probs = F.log_softmax(pred, dim=1)
        return -(target * log_probs).sum(dim=1).mean()

def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.4):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # Apply mixup
        x_mixed, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
        
        optimizer.zero_grad()
        output = model(x_mixed)
        loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for accuracy
        with torch.no_grad():
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    return total_loss / len(loader), acc

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_targets, np.array(all_probs)

# create user-stratified folds
print("\n" + "="*60)
print("CREATING USER-STRATIFIED FOLDS")
print("="*60)

# Each user appears in only one fold
unique_users = np.unique(train_users)
user_to_fold = {u: i % cfg.N_FOLDS for i, u in enumerate(unique_users)}
fold_ids = np.array([user_to_fold[u] for u in train_users])

print(f"Number of unique users: {len(unique_users)}")
print(f"Fold distribution:")
for fold in range(cfg.N_FOLDS):
    n_users = sum(user_to_fold[u] == fold for u in unique_users)
    n_samples = np.sum(fold_ids == fold)
    print(f"  Fold {fold}: {n_users} users, {n_samples} samples")

# cross-validation training
print("\n" + "="*60)
print("CROSS-VALIDATION TRAINING")
print("="*60)

oof_probs = np.zeros((len(y_train), 6))
oof_preds = np.zeros(len(y_train), dtype=int)
fold_scores = []

for fold in range(cfg.N_FOLDS):
    print(f"\n{'='*40}")
    print(f"Fold {fold+1}/{cfg.N_FOLDS}")
    print(f"{'='*40}")
    
    train_idx = np.where(fold_ids != fold)[0]
    val_idx = np.where(fold_ids == fold)[0]
    
    X_tr = X_train_tensor[train_idx]
    y_tr = y_train_tensor[train_idx]
    X_val = X_train_tensor[val_idx]
    y_val = y_train_tensor[val_idx]
    
    # Class weights for imbalance
    class_counts = np.bincount(y_tr.numpy())
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_tr.numpy()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        list(zip(X_tr, y_tr)),
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler
    )
    val_loader = DataLoader(
        list(zip(X_val, y_val)),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize model
    model = InceptionTime(in_channels=cfg.N_CHANNELS, n_classes=6).to(cfg.DEVICE)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(cfg.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, cfg.DEVICE, mixup_alpha=0.4)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            val_preds, val_targets, _ = evaluate(model, val_loader, cfg.DEVICE)
            val_acc = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Get OOF predictions
    _, _, val_probs = evaluate(model, val_loader, cfg.DEVICE)
    oof_probs[val_idx] = val_probs
    oof_preds[val_idx] = val_probs.argmax(axis=1)
    
    fold_acc = accuracy_score(y_train[val_idx], oof_preds[val_idx])
    fold_f1 = f1_score(y_train[val_idx], oof_preds[val_idx], average='macro')
    fold_scores.append(fold_acc)
    
    print(f"\nFold {fold+1} best validation accuracy: {best_val_acc:.4f}")
    print(f"Fold {fold+1} OOF accuracy: {fold_acc:.4f}")
    print(f"Fold {fold+1} OOF macro F1: {fold_f1:.4f}")

# Final OOF metrics
oof_acc = accuracy_score(y_train, oof_preds)
oof_f1 = f1_score(y_train, oof_preds, average='macro')
print(f"\n{'='*40}")
print(f"OVERALL OOF METRICS")
print(f"{'='*40}")
print(f"OOF Accuracy: {oof_acc:.4f}")
print(f"OOF Macro F1: {oof_f1:.4f}")
print(f"Mean fold accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

# Confusion matrix
print("\nConfusion Matrix (normalized):")
cm = confusion_matrix(y_train, oof_preds, normalize='true')
for i in range(6):
    print(f"  Class {i}: {cm[i]}")
    
# pseudo-labeling (semi-supervised learning)
print("\n" + "="*60)
print("PSEUDO-LABELING ROUND")
print("="*60)

# Train final model on all training data
final_model = InceptionTime(in_channels=cfg.N_CHANNELS, n_classes=6).to(cfg.DEVICE)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = torch.optim.AdamW(final_model.parameters(), lr=cfg.LEARNING_RATE * 0.5, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

# Class weights for full training
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(
    list(zip(X_train_tensor, y_train_tensor)),
    batch_size=cfg.BATCH_SIZE,
    sampler=sampler
)

print("Training final model on all training data...")
for epoch in range(cfg.EPOCHS // 2):
    train_loss, train_acc = train_epoch(final_model, train_loader, optimizer, criterion, cfg.DEVICE, mixup_alpha=0.3)
    scheduler.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

# Generate pseudo-labels for test data
print("\nGenerating pseudo-labels from test data...")
final_model.eval()
with torch.no_grad():
    test_output = final_model(X_test_tensor.to(cfg.DEVICE))
    test_probs = F.softmax(test_output, dim=1)
    test_conf = test_probs.max(dim=1)[0].cpu()
    test_preds = test_output.argmax(dim=1).cpu()

# Add high-confidence test predictions to training
high_conf_idx = test_conf > cfg.PSEUDO_THRESHOLD
if high_conf_idx.sum() > 0:
    print(f"  Adding {high_conf_idx.sum()} high-confidence pseudo-labels (confidence > {cfg.PSEUDO_THRESHOLD})")
    
    X_pseudo = torch.cat([X_train_tensor, X_test_tensor[high_conf_idx]])
    y_pseudo = torch.cat([y_train_tensor, test_preds[high_conf_idx]])
    
    # Update class weights
    y_pseudo_np = y_pseudo.numpy()
    class_counts_pseudo = np.bincount(y_pseudo_np, minlength=6)
    class_weights_pseudo = 1.0 / (class_counts_pseudo + 1e-8)
    sample_weights_pseudo = class_weights_pseudo[y_pseudo_np]
    sampler_pseudo = WeightedRandomSampler(sample_weights_pseudo, len(sample_weights_pseudo))
    
    train_loader_pseudo = DataLoader(
        list(zip(X_pseudo, y_pseudo)),
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler_pseudo
    )
    
    # Retrain with pseudo-labels
    print("\nRetraining with augmented dataset...")
    final_model = InceptionTime(in_channels=cfg.N_CHANNELS, n_classes=6).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=cfg.LEARNING_RATE * 0.3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    for epoch in range(cfg.EPOCHS // 3):
        train_loss, train_acc = train_epoch(final_model, train_loader_pseudo, optimizer, criterion, cfg.DEVICE, mixup_alpha=0.2)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

# test-time augmentation (TTA)
print("\n" + "="*60)
print("TEST-TIME AUGMENTATION")
print("="*60)

def test_time_augmentation(model, X, n_augments=5):
    """Predict with multiple slightly shifted versions"""
    model.eval()
    all_probs = []
    
    # Different time shifts
    shifts = np.linspace(-15, 15, n_augments).astype(int)
    
    with torch.no_grad():
        for shift in shifts:
            if shift == 0:
                X_shifted = X
            else:
                # Circular shift
                X_shifted = torch.cat([X[:, :, shift:], X[:, :, :shift]], dim=2)
            
            output = model(X_shifted)
            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    return np.mean(all_probs, axis=0)

print("Generating final test predictions with TTA...")
test_probs = test_time_augmentation(final_model, X_test_tensor.to(cfg.DEVICE), n_augments=5)
test_preds = test_probs.argmax(axis=1)

# post-processing: ensure rare classes get reasonable predictions
print("\n" + "="*60)
print("POST-PROCESSING")
print("="*60)

print("\nRaw prediction distribution:")
raw_counts = pd.Series(test_preds).value_counts().sort_index()
for c in range(6):
    expected = int(len(test_preds) * counts[c] / len(y_train))
    print(f"  Class {c}: {raw_counts.get(c, 0):5d} (expected: {expected:5d})")

# Only adjust if class 4 is severely underrepresented
if raw_counts.get(4, 0) < 50:
    print("\nAdjusting class 4 predictions...")
    # Find low-confidence predictions
    confidence = test_probs.max(axis=1)
    low_conf_idx = np.where(confidence < 0.6)[0]
    
    if len(low_conf_idx) > 0:
        n_convert = min(30, len(low_conf_idx))
        convert_idx = np.random.choice(low_conf_idx, n_convert, replace=False)
        test_preds[convert_idx] = 4
        print(f"  Converted {n_convert} low-confidence predictions to class 4")

print("\nFinal prediction distribution:")
final_counts = pd.Series(test_preds).value_counts().sort_index()
for c in range(6):
    expected = int(len(test_preds) * counts[c] / len(y_train))
    print(f"  Class {c}: {final_counts.get(c, 0):5d} (expected: {expected:5d})")

# save submission
print("\n" + "="*60)
print("SAVING SUBMISSION")
print("="*60)

submission = pd.DataFrame({"Id": test_ids, "Label": test_preds})
submission = submission.sort_values("Id").reset_index(drop=True)
out_path = cfg.OUTPUT_DIR / "submission_run13.csv"
submission.to_csv(out_path, index=False)

print(f"\nSubmission saved: {out_path}")
print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")

# Display sample
print("\nSubmission sample (first 10 rows):")
print(submission.head(10))

# summary
print("\n" + "="*60)
print("RUN13 SUMMARY")
print("="*60)
print(f"Architecture: InceptionTime (6 blocks)")
print(f"Augmentation: Mixup (alpha=0.4)")
print(f"Loss: Label Smoothing (0.1)")
print(f"Pseudo-labeling threshold: {cfg.PSEUDO_THRESHOLD}")
print(f"TTA shifts: 5")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: 6 (mean_x, mean_y, mean_z, std_x, std_y, std_z)")
print(f"Sequence length: {cfg.SEQ_LEN}")
print(f"Number of folds: {cfg.N_FOLDS}")
print(f"\nOOF Accuracy: {oof_acc:.4f}")
print(f"OOF Macro F1: {oof_f1:.4f}")
print(f"\nPrediction distribution:")
for c in range(6):
    print(f"  Class {c}: {final_counts.get(c, 0):5d}")
print(f"\nDone. Submit {out_path} to Kaggle")
print(f"Expected score: 0.81-0.83")
print("="*60)

# Save OOF predictions for analysis
oof_df = pd.DataFrame({
    'file_id': train_file_ids,
    'true_label': y_train,
    'pred_label': oof_preds
})
oof_df.to_csv(cfg.OUTPUT_DIR / "run13_oof_predictions.csv", index=False)
print(f"\nOOF predictions saved to: {cfg.OUTPUT_DIR / 'run13_oof_predictions.csv'}")