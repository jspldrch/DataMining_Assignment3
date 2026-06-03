"""
Assignment 3 – Improved Model (script 06)
Improvements over baseline (0.7366):
  1. Richer features: cross-channel correlations, peak stats, zero-crossing rate
  2. Balanced class weights to fix minority class underperformance
  3. LightGBM (faster and usually more accurate than RF on tabular data)
  4. Soft-voting ensemble: LightGBM + XGBoost + Random Forest
Outputs submission.csv directly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not found. Install: pip install lightgbm")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found. Install: pip install xgboost")

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

FEAT_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(X_3d: np.ndarray) -> np.ndarray:
    """
    X_3d: (N, 300, 6)
    Returns: (N, D) feature matrix
    """
    N, T, C = X_3d.shape
    parts = []

    # ── A. Global statistics (9 × 6 = 54) ────────────────────────────────────
    for c in range(C):
        s = X_3d[:, :, c]
        parts += [
            s.mean(axis=1), s.std(axis=1),
            s.min(axis=1),  s.max(axis=1),
            s.max(axis=1) - s.min(axis=1),
            np.median(s, axis=1),
            np.percentile(s, 75, axis=1) - np.percentile(s, 25, axis=1),
            np.array([skew(r)     for r in s]),
            np.array([kurtosis(r) for r in s]),
        ]

    # ── B. Vector magnitude (3) ───────────────────────────────────────────────
    mag = np.sqrt((X_3d[:, :, :3] ** 2).sum(axis=2))
    parts += [mag.mean(axis=1), mag.std(axis=1), mag.max(axis=1) - mag.min(axis=1)]

    # ── C. Temporal segments — 20 segments of 15s (20×2×6 = 240) ─────────────
    n_seg = 20
    seg_len = T // n_seg
    for i in range(n_seg):
        seg = X_3d[:, i*seg_len:(i+1)*seg_len, :]
        parts += [seg.mean(axis=1), seg.std(axis=1)]

    # ── D. Autocorrelation at 9 lags (9×6 = 54) ──────────────────────────────
    for lag in [1, 2, 3, 5, 10, 15, 20, 30, 60]:
        ac = np.zeros((N, C), dtype=np.float32)
        for c in range(C):
            s = X_3d[:, :, c]
            s1, s2 = s[:, :-lag], s[:, lag:]
            num = ((s1 - s1.mean(axis=1, keepdims=True)) *
                   (s2 - s2.mean(axis=1, keepdims=True))).mean(axis=1)
            den = s1.std(axis=1) * s2.std(axis=1) + 1e-10
            ac[:, c] = num / den
        parts.append(ac)

    # ── E. Linear trend slope (6) ─────────────────────────────────────────────
    t = np.arange(T, dtype=np.float32) - T / 2
    slopes = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        slopes[:, c] = (X_3d[:, :, c] * t).sum(axis=1) / (t**2).sum()
    parts.append(slopes)

    # ── F. Cross-channel correlations between mean axes (3 pairs) ────────────
    # Captures coordinated motion across axes (e.g. x and y correlated during walking)
    axis_pairs = [(0, 1), (0, 2), (1, 2)]
    cross = np.zeros((N, len(axis_pairs)), dtype=np.float32)
    for i, (a, b) in enumerate(axis_pairs):
        sa = X_3d[:, :, a] - X_3d[:, :, a].mean(axis=1, keepdims=True)
        sb = X_3d[:, :, b] - X_3d[:, :, b].mean(axis=1, keepdims=True)
        num = (sa * sb).mean(axis=1)
        den = sa.std(axis=1) * sb.std(axis=1) + 1e-10
        cross[:, i] = num / den
    parts.append(cross)

    # ── G. Zero-crossing rate per channel (6) ────────────────────────────────
    # High for dynamic activities (running), low for stationary ones
    zcr = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        s = X_3d[:, :, c]
        centered = s - s.mean(axis=1, keepdims=True)
        zcr[:, c] = (np.diff(np.sign(centered), axis=1) != 0).sum(axis=1) / T
    parts.append(zcr)

    # ── H. Peak statistics for mean axes (3 axes × 3 stats = 9) ─────────────
    # Captures rhythmic motion patterns (peaks per second, height, spacing)
    peak_feats = np.zeros((N, 9), dtype=np.float32)
    for n in range(N):
        for ci, c in enumerate(range(3)):
            signal = X_3d[n, :, c]
            peaks, props = find_peaks(signal, height=signal.mean())
            peak_feats[n, ci*3]     = len(peaks) / T              # rate
            peak_feats[n, ci*3 + 1] = props["peak_heights"].mean() if len(peaks) > 0 else 0
            peak_feats[n, ci*3 + 2] = np.diff(peaks).mean() if len(peaks) > 1 else T
    parts.append(peak_feats)

    # ── I. Spectral features (5 × 6 = 30) ────────────────────────────────────
    spec = np.zeros((N, 5*C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            signal = X_3d[n, :, c] - X_3d[n, :, c].mean()
            freqs, psd = welch(signal, fs=1.0, nperseg=min(64, T))
            psd_norm = psd / (psd.sum() + 1e-10)
            spec[n, c*5:(c+1)*5] = [
                freqs[np.argmax(psd)],
                -np.sum(psd_norm * np.log(psd_norm + 1e-10)),
                psd[(freqs >= 0.0) & (freqs < 0.5)].sum(),
                psd[(freqs >= 0.5) & (freqs < 2.0)].sum(),
                psd[freqs >= 2.0].sum(),
            ]
    parts.append(spec)

    # ── Concatenate all ───────────────────────────────────────────────────────
    flat = []
    for p in parts:
        flat.append(p.reshape(N, -1))
    return np.hstack(flat).astype(np.float32)


def load_dataset(root_dir: Path):
    sequences, labels, file_ids = [], [], []
    for user_dir in sorted(root_dir.iterdir()):
        for csv_path in sorted(user_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            sequences.append(df[FEAT_COLS].values.astype(np.float32))
            if "label" in df.columns:
                labels.append(int(df["label"].iloc[0]))
            file_ids.append(int(df["file_id"].iloc[0]))
    X = np.array(sequences)
    X = np.clip(np.nan_to_num(X, nan=0.0), -10, 10)
    return X, np.array(labels) if labels else None, np.array(file_ids)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading training data …")
X_train_raw, y_train, train_ids = load_dataset(TRAIN_DIR)
print(f"  Train shape: {X_train_raw.shape}")

print("Extracting training features …")
X_train = extract_features(X_train_raw)
print(f"  Feature matrix: {X_train.shape}")

print("Loading test data …")
X_test_raw, _, test_ids = load_dataset(TEST_DIR)
print("Extracting test features …")
X_test = extract_features(X_test_raw)

# Class distribution + sample weights
unique, counts = np.unique(y_train, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} ({c/len(y_train)*100:.1f}%)")

sample_weights = compute_sample_weight("balanced", y_train)

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

# ── LightGBM ──────────────────────────────────────────────────────────────────
if HAS_LGB:
    print("\nTraining LightGBM …")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05,
        num_leaves=63, max_depth=-1,
        min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    scores = cross_val_score(lgb_clf, X_train_sc, y_train, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    cv_results["LightGBM"] = scores
    print(f"  LightGBM CV acc = {scores.mean():.4f} ± {scores.std():.4f}")
    lgb_clf.fit(X_train_sc, y_train, sample_weight=sample_weights)

# ── XGBoost ───────────────────────────────────────────────────────────────────
if HAS_XGB:
    print("Training XGBoost …")
    scale_pos = {i: len(y_train) / (6 * c) for i, c in zip(unique, counts)}
    xgb_clf = XGBClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    scores = cross_val_score(xgb_clf, X_train_sc, y_train, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    cv_results["XGBoost"] = scores
    print(f"  XGBoost  CV acc = {scores.mean():.4f} ± {scores.std():.4f}")
    xgb_clf.fit(X_train_sc, y_train, sample_weight=sample_weights)

# ── Random Forest ─────────────────────────────────────────────────────────────
print("Training Random Forest …")
rf_clf = RandomForestClassifier(
    n_estimators=500, class_weight="balanced",
    random_state=42, n_jobs=-1,
)
scores = cross_val_score(rf_clf, X_train_sc, y_train, cv=cv,
                         scoring="accuracy", n_jobs=-1)
cv_results["Random Forest"] = scores
print(f"  RF       CV acc = {scores.mean():.4f} ± {scores.std():.4f}")
rf_clf.fit(X_train_sc, y_train, sample_weight=sample_weights)

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE (soft voting = average predicted probabilities)
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding ensemble …")

trained_models = {}
if HAS_LGB: trained_models["LightGBM"] = lgb_clf
if HAS_XGB: trained_models["XGBoost"]  = xgb_clf
trained_models["Random Forest"] = rf_clf

# Average probabilities across all trained models
proba_train = np.mean([m.predict_proba(X_train_sc) for m in trained_models.values()], axis=0)
proba_test  = np.mean([m.predict_proba(X_test_sc)  for m in trained_models.values()], axis=0)

y_pred_train_ens = proba_train.argmax(axis=1)
y_pred_test_ens  = proba_test.argmax(axis=1)

print("\nEnsemble in-sample classification report:")
print(classification_report(y_train, y_pred_train_ens, digits=4))

# ── CV accuracy for ensemble (manual) ─────────────────────────────────────────
ens_cv_preds = np.zeros(len(y_train), dtype=int)
for tr_idx, val_idx in cv.split(X_train_sc, y_train):
    fold_probas = []
    for name, base_clf in [
        ("LightGBM", lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1) if HAS_LGB else None),
        ("XGBoost",  XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42, n_jobs=-1) if HAS_XGB else None),
        ("RF",       RandomForestClassifier(n_estimators=500, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ]:
        if base_clf is None:
            continue
        sw = sample_weights[tr_idx]
        base_clf.fit(X_train_sc[tr_idx], y_train[tr_idx], sample_weight=sw)
        fold_probas.append(base_clf.predict_proba(X_train_sc[val_idx]))
    ens_cv_preds[val_idx] = np.mean(fold_probas, axis=0).argmax(axis=1)

ens_cv_acc = (ens_cv_preds == y_train).mean()
print(f"Ensemble 5-fold CV accuracy: {ens_cv_acc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Submission CSV ────────────────────────────────────────────────────────────
submission = pd.DataFrame({"Id": test_ids, "Label": y_pred_test_ens})
submission = submission.sort_values("Id").reset_index(drop=True)
submission_path = OUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved: {submission_path}")
print(submission["Label"].value_counts().sort_index().to_string())

# ── CV results bar chart ──────────────────────────────────────────────────────
sns.set_style("whitegrid")
all_names  = list(cv_results.keys()) + ["Ensemble (CV)"]
all_means  = [cv_results[k].mean() for k in cv_results] + [ens_cv_acc]
all_stds   = [cv_results[k].std()  for k in cv_results] + [0]
colors     = sns.color_palette("Set2", len(all_names))

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(all_names, all_means, yerr=all_stds, capsize=5,
              color=colors, edgecolor="white")
for bar, m in zip(bars, all_means):
    ax.text(bar.get_x() + bar.get_width()/2, m + 0.005,
            f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(0.7366, color="red", linestyle="--", lw=1.5, label="Previous score (0.7366)")
ax.set_ylabel("5-fold CV Accuracy")
ax.set_title("Improved Model — CV Comparison")
ax.set_ylim(0.5, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "06_model_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 06_model_comparison.png")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm_norm = confusion_matrix(y_train, y_pred_train_ens, normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            xticklabels=range(6), yticklabels=range(6), linewidths=0.5)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Ensemble — Normalized Confusion Matrix (in-sample)")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 06_confusion_matrix.png")
