"""
Assignment 3 – Step 4: Ablation Study
Addresses grading question 4 (10%):
  "Please provide an ablation study for your core technical design choices."

Ablation dimensions:
  A. Feature group ablation:     which feature group contributes most?
  B. Model ablation:             RF vs. XGBoost vs. Logistic Regression
  C. Number of segments:         how many temporal segments are optimal?
  D. Clipping threshold:         does the outlier threshold matter?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import google.colab; IN_COLAB = True
except ImportError:
    IN_COLAB = False

BASE_DIR  = Path("/content/DataMining_Assignment3") if IN_COLAB else Path(__file__).parent
TRAIN_DIR = BASE_DIR / "train" / "train"
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEAT_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def load_sequences():
    sequences, labels = [], []
    for user_dir in sorted(TRAIN_DIR.iterdir()):
        for csv_path in sorted(user_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            sequences.append(df[FEAT_COLS].values.astype(np.float32))
            labels.append(int(df["label"].iloc[0]))
    X = np.array(sequences)
    X = np.clip(np.nan_to_num(X, nan=0.0), -10, 10)
    return X, np.array(labels)


def make_pipeline(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run_cv(X_feat, y, clf=None):
    if clf is None:
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = make_pipeline(clf)
    scores = cross_val_score(pipe, X_feat, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean(), scores.std()


# ── Feature group builders ─────────────────────────────────────────────────────

def feat_global_stats(X):
    """Group A: global mean, std, min, max, range, median, IQR, skew, kurt (54)."""
    parts = []
    for c in range(X.shape[2]):
        s = X[:, :, c]
        parts += [
            s.mean(axis=1), s.std(axis=1), s.min(axis=1), s.max(axis=1),
            s.max(axis=1) - s.min(axis=1),
            np.median(s, axis=1),
            np.percentile(s, 75, axis=1) - np.percentile(s, 25, axis=1),
            np.array([skew(r) for r in s]),
            np.array([kurtosis(r) for r in s]),
        ]
    return np.column_stack(parts)


def feat_magnitude(X):
    """Group B: vector magnitude stats (3)."""
    mag = np.sqrt((X[:, :, :3] ** 2).sum(axis=2))
    return np.column_stack([mag.mean(axis=1), mag.std(axis=1),
                            mag.max(axis=1) - mag.min(axis=1)])


def feat_segments(X, n_segments=10):
    """Group C: temporal segment mean/std per channel (120)."""
    N, T, C = X.shape
    seg_len = T // n_segments
    parts = []
    for i in range(n_segments):
        seg = X[:, i * seg_len:(i + 1) * seg_len, :]
        parts += [seg.mean(axis=1), seg.std(axis=1)]
    return np.hstack(parts)


def feat_autocorr(X, lags=(1, 5, 10, 30)):
    """Group D: autocorrelation at selected lags (24)."""
    N, T, C = X.shape
    parts = []
    for lag in lags:
        acorr = np.zeros((N, C), dtype=np.float32)
        for c in range(C):
            s = X[:, :, c]
            s1, s2 = s[:, :-lag], s[:, lag:]
            mu1 = s1.mean(axis=1, keepdims=True)
            mu2 = s2.mean(axis=1, keepdims=True)
            num = ((s1 - mu1) * (s2 - mu2)).mean(axis=1)
            den = s1.std(axis=1) * s2.std(axis=1) + 1e-10
            acorr[:, c] = num / den
        parts.append(acorr)
    return np.hstack(parts)


def feat_trend(X):
    """Group E: linear trend slope per channel (6)."""
    N, T, C = X.shape
    t = np.arange(T, dtype=np.float32) - T / 2
    slopes = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        slopes[:, c] = (X[:, :, c] * t).sum(axis=1) / (t ** 2).sum()
    return slopes


def feat_spectral(X):
    """Group F: spectral features (30)."""
    N, T, C = X.shape
    out = np.zeros((N, 5 * C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            signal = X[n, :, c] - X[n, :, c].mean()
            freqs, psd = welch(signal, fs=1.0, nperseg=min(64, T))
            psd_norm = psd / (psd.sum() + 1e-10)
            out[n, c * 5:(c + 1) * 5] = [
                freqs[np.argmax(psd)],
                -np.sum(psd_norm * np.log(psd_norm + 1e-10)),
                psd[(freqs >= 0.0) & (freqs < 0.5)].sum(),
                psd[(freqs >= 0.5) & (freqs < 2.0)].sum(),
                psd[(freqs >= 2.0)].sum(),
            ]
    return out


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
X_raw, y = load_sequences()
print(f"  {X_raw.shape}\n")

all_results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION A – Feature group contribution (leave-one-out style)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ABLATION A – Feature Group Contribution")
print("=" * 65)

FULL = np.hstack([
    feat_global_stats(X_raw),
    feat_magnitude(X_raw),
    feat_segments(X_raw),
    feat_autocorr(X_raw),
    feat_trend(X_raw),
    feat_spectral(X_raw),
])

groups = {
    "Global stats":       feat_global_stats(X_raw),
    "Magnitude":          feat_magnitude(X_raw),
    "Temporal segments":  feat_segments(X_raw),
    "Autocorrelation":    feat_autocorr(X_raw),
    "Trend (slope)":      feat_trend(X_raw),
    "Spectral":           feat_spectral(X_raw),
}

ablation_a = {}
# Full model
m, s = run_cv(FULL, y)
print(f"  {'ALL features':<30s}  {X_raw.shape[0]} windows  acc = {m:.4f} ± {s:.4f}")
ablation_a["All features"] = (m, s)

# Leave-one-group-out
for name, feat_matrix in groups.items():
    others = [v for k, v in groups.items() if k != name]
    X_without = np.hstack(others)
    m, s = run_cv(X_without, y)
    drop = ablation_a["All features"][0] - m
    ablation_a[f"w/o {name}"] = (m, s)
    print(f"  {'w/o ' + name:<30s}  Δacc = {-drop:+.4f}  acc = {m:.4f} ± {s:.4f}")

# Only one group
print()
for name, feat_matrix in groups.items():
    m, s = run_cv(feat_matrix, y)
    ablation_a[f"only {name}"] = (m, s)
    print(f"  {'only ' + name:<30s}                acc = {m:.4f} ± {s:.4f}")

all_results["A_feature_groups"] = ablation_a

# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION B – Model choice
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ABLATION B – Model Architecture")
print("=" * 65)

models_to_test = {
    "Logistic Regression":     LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest (50 trees)": RandomForestClassifier(n_estimators=50,  random_state=42, n_jobs=-1),
    "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Random Forest (500 trees)": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
}
if HAS_XGB:
    models_to_test["XGBoost (lr=0.1)"] = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    models_to_test["XGBoost (lr=0.05)"] = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )

ablation_b = {}
for name, clf in models_to_test.items():
    m, s = run_cv(FULL, y, clf)
    ablation_b[name] = (m, s)
    print(f"  {name:<35s}  acc = {m:.4f} ± {s:.4f}")

all_results["B_model_choice"] = ablation_b

# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION C – Number of temporal segments
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ABLATION C – Number of Temporal Segments")
print("=" * 65)

base_feats = np.hstack([
    feat_global_stats(X_raw),
    feat_magnitude(X_raw),
    feat_autocorr(X_raw),
    feat_trend(X_raw),
    feat_spectral(X_raw),
])

ablation_c = {}
for n_seg in [1, 3, 5, 10, 15, 20, 30]:
    seg_f = feat_segments(X_raw, n_segments=n_seg)
    X_combined = np.hstack([base_feats, seg_f])
    m, s = run_cv(X_combined, y)
    ablation_c[n_seg] = (m, s)
    print(f"  n_segments = {n_seg:2d}  ({seg_f.shape[1]:3d} seg features)  acc = {m:.4f} ± {s:.4f}")

all_results["C_n_segments"] = ablation_c

# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION D – Clipping threshold
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ABLATION D – Outlier Clipping Threshold")
print("=" * 65)

ablation_d = {}
for threshold in [None, 1, 3, 5, 10, 20]:
    if threshold is None:
        X_c = X_raw.copy()
        label = "no clipping"
    else:
        X_c = np.clip(X_raw, -threshold, threshold)
        label = f"clip = ±{threshold}"
    X_f = np.hstack([
        feat_global_stats(X_c), feat_magnitude(X_c),
        feat_segments(X_c), feat_autocorr(X_c),
        feat_trend(X_c), feat_spectral(X_c),
    ])
    m, s = run_cv(X_f, y)
    ablation_d[label] = (m, s)
    print(f"  {label:<15s}  acc = {m:.4f} ± {s:.4f}")

all_results["D_clipping"] = ablation_d

# ── Visualize ablation results ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A – feature groups (leave-one-out)
ax = axes[0, 0]
loo_keys = [k for k in ablation_a if k.startswith("w/o") or k == "All features"]
loo_means = [ablation_a[k][0] for k in loo_keys]
loo_stds  = [ablation_a[k][1] for k in loo_keys]
colors = ["steelblue"] + ["salmon"] * (len(loo_keys) - 1)
ax.barh(loo_keys, loo_means, xerr=loo_stds, color=colors, capsize=3)
ax.axvline(ablation_a["All features"][0], color="steelblue", linestyle="--", lw=1)
ax.set_xlabel("CV Accuracy")
ax.set_title("A: Feature Group Leave-One-Out")

# B – model choice
ax = axes[0, 1]
b_keys  = list(ablation_b.keys())
b_means = [ablation_b[k][0] for k in b_keys]
b_stds  = [ablation_b[k][1] for k in b_keys]
ax.barh(b_keys, b_means, xerr=b_stds, color="mediumpurple", capsize=3)
ax.set_xlabel("CV Accuracy")
ax.set_title("B: Model Architecture")

# C – n_segments
ax = axes[1, 0]
c_ns    = list(ablation_c.keys())
c_means = [ablation_c[k][0] for k in c_ns]
c_stds  = [ablation_c[k][1] for k in c_ns]
ax.errorbar(c_ns, c_means, yerr=c_stds, marker="o", color="seagreen", capsize=3)
ax.set_xlabel("Number of segments")
ax.set_ylabel("CV Accuracy")
ax.set_title("C: Number of Temporal Segments")
ax.set_xticks(c_ns)

# D – clipping
ax = axes[1, 1]
d_keys  = list(ablation_d.keys())
d_means = [ablation_d[k][0] for k in d_keys]
d_stds  = [ablation_d[k][1] for k in d_keys]
ax.bar(d_keys, d_means, yerr=d_stds, color="darkorange", capsize=3)
ax.set_xlabel("Clipping threshold")
ax.set_ylabel("CV Accuracy")
ax.set_title("D: Outlier Clipping Threshold")
ax.set_xticklabels(d_keys, rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_DIR / "04_ablation_study.png", dpi=150)
plt.close()

# ── Save results ───────────────────────────────────────────────────────────────
rows = []
for section, res in all_results.items():
    for name, (m, s) in res.items():
        rows.append({"section": section, "config": name, "mean_acc": m, "std_acc": s})
pd.DataFrame(rows).to_csv(OUT_DIR / "04_ablation_results.csv", index=False)

print(f"\nAblation study complete. Outputs saved to: {OUT_DIR}")
