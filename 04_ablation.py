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
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"BASE_DIR : {BASE_DIR}")

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

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
full_acc = ablation_a["All features"][0]

# ── Plot A: Leave-one-out feature group ablation ───────────────────────────────
loo_keys  = ["All features"] + [k for k in ablation_a if k.startswith("w/o")]
loo_means = [ablation_a[k][0] for k in loo_keys]
loo_stds  = [ablation_a[k][1] for k in loo_keys]
loo_drops = [full_acc - m for m in loo_means]
bar_colors = ["#27ae60"] + ["#e74c3c" if d > 0.005 else "#f39c12" for d in loo_drops[1:]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(loo_keys[::-1], loo_means[::-1], xerr=loo_stds[::-1],
               color=bar_colors[::-1], capsize=4, edgecolor="white", height=0.6)
for bar, m, d in zip(bars, loo_means[::-1], loo_drops[::-1]):
    ax.text(m + 0.002, bar.get_y() + bar.get_height()/2,
            f"{m:.4f}  (Δ={-d:+.4f})" if d != 0 else f"{m:.4f}",
            va="center", fontsize=9)
ax.axvline(full_acc, color="#27ae60", linestyle="--", lw=1.5, label=f"Full model = {full_acc:.4f}")
ax.set_xlabel("5-fold CV Accuracy")
ax.set_title("Ablation A: Leave-One-Feature-Group-Out\n(negative Δ = that group hurts; positive Δ = group helps)")
ax.legend(fontsize=9)
ax.set_xlim(min(loo_means) - 0.05, 1.0)
plt.tight_layout()
plt.savefig(OUT_DIR / "04A_feature_group_ablation.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 04A_feature_group_ablation.png")

# ── Plot A2: Only one group ────────────────────────────────────────────────────
only_keys  = [k for k in ablation_a if k.startswith("only")]
only_means = [ablation_a[k][0] for k in only_keys]
only_stds  = [ablation_a[k][1] for k in only_keys]
only_labels = [k.replace("only ", "") for k in only_keys]
palette = sns.color_palette("tab10", len(only_keys))

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(only_labels[::-1], only_means[::-1], color=palette[::-1],
               capsize=4, edgecolor="white", height=0.55)
for bar, m in zip(bars, only_means[::-1]):
    ax.text(m + 0.002, bar.get_y() + bar.get_height()/2,
            f"{m:.4f}", va="center", fontsize=9)
ax.axvline(full_acc, color="black", linestyle="--", lw=1.2, label=f"All features = {full_acc:.4f}")
ax.set_xlabel("5-fold CV Accuracy")
ax.set_title("Ablation A: Accuracy Using Only One Feature Group")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "04A_only_one_group.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 04A_only_one_group.png")

# ── Plot B: Model architecture ────────────────────────────────────────────────
b_keys  = list(ablation_b.keys())
b_means = [ablation_b[k][0] for k in b_keys]
b_stds  = [ablation_b[k][1] for k in b_keys]
b_palette = sns.color_palette("Set2", len(b_keys))

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(b_keys[::-1], b_means[::-1], xerr=b_stds[::-1],
               color=b_palette[::-1], capsize=4, edgecolor="white", height=0.55)
for bar, m in zip(bars, b_means[::-1]):
    ax.text(m + 0.002, bar.get_y() + bar.get_height()/2,
            f"{m:.4f}", va="center", fontsize=9)
best_b = max(b_means)
ax.axvline(best_b, color="red", linestyle="--", lw=1.2, label=f"Best = {best_b:.4f}")
ax.set_xlabel("5-fold CV Accuracy")
ax.set_title("Ablation B: Model Architecture Comparison\n(same 237-feature input)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "04B_model_ablation.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 04B_model_ablation.png")

# ── Plot C: Number of temporal segments ───────────────────────────────────────
c_ns    = list(ablation_c.keys())
c_means = [ablation_c[k][0] for k in c_ns]
c_stds  = [ablation_c[k][1] for k in c_ns]

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(c_ns, [m - s for m, s in zip(c_means, c_stds)],
                       [m + s for m, s in zip(c_means, c_stds)],
                alpha=0.2, color="#2ecc71")
ax.plot(c_ns, c_means, "o-", color="#2ecc71", lw=2, markersize=8)
for n, m in zip(c_ns, c_means):
    ax.text(n, m + 0.003, f"{m:.4f}", ha="center", fontsize=8)
best_ns = c_ns[np.argmax(c_means)]
ax.axvline(best_ns, color="red", linestyle="--", lw=1.2, label=f"Best n={best_ns}")
ax.set_xlabel("Number of temporal segments")
ax.set_ylabel("5-fold CV Accuracy")
ax.set_title("Ablation C: Effect of Number of Temporal Segments\n(300s window divided into N equal parts)")
ax.set_xticks(c_ns)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "04C_segment_ablation.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 04C_segment_ablation.png")

# ── Plot D: Clipping threshold ─────────────────────────────────────────────────
d_keys  = list(ablation_d.keys())
d_means = [ablation_d[k][0] for k in d_keys]
d_stds  = [ablation_d[k][1] for k in d_keys]
best_d = max(d_means)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(d_keys, d_means, yerr=d_stds, color="#e67e22", capsize=4,
              edgecolor="white")
for bar, m in zip(bars, d_means):
    ax.text(bar.get_x() + bar.get_width()/2, m + 0.003,
            f"{m:.4f}", ha="center", va="bottom", fontsize=9)
ax.axhline(best_d, color="red", linestyle="--", lw=1.2, label=f"Best = {best_d:.4f}")
ax.set_xlabel("Clipping threshold (±value)")
ax.set_ylabel("5-fold CV Accuracy")
ax.set_title("Ablation D: Effect of Outlier Clipping Threshold")
ax.set_xticklabels(d_keys, rotation=20, ha="right")
ax.set_ylim(min(d_means) - 0.05, max(d_means) + 0.04)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "04D_clipping_ablation.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 04D_clipping_ablation.png")

# ── Combined 4-panel summary ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Ablation Study — All Dimensions", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.barh(loo_keys[::-1], loo_means[::-1], color=bar_colors[::-1], capsize=4, edgecolor="white", height=0.6)
ax.axvline(full_acc, color="#27ae60", linestyle="--", lw=1.5)
for i, (m, k) in enumerate(zip(loo_means[::-1], loo_keys[::-1])):
    ax.text(max(loo_means)*0.5, i, f"{m:.4f}", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_xlabel("CV Accuracy"); ax.set_title("A: Feature Group LOO")

ax = axes[0, 1]
ax.barh(b_keys[::-1], b_means[::-1], color=b_palette[::-1], capsize=4, edgecolor="white", height=0.55)
for i, m in enumerate(b_means[::-1]):
    ax.text(max(b_means)*0.5, i, f"{m:.4f}", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_xlabel("CV Accuracy"); ax.set_title("B: Model Architecture")

ax = axes[1, 0]
ax.fill_between(c_ns, [m-s for m,s in zip(c_means,c_stds)], [m+s for m,s in zip(c_means,c_stds)], alpha=0.2, color="#2ecc71")
ax.plot(c_ns, c_means, "o-", color="#2ecc71", lw=2, markersize=7)
ax.set_xlabel("# Segments"); ax.set_ylabel("CV Accuracy"); ax.set_title("C: Temporal Segments"); ax.set_xticks(c_ns)

ax = axes[1, 1]
ax.bar(d_keys, d_means, color="#e67e22", capsize=4, edgecolor="white")
ax.set_xlabel("Clip ±threshold"); ax.set_ylabel("CV Accuracy"); ax.set_title("D: Clipping Threshold")
ax.set_xticklabels(d_keys, rotation=20, ha="right")

plt.tight_layout()
plt.savefig(OUT_DIR / "04_ablation_overview.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 04_ablation_overview.png")

# ── Save results ───────────────────────────────────────────────────────────────
rows = []
for section, res in all_results.items():
    for name, (m, s) in res.items():
        rows.append({"section": section, "config": name, "mean_acc": m, "std_acc": s})
pd.DataFrame(rows).to_csv(OUT_DIR / "04_ablation_results.csv", index=False)

print(f"\nAblation study complete. Outputs saved to: {OUT_DIR}")
