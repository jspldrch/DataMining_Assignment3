"""
Assignment 3 – Step 2: Preprocessing & Feature Engineering
Addresses grading question 2 (10%).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

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

# ── Load raw sequences ─────────────────────────────────────────────────────────
print("Loading training data …")
sequences, labels, file_ids = [], [], []

for user_dir in sorted(TRAIN_DIR.iterdir()):
    for csv_path in sorted(user_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        sequences.append(df[FEAT_COLS].values.astype(np.float32))
        labels.append(int(df["label"].iloc[0]))
        file_ids.append(int(df["file_id"].iloc[0]))

X_raw = np.array(sequences)
y     = np.array(labels)
print(f"  Loaded {len(y):,} windows. Shape: {X_raw.shape}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(X_feat, label):
    clf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    scores = cross_val_score(pipe, X_feat, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {label:<55s}  acc = {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean(), scores.std()

# ── Feature builders ───────────────────────────────────────────────────────────
def feat_step0(X):
    return np.hstack([X.mean(axis=1), X.std(axis=1)])

def preprocess_clip(X):
    X = np.where(np.isnan(X), np.nanmedian(X, axis=1, keepdims=True), X)
    return np.clip(X, -10, 10)

def feat_step2(X):
    feats = []
    for ch in range(X.shape[2]):
        s = X[:, :, ch]
        feats += [s.mean(axis=1), s.std(axis=1), s.min(axis=1), s.max(axis=1),
                  s.max(axis=1)-s.min(axis=1), np.median(s,axis=1),
                  np.percentile(s,75,axis=1)-np.percentile(s,25,axis=1),
                  np.array([skew(r) for r in s]), np.array([kurtosis(r) for r in s])]
    return np.column_stack(feats)

def feat_step3(X):
    mag = np.sqrt((X[:,:,:3]**2).sum(axis=2))
    return np.column_stack([feat_step2(X), mag.mean(axis=1), mag.std(axis=1),
                            mag.max(axis=1)-mag.min(axis=1)])

def feat_step4(X, n_segments=10):
    N, T, C = X.shape
    seg_len = T // n_segments
    seg_feats = []
    for i in range(n_segments):
        seg = X[:, i*seg_len:(i+1)*seg_len, :]
        seg_feats += [seg.mean(axis=1), seg.std(axis=1)]
    return np.hstack([feat_step3(X), np.hstack(seg_feats)])

def spectral_features(X):
    N, T, C = X.shape
    out = np.zeros((N, 5*C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            signal = X[n,:,c] - X[n,:,c].mean()
            freqs, psd = welch(signal, fs=1.0, nperseg=min(64, T))
            psd_norm = psd / (psd.sum()+1e-10)
            out[n, c*5:(c+1)*5] = [
                freqs[np.argmax(psd)],
                -np.sum(psd_norm * np.log(psd_norm+1e-10)),
                psd[(freqs>=0.0)&(freqs<0.5)].sum(),
                psd[(freqs>=0.5)&(freqs<2.0)].sum(),
                psd[freqs>=2.0].sum(),
            ]
    return out

def feat_step5(X):
    return np.hstack([feat_step4(X), spectral_features(X)])

# ── Evaluate each step ─────────────────────────────────────────────────────────
print("=" * 72)
print("PREPROCESSING ABLATION – incremental CV accuracy (RF, 5-fold)")
print("=" * 72)

step_info = [
    ("Step 0", "Global mean+std only",    12,  None),
    ("Step 1", "+ Outlier clipping",       12,  None),
    ("Step 2", "+ Extended statistics",    54,  None),
    ("Step 3", "+ Vector magnitude",       57,  None),
    ("Step 4", "+ Temporal segments",     177,  None),
    ("Step 5", "+ Frequency domain",      207,  None),
]

results = {}
results["Step 0: global mean + std (12 feat)"] = evaluate(feat_step0(X_raw), "Step 0: global mean + std (12 feat)")
X_clip = preprocess_clip(X_raw)
results["Step 1: + clip outliers"] = evaluate(feat_step0(X_clip), "Step 1: + clip outliers")
results["Step 2: + extended stats (54 feat)"] = evaluate(feat_step2(X_clip), "Step 2: + extended stats (54 feat)")
results["Step 3: + vector magnitude (57 feat)"] = evaluate(feat_step3(X_clip), "Step 3: + vector magnitude (57 feat)")
results["Step 4: + temporal segments (177 feat)"] = evaluate(feat_step4(X_clip), "Step 4: + temporal segments (177 feat)")
print("  [Computing spectral features – may take a minute …]")
results["Step 5: + frequency domain (207 feat)"] = evaluate(feat_step5(X_clip), "Step 5: + frequency domain (207 feat)")

steps  = list(results.keys())
means  = [v[0] for v in results.values()]
stds   = [v[1] for v in results.values()]
labels_short = ["Step 0\n(12 feat)", "Step 1\n(12 feat)", "Step 2\n(54 feat)",
                "Step 3\n(57 feat)", "Step 4\n(177 feat)", "Step 5\n(207 feat)"]
descriptions = ["Global mean+std", "+ Outlier clipping", "+ Extended statistics",
                "+ Vector magnitude", "+ Temporal segments", "+ Frequency domain"]

# ── Plot 1: Preprocessing progression bar chart ────────────────────────────────
blue_gradient = [plt.cm.Blues(v) for v in np.linspace(0.35, 0.9, len(steps))]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(np.arange(len(steps)), means, yerr=stds, capsize=4,
              color=blue_gradient, edgecolor="white", linewidth=0.8)
for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.005,
            f"{m:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    if i > 0:
        delta = m - means[i-1]
        color = "#27ae60" if delta >= 0 else "#e74c3c"
        sign  = "+" if delta >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2, m/2,
                f"{sign}{delta:.4f}", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")

ax.set_xticks(np.arange(len(steps)))
ax.set_xticklabels(labels_short, fontsize=9)
ax.set_ylabel("5-fold CV Accuracy (Random Forest)")
ax.set_title("Preprocessing Pipeline: Incremental Accuracy Improvement", fontsize=13)
ax.set_ylim(0, 1.05)
ax.axhline(means[0], color="gray", linestyle="--", lw=1, label=f"Baseline = {means[0]:.4f}")
ax.legend(fontsize=9)

legend_patches = [mpatches.Patch(color=blue_gradient[i], label=descriptions[i]) for i in range(len(steps))]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8, title="Feature group")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_preprocessing_steps.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 02_preprocessing_steps.png")

# ── Plot 2: Delta improvements ─────────────────────────────────────────────────
deltas = [means[i] - means[i-1] for i in range(1, len(means))]
delta_labels = descriptions[1:]
colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(delta_labels[::-1], deltas[::-1], color=colors[::-1], edgecolor="white")
for bar, d in zip(bars, deltas[::-1]):
    ax.text(d + 0.001 if d >= 0 else d - 0.001,
            bar.get_y() + bar.get_height()/2,
            f"+{d:.4f}" if d >= 0 else f"{d:.4f}",
            va="center", ha="left" if d >= 0 else "right", fontsize=10, fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Accuracy Improvement over Previous Step")
ax.set_title("Marginal Accuracy Gain of Each Preprocessing Step")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_preprocessing_deltas.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 02_preprocessing_deltas.png")

# ── Plot 3: Summary table as image ────────────────────────────────────────────
n_feats = [12, 12, 54, 57, 177, 207]
table_data = []
for i, (step, desc, n_feat) in enumerate(zip(steps, descriptions, n_feats)):
    delta = f"+{means[i]-means[i-1]:.4f}" if i > 0 else "—"
    table_data.append([descriptions[i], str(n_feat), f"{means[i]:.4f} ± {stds[i]:.4f}", delta])

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
col_headers = ["Feature Group Added", "# Features", "CV Accuracy", "Δ vs Previous"]
tbl = ax.table(cellText=table_data, colLabels=col_headers,
               cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#ecf0f1")
    if c == 3 and r > 0:
        val = table_data[r-1][3]
        if val.startswith("+") and float(val[1:]) > 0.001:
            cell.set_text_props(color="#27ae60", fontweight="bold")
ax.set_title("Preprocessing Steps — Accuracy Summary", fontsize=12, pad=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_preprocessing_table.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 02_preprocessing_table.png")

# ── Save feature matrix ────────────────────────────────────────────────────────
print("\nSaving best feature matrix …")
X_best = feat_step5(X_clip)
np.save(OUT_DIR / "X_train_features.npy", X_best)
np.save(OUT_DIR / "y_train.npy", y)
np.save(OUT_DIR / "file_ids_train.npy", np.array(file_ids))

pd.DataFrame(results, index=["mean_acc", "std_acc"]).T\
    .to_csv(OUT_DIR / "02_preprocessing_results.csv")

print(f"Best feature set: {X_best.shape[1]} features")
print(f"All outputs saved to: {OUT_DIR}")
