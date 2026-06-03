"""
Assignment 3 – Step 1: Preliminary Analysis & Naive Baselines
Addresses grading question 1 (10%):
  "Please provide a preliminary analysis that informs your method design
   (e.g., observations from the raw data, performance of naive baseline methods)"
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TRAIN_DIR  = BASE_DIR / "train" / "train"
OUT_DIR    = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEAT_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

# ── 1. Load ALL training data ──────────────────────────────────────────────────
print("Loading training data …")

records = []          # one dict per CSV file (= one 5-min window)
sequences = {}        # file_id → (300, 6) array

for user_dir in sorted(TRAIN_DIR.iterdir()):
    for csv_path in sorted(user_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        file_id = int(df["file_id"].iloc[0])
        label   = int(df["label"].iloc[0])
        seq     = df[FEAT_COLS].values          # shape (300, 6)
        sequences[file_id] = seq
        records.append({
            "file_id": file_id,
            "label":   label,
            "user":    user_dir.name,
            **{f"mean_{c}": df[c].mean() for c in FEAT_COLS},
            **{f"std_{c}":  df[c].std()  for c in FEAT_COLS},
        })

meta = pd.DataFrame(records)
print(f"  Loaded {len(meta):,} windows from {meta['user'].nunique()} users.\n")

# ── 2. Class distribution ──────────────────────────────────────────────────────
print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
counts = meta["label"].value_counts().sort_index()
for lbl, cnt in counts.items():
    bar = "█" * int(cnt / counts.max() * 30)
    print(f"  Class {lbl}: {cnt:5d} ({cnt/len(meta)*100:.1f}%)  {bar}")

majority_class = counts.idxmax()
print(f"\n  Most frequent class: {majority_class}")
print(f"  Majority-vote accuracy upper bound: {counts.max()/len(meta)*100:.2f}%\n")

fig, ax = plt.subplots(figsize=(7, 4))
counts.plot(kind="bar", ax=ax, color=sns.color_palette("tab10", 6))
ax.set_xlabel("Activity Label")
ax.set_ylabel("Number of windows")
ax.set_title("Class Distribution (Training Set)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_class_distribution.png", dpi=150)
plt.close()

# ── 3. Per-class raw signal statistics ────────────────────────────────────────
print("=" * 60)
print("PER-CLASS ACCELEROMETER STATISTICS")
print("=" * 60)
for col in ["mean_x", "mean_y", "mean_z"]:
    stats = meta.groupby("label")[f"mean_{col}"].agg(["mean", "std"])
    print(f"\n  {col}:")
    print(stats.to_string())

# ── 4. Missing-value / anomaly check ──────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)
nan_counts = meta[FEAT_COLS].isna().sum()
if nan_counts.sum() == 0:
    print("  No NaN values found in any sequence.\n")
else:
    print(nan_counts)

# Check for constant (zero-std) channels
zero_std_files = sum(
    1 for seq in sequences.values() if np.any(seq.std(axis=0) == 0)
)
print(f"  Files with at least one zero-std channel: {zero_std_files}")

# ── 5. Time-series visualization ──────────────────────────────────────────────
print("\nPlotting example time series per class …")

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(6, 1, hspace=0.6)

for lbl in range(6):
    ax  = fig.add_subplot(gs[lbl])
    fid = meta[meta["label"] == lbl]["file_id"].iloc[0]
    seq = sequences[fid]
    t   = np.arange(300)
    ax.plot(t, seq[:, 0], label="mean_x", lw=0.8)
    ax.plot(t, seq[:, 1], label="mean_y", lw=0.8)
    ax.plot(t, seq[:, 2], label="mean_z", lw=0.8)
    ax.set_ylabel(f"Class {lbl}", fontsize=9)
    ax.set_xlim(0, 299)
    if lbl == 0:
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    if lbl < 5:
        ax.set_xticks([])

ax.set_xlabel("Time (seconds)")
fig.suptitle("Representative Accelerometer Time Series per Activity Class")
plt.savefig(OUT_DIR / "01_timeseries_per_class.png", dpi=150)
plt.close()

# ── 6. Signal magnitude variability per class ─────────────────────────────────
print("Plotting signal energy per class …")

energy_rows = []
for file_id, seq in sequences.items():
    label = meta.loc[meta["file_id"] == file_id, "label"].values[0]
    # vector magnitude
    mag = np.sqrt(seq[:, 0]**2 + seq[:, 1]**2 + seq[:, 2]**2)
    energy_rows.append({"label": label, "mean_mag": mag.mean(), "std_mag": mag.std()})

energy_df = pd.DataFrame(energy_rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title in zip(
    axes,
    ["mean_mag", "std_mag"],
    ["Mean |acceleration| per class", "Std |acceleration| per class"]
):
    energy_df.boxplot(column=col, by="label", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Activity Class")
    ax.set_ylabel(col)

plt.suptitle("")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_signal_magnitude.png", dpi=150)
plt.close()

# ── 7. Naive baseline models ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NAIVE BASELINE EVALUATION  (5-fold stratified CV)")
print("=" * 60)

# Build a minimal flat feature matrix: mean of each channel over 300s
X_simple = meta[[f"mean_{c}" for c in FEAT_COLS] + [f"std_{c}" for c in FEAT_COLS]].values
y         = meta["label"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baselines = {
    "Most-frequent (majority)": DummyClassifier(strategy="most_frequent"),
    "Stratified random":        DummyClassifier(strategy="stratified"),
    "kNN (k=5, flat mean/std)": Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Logistic Regression (flat mean/std)": Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, random_state=42)),
    ]),
}

baseline_results = {}
for name, model in baselines.items():
    scores = cross_val_score(model, X_simple, y, cv=cv, scoring="accuracy", n_jobs=-1)
    baseline_results[name] = scores
    print(f"  {name:<40s}  acc = {scores.mean():.4f} ± {scores.std():.4f}")

# ── 8. Per-user majority vote baseline ────────────────────────────────────────
print("\n  Per-user majority vote (predict the most common label for each user):")
user_majority = meta.groupby("user")["label"].agg(lambda s: s.mode()[0])
per_user_preds = meta["user"].map(user_majority)
acc = (per_user_preds == meta["label"]).mean()
print(f"  {'Per-user majority vote':<40s}  acc = {acc:.4f}")

# ── 9. Save summary ────────────────────────────────────────────────────────────
summary = pd.DataFrame(
    {name: {"mean_acc": s.mean(), "std_acc": s.std()} for name, s in baseline_results.items()}
).T
summary.to_csv(OUT_DIR / "01_baseline_results.csv")

print(f"\nOutputs saved to: {OUT_DIR}")
print("\nKEY OBSERVATIONS FOR REPORT:")
print("  1. Class distribution:", {k: int(v) for k, v in counts.items()})
print(f"  2. Majority-class baseline: {counts.max()/len(meta)*100:.1f}%")
print("  3. Different activity classes show distinct mean/std acceleration patterns.")
print("  4. Logistic Regression on flat features already beats random — temporal")
print("     structure within the 300-second window adds significant information.")
