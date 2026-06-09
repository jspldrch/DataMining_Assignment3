"""
Assignment 3 – Step 1: Preliminary Analysis & Naive Baselines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
PALETTE = sns.color_palette("tab10", 6)

# ── Paths (auto-detects Kaggle / Colab / local) ────────────────────────────────
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
print(f"TRAIN_DIR: {TRAIN_DIR}")

FEAT_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

# ── 1. Load ALL training data ──────────────────────────────────────────────────
print("Loading training data …")
records, sequences = [], {}

for user_dir in sorted(TRAIN_DIR.iterdir()):
    for csv_path in sorted(user_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        file_id = int(df["file_id"].iloc[0])
        label   = int(df["label"].iloc[0])
        seq     = df[FEAT_COLS].values
        sequences[file_id] = seq
        records.append({
            "file_id": file_id, "label": label, "user": user_dir.name,
            **{f"mean_{c}": df[c].mean() for c in FEAT_COLS},
            **{f"std_{c}":  df[c].std()  for c in FEAT_COLS},
        })

meta = pd.DataFrame(records)
print(f"  Loaded {len(meta):,} windows from {meta['user'].nunique()} users.\n")

# ── 2. Class distribution ──────────────────────────────────────────────────────
counts = meta["label"].value_counts().sort_index()
majority_class = counts.idxmax()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white", linewidth=0.8)
for bar, (lbl, cnt) in zip(bars, counts.items()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
            f"{cnt}\n({cnt/len(meta)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Activity Class Label")
ax.set_ylabel("Number of 5-minute Windows")
ax.set_title("Class Distribution — Training Set (11,020 windows)")
ax.set_xticks(counts.index)
ax.set_ylim(0, counts.max() * 1.18)
ax.axhline(counts.mean(), color="red", linestyle="--", lw=1.2, label=f"Mean = {counts.mean():.0f}")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "01_class_distribution.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 01_class_distribution.png")

# ── 3. Per-class accelerometer stats heatmap ───────────────────────────────────
stat_cols = ["mean_mean_x", "mean_mean_y", "mean_mean_z",
             "mean_std_x",  "mean_std_y",  "mean_std_z"]
col_labels = ["mean(x)", "mean(y)", "mean(z)", "std(x)", "std(y)", "std(z)"]
heat_data = meta.groupby("label")[[f"mean_{c}" for c in FEAT_COLS]].mean()
heat_data.columns = col_labels

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlBu_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Mean value"})
ax.set_title("Per-Class Mean Accelerometer Statistics\n(averaged over all windows in that class)")
ax.set_ylabel("Activity Class")
ax.set_xlabel("Feature")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_class_stats_heatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 01_class_stats_heatmap.png")

# ── 4. Missing value / anomaly summary table ───────────────────────────────────
meta_feat_cols = [f"mean_{c}" for c in FEAT_COLS] + [f"std_{c}" for c in FEAT_COLS]
nan_counts = meta[meta_feat_cols].isna().sum()
zero_std_files = sum(1 for seq in sequences.values() if np.any(seq.std(axis=0) == 0))

print(f"\nMissing values: {nan_counts.sum()}")
print(f"Files with zero-std channel: {zero_std_files}")

# ── 5. Time-series per class ───────────────────────────────────────────────────
class_names = {0: "Class 0", 1: "Class 1", 2: "Class 2",
               3: "Class 3", 4: "Class 4", 5: "Class 5"}

fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Representative Accelerometer Time Series per Activity Class", fontsize=13, y=1.01)

for lbl, ax in enumerate(axes):
    fid = meta[meta["label"] == lbl]["file_id"].iloc[0]
    seq = sequences[fid]
    t   = np.arange(300)
    ax.plot(t, seq[:, 0], lw=0.9, color="#e74c3c", label="mean_x")
    ax.plot(t, seq[:, 1], lw=0.9, color="#2ecc71", label="mean_y")
    ax.plot(t, seq[:, 2], lw=0.9, color="#3498db", label="mean_z")
    ax.set_ylabel(f"Class {lbl}", fontsize=10, rotation=0, labelpad=42)
    ax.set_xlim(0, 299)
    ax.tick_params(labelsize=8)
    if lbl == 0:
        ax.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.7)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_timeseries_per_class.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 01_timeseries_per_class.png")

# ── 6. Signal magnitude variability per class ──────────────────────────────────
energy_rows = []
for fid, seq in sequences.items():
    label = meta.loc[meta["file_id"] == fid, "label"].values[0]
    mag = np.sqrt(seq[:, 0]**2 + seq[:, 1]**2 + seq[:, 2]**2)
    energy_rows.append({"Class": f"C{label}", "label": label,
                        "Mean |a|": mag.mean(), "Std |a|": mag.std()})

energy_df = pd.DataFrame(energy_rows).sort_values("label")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in zip(axes,
    ["Mean |a|", "Std |a|"],
    ["Mean Vector Magnitude per Class", "Std of Vector Magnitude per Class"]):
    sns.boxplot(data=energy_df, x="Class", y=col, palette=PALETTE, ax=ax,
                order=[f"C{i}" for i in range(6)])
    ax.set_title(title)
    ax.set_xlabel("Activity Class")
    ax.set_ylabel(col)

plt.suptitle("Acceleration Magnitude Distribution by Class", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_signal_magnitude.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 01_signal_magnitude.png")

# ── 7. Naive baseline evaluation ──────────────────────────────────────────────
print("\nRunning baseline models …")
X_simple = meta[[f"mean_{c}" for c in FEAT_COLS] + [f"std_{c}" for c in FEAT_COLS]].values
y = meta["label"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baselines = {
    "Majority class": DummyClassifier(strategy="most_frequent"),
    "Stratified random": DummyClassifier(strategy="stratified"),
    "kNN (k=5)": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(5))]),
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
}

baseline_results = {}
for name, model in baselines.items():
    scores = cross_val_score(model, X_simple, y, cv=cv, scoring="accuracy", n_jobs=-1)
    baseline_results[name] = scores
    print(f"  {name:<30s}  acc = {scores.mean():.4f} ± {scores.std():.4f}")

user_majority = meta.groupby("user")["label"].agg(lambda s: s.mode()[0])
per_user_acc = (meta["user"].map(user_majority) == meta["label"]).mean()
baseline_results["Per-user majority"] = np.full(5, per_user_acc)
print(f"  {'Per-user majority':<30s}  acc = {per_user_acc:.4f}")

# Baseline bar chart
fig, ax = plt.subplots(figsize=(9, 5))
names  = list(baseline_results.keys())
means  = [v.mean() for v in baseline_results.values()]
stds   = [v.std()  for v in baseline_results.values()]
colors = ["#e74c3c" if m < 0.5 else "#f39c12" if m < 0.8 else "#27ae60" for m in means]

bars = ax.barh(names, means, xerr=stds, color=colors, capsize=4,
               edgecolor="white", height=0.55)
for bar, m in zip(bars, means):
    ax.text(m + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{m:.4f}", va="center", fontsize=10)
ax.set_xlabel("5-fold CV Accuracy")
ax.set_title("Naive Baseline Model Comparison\n(features: global mean & std of each channel)")
ax.set_xlim(0, 1.05)
ax.axvline(1/6, color="gray", linestyle=":", lw=1, label="Random (1/6 classes)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_baselines_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 01_baselines_comparison.png")

# Baseline summary table as image
summary_df = pd.DataFrame({
    "Model": names,
    "Mean Accuracy": [f"{v.mean():.4f}" for v in baseline_results.values()],
    "Std":           [f"± {v.std():.4f}" for v in baseline_results.values()],
})

fig, ax = plt.subplots(figsize=(7, 2.5))
ax.axis("off")
tbl = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
               cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#ecf0f1")
plt.title("Baseline Results Summary", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_baselines_table.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: 01_baselines_table.png")

# Save CSV
pd.DataFrame({n: {"mean_acc": v.mean(), "std_acc": v.std()} for n, v in baseline_results.items()}).T\
    .to_csv(OUT_DIR / "01_baseline_results.csv")

print(f"\nAll outputs saved to: {OUT_DIR}")
print(f"  Class imbalance: classes 2,4,5 are rare ({counts[2]}+{counts[4]}+{counts[5]} = {counts[2]+counts[4]+counts[5]} samples vs {counts[0]+counts[1]} for 0+1)")
print(f"  Best naive baseline: kNN acc={baseline_results['kNN (k=5)'].mean():.4f}")
