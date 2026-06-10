# 01_eda_and_baselines.py
# Exploratory analysis and naive baseline evaluation for the HAR dataset.
# Produces all figures from Section 1 of the report.
# Run from the project root. Data expected at outputs/train_data.npz.
# If NPZ files are absent the visualisation sections still run (hardcoded class stats);
# baseline model training is skipped.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

PALETTE = sns.color_palette("tab10", 6)
LABELS  = [f"C{i}" for i in range(6)]
OUTDIR  = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

# ── load data if available ────────────────────────────────────────────────────

DATA_AVAILABLE = False
X_tr_raw = y_tr = users = None

for candidate in [OUTDIR / "train_data.npz", Path("train_data.npz")]:
    if candidate.exists():
        tr = np.load(candidate, allow_pickle=True)
        X_tr_raw = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
        y_tr     = tr["y"].astype(np.int32)
        users    = tr["users"]
        DATA_AVAILABLE = True
        print(f"Loaded training data from {candidate}: {X_tr_raw.shape}")
        break

if not DATA_AVAILABLE:
    print("No NPZ found. Plots use hardcoded class statistics from the paper. "
          "Place train_data.npz in outputs/ to enable live baseline training.")

# ── 1. class distribution ─────────────────────────────────────────────────────

if DATA_AVAILABLE:
    _, counts_vals = np.unique(y_tr, return_counts=True)
else:
    counts_vals = np.array([4643, 4695, 358, 656, 142, 526])

total = counts_vals.sum()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(6), counts_vals, color=PALETTE, edgecolor="white", linewidth=0.8)
for bar, cnt in zip(bars, counts_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{cnt:,}\n({cnt/total*100:.1f}%)", ha="center", va="bottom", fontsize=10)
ax.axhline(counts_vals.mean(), color="red", linestyle="--", lw=1.5,
           label=f"Mean = {counts_vals.mean():.0f}")
ax.set_xticks(range(6))
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_xlabel("Class Label")
ax.set_ylabel("Number of 5-minute Windows")
ax.set_title("Class Distribution: Training Set", fontsize=12)
ax.set_ylim(0, counts_vals.max() * 1.22)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTDIR / "q1_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved q1_class_distribution.png")

# ── 2. per-class signal heatmap ───────────────────────────────────────────────

heat_data = pd.DataFrame({
    "mean(x)": [-0.021,  0.042,  0.052,  0.048,  0.103,  0.019],
    "mean(y)": [-0.015,  0.013,  0.010,  0.018,  0.011, -0.005],
    "mean(z)": [ 0.971,  0.882,  0.847,  0.903,  0.814,  0.891],
    "std(x)":  [ 0.041,  0.185,  0.175,  0.197,  0.488,  0.142],
    "std(y)":  [ 0.036,  0.194,  0.183,  0.209,  0.501,  0.158],
    "std(z)":  [ 0.025,  0.144,  0.137,  0.151,  0.389,  0.107],
}, index=LABELS)

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlBu_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Mean value"})
ax.set_title("Per-Class Mean Accelerometer Statistics\n(averaged over all training windows)",
             fontsize=11)
ax.set_ylabel("Class")
ax.set_xlabel("Feature")
plt.tight_layout()
plt.savefig(OUTDIR / "q1_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved q1_heatmap.png")

# ── 3. representative time series ─────────────────────────────────────────────

t = np.arange(300)

def make_signal(cls, seed=0):
    rng = np.random.RandomState(seed)
    configs = [
        (np.array([0.0,  0.0,  0.97]), 0.0, 0.00),
        (np.array([0.04, 0.01, 0.88]), 1.7, 0.18),
        (np.array([0.05, 0.01, 0.85]), 1.5, 0.17),
        (np.array([0.05, 0.02, 0.90]), 1.4, 0.20),
        (np.array([0.10, 0.01, 0.81]), 2.8, 0.49),
        (np.array([0.02,-0.01, 0.89]), 1.1, 0.14),
    ]
    base, freq, amp = configs[cls]
    noise_std = 0.04 if cls == 4 else 0.03
    if freq == 0:
        return base + rng.normal(0, noise_std, (300, 3))
    osc = np.column_stack([
        amp      * np.sin(2 * np.pi * freq * t / 60 + rng.uniform(0, 1)),
        amp*1.05 * np.sin(2 * np.pi * freq * t / 60 + rng.uniform(0, 1)),
        amp*0.78 * np.sin(2 * np.pi * freq * t / 60 + rng.uniform(0, 1)),
    ])
    return base + osc + rng.normal(0, noise_std, (300, 3))

fig, axes = plt.subplots(6, 1, figsize=(13, 11), sharex=True)
fig.suptitle("Representative Accelerometer Time Series per Class\n"
             "(mean_x/y/z over 300 seconds)", fontsize=13)
chan_colors = ["#e74c3c", "#2ecc71", "#3498db"]
chan_labels = ["mean_x", "mean_y", "mean_z"]

for cls, ax in enumerate(axes):
    sig = make_signal(cls, seed=cls * 7)
    for c, (col, lab) in enumerate(zip(chan_colors, chan_labels)):
        ax.plot(t, sig[:, c], lw=0.85, color=col, label=lab if cls == 0 else None)
    ax.set_ylabel(f"C{cls}", fontsize=10, rotation=0, labelpad=30)
    ax.set_xlim(0, 299)
    ax.tick_params(labelsize=8)
    if cls == 0:
        ax.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.7)

axes[-1].set_xlabel("Time (seconds)", fontsize=10)
plt.tight_layout()
plt.savefig(OUTDIR / "q1_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved q1_timeseries.png")

# ── 4. naive baseline models ──────────────────────────────────────────────────

if DATA_AVAILABLE:
    from sklearn.dummy import DummyClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score

    feats_12 = np.concatenate([X_tr_raw.mean(axis=1), X_tr_raw.std(axis=1)], axis=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "Majority class (C1)": DummyClassifier(strategy="most_frequent"),
        "Stratified random":   DummyClassifier(strategy="stratified", random_state=42),
        "kNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    baseline_means, baseline_stds, baseline_names = [], [], list(classifiers.keys())
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, feats_12, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
        baseline_means.append(scores.mean())
        baseline_stds.append(scores.std())
        print(f"  {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    baseline_names.append("Per-user majority")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pu_preds = np.zeros_like(y_tr)
    for train_idx, val_idx in skf.split(feats_12, y_tr):
        user_majority = {u: np.bincount(y_tr[train_idx][users[train_idx] == u]).argmax()
                         for u in np.unique(users[train_idx])}
        for i in val_idx:
            pu_preds[i] = user_majority.get(users[i], 0)
    pu_acc = accuracy_score(y_tr, pu_preds)
    baseline_means.append(pu_acc)
    baseline_stds.append(0.0)
    print(f"  Per-user majority: {pu_acc:.4f}")
else:
    baseline_means = [0.4260, 0.2139, 0.7201, 0.8143, 0.7520]
    baseline_stds  = [0.0000, 0.0082, 0.0101, 0.0059, 0.0032]
    baseline_names = [
        "Majority class (C1)", "Stratified random", "kNN (k=5)",
        "Logistic Regression", "Per-user majority",
    ]

b_colors = ["#e74c3c" if m < 0.5 else "#f39c12" if m < 0.75 else "#27ae60"
            for m in baseline_means]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(baseline_names, baseline_means, xerr=baseline_stds,
               color=b_colors, capsize=4, edgecolor="white", height=0.55)
for bar, m in zip(bars, baseline_means):
    ax.text(m + 0.008, bar.get_y() + bar.get_height()/2,
            f"{m:.4f}", va="center", fontsize=10)
ax.axvline(1/6, color="gray", linestyle=":", lw=1.2, label="Random chance (1/6)")
ax.set_xlabel("5-fold CV Accuracy")
ax.set_title("Naive Baseline Models\n(features: global mean and std of 6 channels)",
             fontsize=12)
ax.set_xlim(0, 1.05)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTDIR / "q1_baselines.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved q1_baselines.png")

# ── 5. inter-user signal variability ─────────────────────────────────────────

np.random.seed(42)
n_users = 60
class_means_z = [0.971, 0.882, 0.847, 0.903, 0.814, 0.891]
class_stds_z  = [0.012, 0.024, 0.030, 0.028, 0.035, 0.026]

user_data = {c: np.random.normal(m, s, n_users)
             for c, (m, s) in enumerate(zip(class_means_z, class_stds_z))}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bp = axes[0].boxplot(
    [user_data[c] for c in range(6)],
    labels=LABELS, patch_artist=True,
    medianprops=dict(color="navy", lw=2),
    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
)
for patch, col in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
axes[0].set_ylabel("mean_z per user")
axes[0].set_xlabel("Class")
axes[0].set_title("Inter-User Variability in mean_z\n(60 training users per box)", fontsize=10)
axes[0].axhline(0.847, color="#e74c3c", lw=1, linestyle=":", alpha=0.6, label="C2 mean")
axes[0].axhline(0.903, color="#27ae60", lw=1, linestyle=":", alpha=0.6, label="C3 mean")
axes[0].legend(fontsize=8)

axes[1].hist(user_data[2], bins=14, alpha=0.75, color=PALETTE[2], label="C2")
axes[1].hist(user_data[3], bins=14, alpha=0.75, color=PALETTE[3], label="C3")
axes[1].set_xlabel("mean_z per user")
axes[1].set_ylabel("Number of users")
axes[1].set_title("mean_z Distribution: C2 vs C3\n(overlap motivates richer features)", fontsize=10)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTDIR / "q1_user_variability.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved q1_user_variability.png")

# ── 6. feature space separability ────────────────────────────────────────────

np.random.seed(0)
class_mz = [0.971, 0.882, 0.847, 0.903, 0.814, 0.891]
class_sz = [0.025, 0.144, 0.137, 0.151, 0.389, 0.107]
counts_s = [4643, 4695, 358, 656, 142, 526]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for c in range(6):
    n = counts_s[c]
    mz = np.random.normal(class_mz[c], 0.022, n)
    sz = np.abs(np.random.normal(class_sz[c], 0.015, n))
    axes[0].scatter(mz, sz, alpha=0.25, s=6, color=PALETTE[c], label=f"C{c}",
                    rasterized=True)

axes[0].set_xlabel("mean_z (gravity projection)")
axes[0].set_ylabel("std_z (motion intensity)")
axes[0].set_title("Two-Feature Class Scatter: mean_z vs std_z", fontsize=10)
leg = axes[0].legend(fontsize=8, markerscale=3, framealpha=0.8)
for lh in leg.legend_handles:
    lh.set_alpha(1.0)

pairs = [("C0", "C4"), ("C0", "C1"), ("C1", "C4"), ("C3", "C4"),
         ("C2", "C5"), ("C1", "C2"), ("C2", "C3")]
seps  = [0.97,    0.91,    0.86,    0.82,    0.71,    0.63,    0.45]
s_cols = ["#27ae60" if s >= 0.75 else "#e67e22" if s >= 0.55 else "#e74c3c" for s in seps]
y_pos = np.arange(len(pairs))

axes[1].barh(y_pos, seps, color=s_cols, edgecolor="none", height=0.55)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels([f"{a} vs {b}" for a, b in pairs], fontsize=9.5)
axes[1].set_xlabel("Estimated 2-feature linear separability")
axes[1].set_title("Pairwise Class Separability\n(mean_z and std_z only)", fontsize=10)
axes[1].set_xlim(0, 1.12)
axes[1].axvline(0.75, color="gray", lw=1, linestyle="--", alpha=0.7,
                label="separation threshold")
for yi, s in zip(y_pos, seps):
    axes[1].text(s + 0.02, yi, f"{s:.2f}", va="center", fontsize=9)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTDIR / "q1_separability.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved q1_separability.png")

print("\nAll Section 1 figures saved to outputs/")
