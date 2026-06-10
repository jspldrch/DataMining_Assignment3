# run21: diagnostic script only — no model training, class 2 analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() \
          else Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

def find_npz(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits: return hits[0]
    local = Path(__file__).parent / "outputs" / name
    if local.exists(): return str(local)
    raise FileNotFoundError(f"{name} not found")

tr     = np.load(find_npz("train_data.npz"), allow_pickle=True)
X_tr   = np.nan_to_num(tr["X"].astype(np.float32), nan=0.0)
y_tr   = tr["y"].astype(np.int32)
users  = tr["users"]

unique_users = np.unique(users)
N_USERS   = len(unique_users)
N_CLASSES = 6
CLASS_NAMES = ["sit/stand", "walk flat", "walk down", "walk up", "running", "other"]

print(f"Train shape: {X_tr.shape}  --  {N_USERS} users, {len(y_tr)} windows")


# 1. per-user class distribution
print("\n1. per-user class distribution")

user_class_counts = np.zeros((N_USERS, N_CLASSES), dtype=int)
for i, uid in enumerate(unique_users):
    idx = np.where(users == uid)[0]
    for c in range(N_CLASSES):
        user_class_counts[i, c] = (y_tr[idx] == c).sum()

for c in range(N_CLASSES):
    n_users_with_class = (user_class_counts[:, c] > 0).sum()
    total = user_class_counts[:, c].sum()
    print(f"\nClass {c} ({CLASS_NAMES[c]}):  {total} windows  "
          f"present in {n_users_with_class}/{N_USERS} users")
    if c in (2, 4, 5):  # print detail for minority classes
        for i, uid in enumerate(unique_users):
            n = user_class_counts[i, c]
            if n > 0:
                print(f"    User {uid:>4}: {n:>4} windows")


# 2. cv fold structure - class 2 samples per fold
print("\n2. cv fold - class 2 validation samples per fold")

user_folds = {u: i % 5 for i, u in enumerate(unique_users)}
fold_ids   = np.array([user_folds[u] for u in users])

for fold in range(5):
    va_idx   = np.where(fold_ids == fold)[0]
    va_users = np.unique(users[va_idx])
    counts_per_class = [(y_tr[va_idx] == c).sum() for c in range(N_CLASSES)]
    print(f"\nFold {fold+1}  ({len(va_users)} users, {len(va_idx)} windows)")
    print(f"  Val class counts: {counts_per_class}")
    c2_users_in_fold = []
    for uid in va_users:
        uidx = va_idx[users[va_idx] == uid]
        n_c2 = (y_tr[uidx] == 2).sum()
        if n_c2 > 0:
            c2_users_in_fold.append((uid, n_c2))
    if c2_users_in_fold:
        print(f"  Class 2 from users: "
              + ", ".join(f"User {u}:{n}" for u, n in c2_users_in_fold))
    else:
        print(f"  !! NO Class 2 windows in validation set !!")


# 3. per-user normalization
def user_normalise(X, user_ids):
    X_out = X.copy()
    for uid in np.unique(user_ids):
        idx = np.where(user_ids == uid)[0]
        data = X[idx]
        mu  = data.mean(axis=(0, 1), keepdims=True)
        sig = data.std(axis=(0, 1), keepdims=True) + 1e-8
        X_out[idx] = (data - mu) / sig
    return X_out

print("\nNormalizing for signal analysis...")
X_norm = user_normalise(X_tr, users)

mx, my, mz = X_norm[:,:,0], X_norm[:,:,1], X_norm[:,:,2]
sx, sy, sz = X_norm[:,:,3], X_norm[:,:,4], X_norm[:,:,5]
mag_mean = np.sqrt(mx**2 + my**2 + mz**2)
mag_std  = np.sqrt(sx**2 + sy**2 + sz**2)
jx = np.diff(mx, axis=1)
jy = np.diff(my, axis=1)
jz = np.diff(mz, axis=1)
mag_jerk = np.sqrt(jx**2 + jy**2 + jz**2)


# 4. signal profiles: C0 vs C1 vs C2 vs C4

class_colors  = {0: "green", 1: "steelblue", 2: "red", 4: "orange"}
class_labels  = {0: "C0 sit/stand", 1: "C1 walk flat", 2: "C2 walk down", 4: "C4 running"}
class_indices = {c: np.where(y_tr == c)[0] for c in range(N_CLASSES)}

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Mean ± 1 std signal over time — focus: C1 vs C2", fontsize=12)

signals_info = [
    (mag_mean, 300, "mag_mean (accel magnitude)"),
    (mag_std,  300, "mag_std (gyro magnitude)"),
    (mag_jerk, 299, "mag_jerk"),
]

for row, (sig, T, title) in enumerate(signals_info):
    t = np.arange(T)
    ax_ts   = axes[row, 0]
    ax_dist = axes[row, 1]

    for c, color in class_colors.items():
        idx = class_indices[c]
        m = sig[idx].mean(0)
        s = sig[idx].std(0)
        ax_ts.plot(t, m, label=class_labels[c], color=color, lw=1.5)
        ax_ts.fill_between(t, m - s, m + s, alpha=0.12, color=color)

    ax_ts.set_title(f"Time profile: {title}")
    ax_ts.set_xlabel("Timestep")
    ax_ts.legend(fontsize=8)

    for c, color in class_colors.items():
        idx = class_indices[c]
        vals = sig[idx].mean(1)
        ax_dist.hist(vals, bins=50, alpha=0.5, label=class_labels[c],
                     color=color, density=True)
    ax_dist.set_title(f"Distribution of window mean: {title}")
    ax_dist.legend(fontsize=8)

plt.tight_layout()
path = OUT_DIR / "run21_signal_profiles.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {path}")


# 5. heatmap: windows per user per class

norm_counts = user_class_counts / (user_class_counts.max(0, keepdims=True) + 1e-10)

fig, ax = plt.subplots(figsize=(9, max(10, N_USERS // 3)))
sns.heatmap(norm_counts, ax=ax, cmap="YlOrRd",
            xticklabels=[f"C{c}\n{CLASS_NAMES[c]}" for c in range(N_CLASSES)],
            yticklabels=[str(u) for u in unique_users],
            cbar_kws={"label": "Normalized count (per class max)"})
ax.set_title("Windows per user per class  (column-normalized)")
ax.set_xlabel("Class"); ax.set_ylabel("User ID")
plt.tight_layout()
path = OUT_DIR / "run21_user_class_heatmap.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# 6. class 2 inter-user variability
print("\n6. class 2 inter-user variability")

c2_user_stats = {}
for uid in unique_users:
    idx = np.where((users == uid) & (y_tr == 2))[0]
    if len(idx) == 0:
        continue
    c2_user_stats[uid] = {
        "n":              len(idx),
        "mag_jerk_mean":  mag_jerk[idx].mean(),
        "mag_mean_mean":  mag_mean[idx].mean(),
        "mag_std_mean":   mag_std[idx].mean(),
        "mag_jerk_std":   mag_jerk[idx].std(axis=1).mean(),
    }

print(f"\n{'User':>6} | {'N':>4} | {'jerk mean':>10} | {'accel mean':>10} | {'gyro mean':>10}")
print("-" * 52)
for uid, st in c2_user_stats.items():
    print(f"{uid:>6} | {st['n']:>4} | "
          f"{st['mag_jerk_mean']:>10.4f} | "
          f"{st['mag_mean_mean']:>10.4f} | "
          f"{st['mag_std_mean']:>10.4f}")

def cv(vals): return np.std(vals) / (np.mean(vals) + 1e-10)

jerk_vals = [s["mag_jerk_mean"] for s in c2_user_stats.values()]
mean_vals = [s["mag_mean_mean"] for s in c2_user_stats.values()]
std_vals  = [s["mag_std_mean"]  for s in c2_user_stats.values()]

print(f"\nCross-user coefficient of variation (std/mean):")
print(f"  Class 2 mag_jerk: {cv(jerk_vals):.3f}")
print(f"  Class 2 mag_mean: {cv(mean_vals):.3f}")
print(f"  Class 2 mag_std:  {cv(std_vals):.3f}")

# Compare to Class 1 (should be lower — easier to generalize)
c1_jerk, c1_mean = [], []
for uid in unique_users:
    idx = np.where((users == uid) & (y_tr == 1))[0]
    if len(idx) == 0: continue
    c1_jerk.append(mag_jerk[idx].mean())
    c1_mean.append(mag_mean[idx].mean())

print(f"\n  Class 1 mag_jerk: {cv(c1_jerk):.3f}  (baseline — easy class)")
print(f"  Class 1 mag_mean: {cv(c1_mean):.3f}")
print(f"\n  high Class 2 CV = users vary a lot = hard to generalize")


# 7. mean feature values by class
print("\n7. mean feature values by class")

rows = []
for c in range(N_CLASSES):
    idx = class_indices[c]
    rows.append({
        "Class":          f"C{c} {CLASS_NAMES[c]}",
        "N":              len(idx),
        "mag_mean":       mag_mean[idx].mean().round(4),
        "mag_mean_std":   mag_mean[idx].std(axis=1).mean().round(4),
        "mag_jerk":       mag_jerk[idx].mean().round(4),
        "mag_jerk_std":   mag_jerk[idx].std(axis=1).mean().round(4),
        "mag_gyro":       mag_std[idx].mean().round(4),
    })

df = pd.DataFrame(rows).set_index("Class")
print(df.to_string())

path = OUT_DIR / "run21_class_summary.csv"
df.to_csv(path)
print(f"\nSaved: {path}")
print(f"\nDiagnostic complete. Check the three plots + summary CSV in {OUT_DIR}")
