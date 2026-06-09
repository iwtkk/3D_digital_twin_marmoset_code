"""
Plot distribution of kp3d outlier deviations (mm) across all sessions.

Outputs → outputs/analysis_kp3d_outliers/
  deviation_dist.png   — overall distribution (log-x + linear zoom + per-kp + per-individual)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("outputs/analysis_kp3d_outliers")

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]

# ── load all events ──────────────────────────────────────────────────────────
print("Loading events …")
devs_all = []
devs_kp  = defaultdict(list)   # kp_name -> [deviation_mm, ...]
devs_ind = defaultdict(list)   # individual -> [deviation_mm, ...]

files = sorted(OUT_DIR.glob("*_outlier.json"))
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    ind = d["session"].split("_")[0]
    for ev in d["outlier_events"]:
        dm = ev["deviation_mm"]
        devs_all.append(dm)
        devs_kp[ev["kp_name"]].append(dm)
        devs_ind[ind].append(dm)

devs_all = np.array(devs_all)
print(f"  {len(devs_all):,} events loaded")

# ── print stats ──────────────────────────────────────────────────────────────
print("Distribution stats (mm):")
for p in [25, 50, 75, 90, 95, 99, 99.9, 100]:
    print(f"  p{p:5.1f}: {np.percentile(devs_all, p):.2f}")

bins_lo = [0, 10, 20, 50, 100, 200, 500, 1000, 5000, 20000]
counts, _ = np.histogram(devs_all, bins=bins_lo)
print("\nBin counts:")
for lo, hi, cnt in zip(bins_lo, bins_lo[1:], counts):
    print(f"  [{lo:>6}, {hi:>6}) : {cnt:>10,}  ({cnt/len(devs_all)*100:.2f}%)")

# ── precompute ───────────────────────────────────────────────────────────────
p50  = float(np.percentile(devs_all, 50))
p95  = float(np.percentile(devs_all, 95))
p99  = float(np.percentile(devs_all, 99))
near = devs_all[devs_all <= 200]
far  = devs_all[devs_all > 200]

# ── figure layout: 4 rows ────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(18, 20),
                         gridspec_kw={"height_ratios": [2, 2, 1.5, 2.5]})
fig.suptitle("kp3d outlier deviation from global bbox boundary (all sessions)",
             fontsize=14, y=0.995)

# ── row 0 left: log-x histogram (all events) ────────────────────────────────
ax = axes[0, 0]
log_bins = np.logspace(np.log10(max(devs_all.min(), 0.1)),
                       np.log10(devs_all.max() + 1), 100)
ax.hist(devs_all, bins=log_bins, color="steelblue", edgecolor="none", alpha=0.85)
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax.set_xlabel("Deviation from bbox boundary (mm) — log scale")
ax.set_ylabel("Count (log scale)")
ax.set_title(f"All events  ·  N = {len(devs_all):,}")
for v, color, label in [(p50, "green", f"p50 = {p50:.0f} mm"),
                        (p95, "orange", f"p95 = {p95:.0f} mm"),
                        (p99, "red",    f"p99 = {p99:.0f} mm")]:
    ax.axvline(v, color=color, lw=1.4, ls="--", label=label)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.25)

# ── row 0 right: per-individual log-x histogram ──────────────────────────────
ax = axes[0, 1]
colors_ind = {"cj357": "#1f77b4", "cj425": "#ff7f0e", "cj611": "#2ca02c"}
log_bins_ind = np.logspace(np.log10(max(devs_all.min(), 0.1)),
                           np.log10(devs_all.max() + 1), 80)
for ind, color in colors_ind.items():
    d = np.array(devs_ind[ind])
    if len(d) == 0:
        continue
    ax.hist(d, bins=log_bins_ind, color=color, alpha=0.55,
            label=f"{ind}  (N={len(d):,})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax.set_xlabel("Deviation (mm) — log scale")
ax.set_ylabel("Count (log scale)")
ax.set_title("Per individual")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.25)

# ── row 1 left: zoom 0–200 mm ────────────────────────────────────────────────
ax = axes[1, 0]
ax.hist(near, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
ax.set_xlabel("Deviation (mm)")
ax.set_ylabel("Count")
ax.set_title(f"Zoom: 0–200 mm  ·  N = {len(near):,}  ({len(near)/len(devs_all)*100:.1f}%)")
ax.axvline(p50, color="green",  lw=1.4, ls="--", label=f"p50 = {p50:.0f} mm")
ax.axvline(p95, color="orange", lw=1.4, ls="--", label=f"p95 = {p95:.0f} mm")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── row 1 right: far outliers > 200 mm ──────────────────────────────────────
ax = axes[1, 1]
if len(far) > 0:
    ax.hist(far, bins=100, color="tomato", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Deviation (mm)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Far outliers: >200 mm  ·  N = {len(far):,}  ({len(far)/len(devs_all)*100:.2f}%)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
else:
    axes[1, 1].set_visible(False)

# ── row 2: per-individual, near vs far breakdown bar ────────────────────────
ax = axes[2, 0]
ind_near = {k: int((np.array(v) <= 200).sum()) for k, v in devs_ind.items()}
ind_far  = {k: int((np.array(v) >  200).sum()) for k, v in devs_ind.items()}
inds = list(colors_ind.keys())
x = np.arange(len(inds))
w = 0.35
bars_n = ax.bar(x - w/2, [ind_near[i] for i in inds], w,
                label="near (0–200 mm)", color=[colors_ind[i] for i in inds], alpha=0.8)
bars_f = ax.bar(x + w/2, [ind_far[i]  for i in inds], w,
                label="far (>200 mm)",  color=[colors_ind[i] for i in inds], alpha=0.4,
                hatch="///")
ax.set_xticks(x)
ax.set_xticklabels(inds)
ax.set_ylabel("N outlier events")
ax.set_title("Near vs far outlier events per individual")
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(axis="y", alpha=0.3)
for bar in bars_n:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)
for bar in bars_f:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)

# ── row 2 right: CDF of near outliers (0–200 mm) ────────────────────────────
ax = axes[2, 1]
near_sorted = np.sort(near)
cdf = np.arange(1, len(near_sorted) + 1) / len(devs_all) * 100  # % of ALL events
ax.plot(near_sorted, cdf, color="steelblue", lw=1.5)
ax.set_xlabel("Deviation (mm)")
ax.set_ylabel("Cumulative % of all events")
ax.set_title("CDF — near outliers (as % of all outlier events)")
ax.axhline(95, color="orange", ls="--", lw=1, label=f"95% → {p95:.0f} mm")
ax.axhline(97.5, color="red",  ls="--", lw=1, label=f"97.5% → {np.percentile(devs_all,97.5):.0f} mm")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, 200)

# ── row 3: per-keypoint boxplot (capped at 200 mm) ──────────────────────────
ax = fig.add_subplot(4, 1, 4)
axes[3, 0].set_visible(False)
axes[3, 1].set_visible(False)
kp_data     = [np.array(devs_kp[k]) for k in KP_NAMES]
kp_data_cap = [np.minimum(d, 200) for d in kp_data]
bp = ax.boxplot(kp_data_cap, patch_artist=True, showfliers=False,
                medianprops=dict(color="red", lw=1.5))
for patch in bp["boxes"]:
    patch.set_facecolor("lightsteelblue")
    patch.set_alpha(0.8)
ax.set_xticks(range(1, len(KP_NAMES) + 1))
ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Deviation (mm, capped at 200)")
ax.set_title("Per-keypoint deviation distribution  (fliers hidden, values capped at 200 mm for readability)")
medians = [float(np.median(d)) for d in kp_data]
for i, m in enumerate(medians):
    ax.text(i + 1, m + 1.5, f"{m:.0f}", ha="center", va="bottom", fontsize=7, color="darkred")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout(pad=2.0)
out_path = OUT_DIR / "deviation_dist.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
