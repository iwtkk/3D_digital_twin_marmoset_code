"""
kp3d 外れ値の bbox 境界からのずれ分布を可視化する (plot)。

run/analyze_kp3d_outliers.py が出力した {session}_outlier.json を読み、
log-x ヒストグラム / 0–200mm ズーム / 個体別 / per-kp boxplot を 1 枚にまとめる。

入力 (--input-dir, 既定 scripts/dev/outliers_files):  {session}_outlier.json
出力 (--output-dir, 既定 scripts/dev/outliers_plot):  deviation_dist.png
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DEFAULT_INPUT  = Path("scripts/dev/outliers_files")
DEFAULT_OUTPUT = Path("scripts/dev/outliers_plot")

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
IND_COLORS = {"cj357": "#1f77b4", "cj425": "#ff7f0e", "cj611": "#2ca02c"}


def load(input_dir: Path, session: str | None):
    pat = f"{session}_outlier.json" if session else "*_outlier.json"
    files = sorted(p for p in input_dir.glob(pat) if not p.name.startswith("outlier_summary"))
    devs_all, devs_kp, devs_ind = [], defaultdict(list), defaultdict(list)
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        ind = d["session"].split("_")[0]
        for ev in d["outlier_events"]:
            dm = ev["deviation_mm"]
            devs_all.append(dm)
            devs_kp[ev["kp_name"]].append(dm)
            devs_ind[ind].append(dm)
    return np.array(devs_all), devs_kp, devs_ind, len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help="{session}_outlier.json の場所")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="セッション名 (前方一致) で絞る")
    args = ap.parse_args()

    devs_all, devs_kp, devs_ind, n_files = load(args.input_dir, args.session)
    if len(devs_all) == 0:
        print(f"{args.input_dir} に *_outlier.json がありません。先に run を実行してください。",
              file=sys.stderr)
        sys.exit(1)
    print(f"  {len(devs_all):,} events loaded from {n_files} sessions")

    p50 = float(np.percentile(devs_all, 50))
    p95 = float(np.percentile(devs_all, 95))
    p99 = float(np.percentile(devs_all, 99))
    near, far = devs_all[devs_all <= 200], devs_all[devs_all > 200]

    fig, axes = plt.subplots(4, 2, figsize=(18, 20),
                             gridspec_kw={"height_ratios": [2, 2, 1.5, 2.5]})
    fig.suptitle("kp3d outlier deviation from global bbox boundary (all sessions)",
                 fontsize=14, y=0.995)

    ax = axes[0, 0]
    log_bins = np.logspace(np.log10(max(devs_all.min(), 0.1)), np.log10(devs_all.max() + 1), 100)
    ax.hist(devs_all, bins=log_bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Deviation from bbox boundary (mm) — log scale"); ax.set_ylabel("Count (log scale)")
    ax.set_title(f"All events  ·  N = {len(devs_all):,}")
    for v, c, lb in [(p50, "green", f"p50 = {p50:.0f} mm"), (p95, "orange", f"p95 = {p95:.0f} mm"),
                     (p99, "red", f"p99 = {p99:.0f} mm")]:
        ax.axvline(v, color=c, lw=1.4, ls="--", label=lb)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    ax = axes[0, 1]
    log_bins_ind = np.logspace(np.log10(max(devs_all.min(), 0.1)), np.log10(devs_all.max() + 1), 80)
    for ind, color in IND_COLORS.items():
        d = np.array(devs_ind.get(ind, []))
        if len(d):
            ax.hist(d, bins=log_bins_ind, color=color, alpha=0.55, label=f"{ind}  (N={len(d):,})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Deviation (mm) — log scale"); ax.set_ylabel("Count (log scale)")
    ax.set_title("Per individual"); ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 0]
    ax.hist(near, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Deviation (mm)"); ax.set_ylabel("Count")
    ax.set_title(f"Zoom: 0–200 mm  ·  N = {len(near):,}  ({len(near)/len(devs_all)*100:.1f}%)")
    ax.axvline(p50, color="green", lw=1.4, ls="--", label=f"p50 = {p50:.0f} mm")
    ax.axvline(p95, color="orange", lw=1.4, ls="--", label=f"p95 = {p95:.0f} mm")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    if len(far):
        ax.hist(far, bins=100, color="tomato", edgecolor="none", alpha=0.85)
        ax.set_xlabel("Deviation (mm)"); ax.set_ylabel("Count")
        ax.set_title(f"Far outliers: >200 mm  ·  N = {len(far):,}  ({len(far)/len(devs_all)*100:.2f}%)")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_visible(False)

    ax = axes[2, 0]
    inds = list(IND_COLORS.keys())
    x, w = np.arange(len(inds)), 0.35
    ind_near = [int((np.array(devs_ind.get(i, [])) <= 200).sum()) for i in inds]
    ind_far = [int((np.array(devs_ind.get(i, [])) > 200).sum()) for i in inds]
    bn = ax.bar(x - w/2, ind_near, w, label="near (0–200 mm)", color=[IND_COLORS[i] for i in inds], alpha=0.8)
    bf = ax.bar(x + w/2, ind_far, w, label="far (>200 mm)", color=[IND_COLORS[i] for i in inds], alpha=0.4, hatch="///")
    ax.set_xticks(x); ax.set_xticklabels(inds); ax.set_ylabel("N outlier events")
    ax.set_title("Near vs far outlier events per individual"); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")); ax.grid(axis="y", alpha=0.3)
    for bar in list(bn) + list(bf):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():,.0f}",
                    ha="center", va="bottom", fontsize=7)

    ax = axes[2, 1]
    near_sorted = np.sort(near)
    cdf = np.arange(1, len(near_sorted) + 1) / len(devs_all) * 100
    ax.plot(near_sorted, cdf, color="steelblue", lw=1.5)
    ax.set_xlabel("Deviation (mm)"); ax.set_ylabel("Cumulative % of all events")
    ax.set_title("CDF — near outliers (as % of all outlier events)")
    ax.axhline(95, color="orange", ls="--", lw=1, label=f"95% → {p95:.0f} mm")
    ax.axhline(97.5, color="red", ls="--", lw=1, label=f"97.5% → {np.percentile(devs_all, 97.5):.0f} mm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 200)

    ax = fig.add_subplot(4, 1, 4)
    axes[3, 0].set_visible(False); axes[3, 1].set_visible(False)
    kp_data = [np.array(devs_kp.get(k, [])) for k in KP_NAMES]
    kp_cap = [np.minimum(d, 200) if len(d) else np.array([0.0]) for d in kp_data]
    bp = ax.boxplot(kp_cap, patch_artist=True, showfliers=False, medianprops=dict(color="red", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue"); patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(KP_NAMES) + 1)); ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Deviation (mm, capped at 200)")
    ax.set_title("Per-keypoint deviation distribution (fliers hidden, capped at 200 mm)")
    for i, d in enumerate(kp_data):
        if len(d):
            m = float(np.median(d))
            ax.text(i + 1, min(m, 200) + 1.5, f"{m:.0f}", ha="center", va="bottom", fontsize=7, color="darkred")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "deviation_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
