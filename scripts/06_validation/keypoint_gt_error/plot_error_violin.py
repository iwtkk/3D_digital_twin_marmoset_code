#!/usr/bin/env python
"""Per-keypoint 3D error against the labeled ground truth (paper Fig. 2E): violin and box plots.

Reads only the dataset: 3dkp_gt/matched_keypoints/SingleAnimal_20.npz (or the original .mat with
--source mat); 1 animal, 100 labeled frames, 20 keypoints.
  - violin: port of violin.m (Hoffmann 2015; original_code/) as called in plotVIolin.mlx: Gaussian KDE with the
    MATLAB ksdensity defaults (normal-approximation bandwidth with the robust sigma = MAD / 0.6745, 100 points
    over [min - 3 bw, max + 3 bw]), width normalised to 0.3, face [1 0.5 0] alpha 0.5, black edge, red median
    line, no mean line; y range 0-40 mm, dotted lines at 5/10/15/20 mm
  - box: as plotBox.py (original_code/): whiskers 1.5 IQR, outliers shown
Unlabeled keypoints (NaN) are left out. (plotBox.py replaced them with 0, which adds false zero errors.)

Outputs ({output_dir}/): fig2e_error_violin.{png,pdf}, fig2e_error_box.{png,pdf}, fig2e_error_summary.csv

Usage:
  .venv/bin/python scripts/06_validation/keypoint_gt_error/plot_error_violin.py [--dataset-root <dataset>] [--source mat]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "config" / "keypoint_gt_error.yaml"
KP_NAMES = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder", "L elbow", "R elbow",
            "L wrist", "R wrist", "L hip", "R hip", "L knee", "R knee", "L ankle", "R ankle", "backbone",
            "tail 1", "tail 2"]


def load_errors(src_dir, stem, source):
    """Per-keypoint error (n_kp, n_sample)."""
    if source == "mat":
        m = loadmat(src_dir / f"{stem}.mat")
        return np.linalg.norm(m["annotKp3ds"] - m["decodedKp3ds"], axis=1)   # (20, 3, n) -> (20, n)
    return np.load(src_dir / f"{stem}.npz")["error_mm"].T                    # (n, 20) -> (20, n)


def ksdensity(x, n_points=100):
    """MATLAB ksdensity defaults for unbounded data: returns density f, grid u, bandwidth bw."""
    x = x[np.isfinite(x)]
    n = len(x)
    sig = np.median(np.abs(x - np.median(x))) / 0.6745
    if sig <= 0:
        sig = x.max() - x.min()
    bw = sig * (4 / (3 * n)) ** 0.2 if sig > 0 else 1.0
    u = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, n_points)
    f = np.exp(-0.5 * ((u[:, None] - x[None, :]) / bw) ** 2).sum(1) / (n * bw * np.sqrt(2 * np.pi))
    return f, u, bw


def finish(fig, ax, n, kp_names, out):
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(kp_names, rotation=45, ha="right")
    ax.set_ylabel("Error (mm)")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_violin(err, kp_names, out):
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for i, x in enumerate(err, start=1):
        f, u, _ = ksdensity(x)
        f = f / f.max() * 0.3
        ax.fill(np.r_[f + i, (i - f)[::-1]], np.r_[u, u[::-1]], color=(1, 0.5, 0), alpha=0.5, lw=0)
        ax.plot(np.r_[f + i, (i - f)[::-1]], np.r_[u, u[::-1]], color="k", lw=0.5)
        med = np.nanmedian(x)
        w = np.interp(med, u, f)
        ax.plot([i - w, i + w], [med, med], color="r", lw=2)
    for y in (5, 10, 15, 20):
        ax.plot([0, len(err) + 1], [y, y], ":k", lw=0.8)
    ax.set_ylim(0, 40)
    finish(fig, ax, len(err), kp_names, out)


def plot_box(err, kp_names, out):
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.boxplot([x[np.isfinite(x)] for x in err], whis=1.5, showfliers=True, widths=0.6,
               medianprops=dict(color="r"), flierprops=dict(marker="o", markersize=3, markerfacecolor="none"))
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    finish(fig, ax, len(err), kp_names, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dataset-root", type=Path, help="overrides dataset_root of the config")
    ap.add_argument("--source", choices=["npz", "mat"], default="npz", help="read the .npz (default) or the original .mat")
    ap.add_argument("--output-dir", type=Path, help="overrides output_dir of the config")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    src = (a.dataset_root or Path(cfg["dataset_root"])) / cfg["gt_dir"] / "matched_keypoints"
    out_dir = a.output_dir or (REPO / cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem, n_kp = cfg["figure"]["stem"], cfg["figure"]["n_kp"]

    err = load_errors(src, stem, a.source)
    assert err.shape[0] == n_kp, err.shape
    names = KP_NAMES[:n_kp]
    plot_violin(err, names, out_dir / "fig2e_error_violin")
    plot_box(err, names, out_dir / "fig2e_error_box")

    q = lambda x, p: np.nanpercentile(x, p, axis=1)
    summ = pd.DataFrame({
        "keypoint": names, "n_labeled": np.isfinite(err).sum(1),
        "median_mm": np.nanmedian(err, axis=1), "mean_mm": np.nanmean(err, axis=1),
        "q25_mm": q(err, 25), "q75_mm": q(err, 75), "max_mm": np.nanmax(err, axis=1),
    })
    summ.loc[len(summ)] = ["all", int(np.isfinite(err).sum()), np.nanmedian(err), np.nanmean(err),
                           np.nanpercentile(err, 25), np.nanpercentile(err, 75), np.nanmax(err)]
    summ.to_csv(out_dir / "fig2e_error_summary.csv", index=False, float_format="%.4f")
    print(f"source: {src / stem}.{a.source}; {err.shape[1]} samples x {n_kp} keypoints")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"-> {out_dir}/fig2e_error_{{violin,box}}.{{png,pdf}}, fig2e_error_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
