"""
3D キーポイント品質 (score/err) 分布の可視化 (plot)。

run/analyze_quality_score.py が出力した quality_dist_summary.json と
quality_subsample.npz を読み、全体ヒスト・joint hexbin・per-kp 分布・
session×kp ヒートマップ・clamp/NaN 率を描く。

入力 (--input-dir, 既定 scripts/dev/quality_score_files):
  quality_dist_summary.json, quality_subsample.npz
出力 (--output-dir, 既定 scripts/dev/quality_score_plot):
  dist_score.png / dist_err.png / joint_score_err.png
  perkp_score.png / perkp_err.png / heatmap_score.png / heatmap_err.png / clamp_nan_rate.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_INPUT  = Path("scripts/dev/quality_score_files")
DEFAULT_OUTPUT = Path("scripts/dev/quality_score_plot")

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
N_KP = 20
SCORE_CLAMP = 0.5
CLAMP_EPS = 0.05


def plot_global_hist(score_all, err_all, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(score_all, bins=np.linspace(0.5, 1.05, 120), color="steelblue")
    ax.axvline(SCORE_CLAMP, color="red", ls="--", lw=1, label=f"clamp ({SCORE_CLAMP})")
    ax.axvline(SCORE_CLAMP + CLAMP_EPS, color="orange", ls=":", lw=1, label=f"near-clamp band (+{CLAMP_EPS})")
    ax.set_yscale("log"); ax.set_xlabel("kp3d_score"); ax.set_ylabel("count (log)")
    ax.set_title("Global kp3d_score distribution (all sessions, subsampled)"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "dist_score.png", dpi=150); plt.close(fig)
    print("  saved dist_score.png")

    ev = err_all[~np.isnan(err_all)]
    p995 = np.percentile(ev, 99.5)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(ev, bins=np.linspace(0, p995, 120), color="seagreen")
    for q in (50, 95, 99):
        v = np.percentile(ev, q)
        ax.axvline(v, ls="--", lw=1, color="black", alpha=0.6)
        ax.text(v, ax.get_ylim()[1] * 0.9, f"p{q}={v:.0f}", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("kp3d_err (px)"); ax.set_ylabel("count")
    ax.set_title("Global kp3d_err distribution (all sessions, subsampled)")
    fig.tight_layout(); fig.savefig(out_dir / "dist_err.png", dpi=150); plt.close(fig)
    print("  saved dist_err.png")


def plot_joint(score_all, err_all, out_dir):
    m = ~np.isnan(err_all)
    s = score_all[m]
    e = np.clip(err_all[m], 0, np.percentile(err_all[m], 99.5))
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(s, e, gridsize=60, cmap="viridis", bins="log", mincnt=1)
    ax.axvline(SCORE_CLAMP + CLAMP_EPS, color="orange", ls=":", lw=1)
    ax.set_xlabel("kp3d_score (higher better)"); ax.set_ylabel("kp3d_err px (lower better)")
    ax.set_title("Joint distribution: score vs err\n(top-left = worst; clamp layer at left)")
    fig.colorbar(hb, ax=ax, label="log10 count")
    fig.tight_layout(); fig.savefig(out_dir / "joint_score_err.png", dpi=150); plt.close(fig)
    print("  saved joint_score_err.png")


def plot_perkp(summaries, out_dir):
    def gather(pct_field, idx):
        return np.array([[s["per_kp"][ki][pct_field][idx] if s["per_kp"][ki][pct_field] else np.nan
                          for s in summaries] for ki in range(N_KP)])
    i5, i25, i50, i75, i95 = 1, 2, 3, 4, 5
    for field, pct_field, fname, ylab, invert in [
        ("score", "score_pcts", "perkp_score.png", "kp3d_score", False),
        ("err", "err_pcts", "perkp_err.png", "kp3d_err (px)", True),
    ]:
        med = np.nanmedian(gather(pct_field, i50), axis=1)
        lo = np.nanmedian(gather(pct_field, i25), axis=1)
        hi = np.nanmedian(gather(pct_field, i75), axis=1)
        w5 = np.nanmedian(gather(pct_field, i5), axis=1)
        w95 = np.nanmedian(gather(pct_field, i95), axis=1)
        order = np.argsort(med) if invert else np.argsort(med)[::-1]
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(N_KP)
        ax.vlines(x, w5[order], w95[order], color="lightgray", lw=4, label="p5–p95")
        ax.vlines(x, lo[order], hi[order], color="steelblue", lw=8, label="p25–p75")
        ax.plot(x, med[order], "o", color="black", ms=5, label="median")
        ax.set_xticks(x); ax.set_xticklabels([KP_NAMES[i] for i in order], rotation=45, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(f"Per-keypoint {field} distribution (median of session percentiles)")
        ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=150); plt.close(fig)
        print(f"  saved {fname}")


def plot_heatmaps(summaries, out_dir):
    sessions = [s["session"] for s in summaries]
    sc_mat = np.array([[s["per_kp"][ki]["score_mean"] for ki in range(N_KP)] for s in summaries])
    er_mat = np.array([[(s["per_kp"][ki]["err_mean"] or np.nan) for ki in range(N_KP)] for s in summaries])
    for mat, fname, cmap, title in [
        (sc_mat, "heatmap_score.png", "RdYlGn", "Mean kp3d_score (green=good)"),
        (er_mat, "heatmap_err.png", "RdYlGn_r", "Mean kp3d_err px (green=good)"),
    ]:
        fig, ax = plt.subplots(figsize=(22, max(6, len(sessions) * 0.28 + 2)))
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(N_KP)); ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(sessions))); ax.set_yticklabels(sessions, fontsize=6)
        ax.set_title(f"{title} — session x keypoint")
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=150); plt.close(fig)
        print(f"  saved {fname}")


def plot_clamp_nan(summaries, out_dir):
    clamp = np.array([[s["per_kp"][ki]["clamp_rate"] for ki in range(N_KP)] for s in summaries]).mean(0)
    nan = np.array([[s["per_kp"][ki]["err_nan_rate"] for ki in range(N_KP)] for s in summaries]).mean(0)
    x = np.arange(N_KP)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7))
    a1.bar(x, clamp, color="tomato")
    a1.set_xticks(x); a1.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    a1.set_ylabel(f"score <= {SCORE_CLAMP+CLAMP_EPS} rate")
    a1.set_title("Per-keypoint near-clamp score rate (mean over sessions)")
    a2.bar(x, nan, color="slateblue")
    a2.set_xticks(x); a2.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    a2.set_ylabel("err NaN rate")
    a2.set_title("Per-keypoint reprojection-failure (NaN err) rate")
    fig.tight_layout(); fig.savefig(out_dir / "clamp_nan_rate.png", dpi=150); plt.close(fig)
    print("  saved clamp_nan_rate.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help="quality_dist_summary.json / quality_subsample.npz の場所")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="(互換用; 全体集計のため未使用)")
    args = ap.parse_args()

    js = args.input_dir / "quality_dist_summary.json"
    npz = args.input_dir / "quality_subsample.npz"
    if not js.exists() or not npz.exists():
        print(f"{js} か {npz} がありません。先に run を実行してください。", file=sys.stderr)
        sys.exit(1)
    with open(js) as f:
        summaries = json.load(f)["sessions"]
    d = np.load(npz)
    score_all, err_all = d["score_all"], d["err_all"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(summaries)} sessions. Generating plots …")
    plot_global_hist(score_all, err_all, args.output_dir)
    plot_joint(score_all, err_all, args.output_dir)
    plot_perkp(summaries, args.output_dir)
    plot_heatmaps(summaries, args.output_dir)
    plot_clamp_nan(summaries, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
