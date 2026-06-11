"""
補正前後 (KDE affine 補正) のキーポイント位置分布を並べて比較する図。

affine 補正は Z 軸のみを変更するため、垂直方向 (Z) と側面図 (X–Z) が主な比較対象。
水平 (X/Y) は原理上ほぼ不変だが、参考として併記する。

位置ヒストグラムは plot_blender_bbox_outlier_dist.py のヘルパーで集計し、
各 CSV ディレクトリ隣の blender_bbox_outliers/position_hist.npz にキャッシュする
（存在すれば再計算せず読み込む）。

使い方:
  python3 scripts/dev/outliers/blender_bbox/plot_pre_post_compare.py

  # 明示指定
  python3 scripts/dev/outliers/blender_bbox/plot_pre_post_compare.py \
      --pre-dir  /data/iwata/ez2use/outputs/animate_temp/transformedcsv \
      --pre-glob '*_transformedtest.csv' \
      --post-dir outputs/dev/affine_pkl_csv/corrected \
      --post-glob '*_corrected.csv'

出力:
  position_pre_post.png  (--out で変更可)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

# 同ディレクトリのメインスクリプトからヘルパーを再利用
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_blender_bbox_outlier_dist import (   # noqa: E402
    BBOX_LO, BBOX_HI, position_edges, accumulate_position_hists,
)

DEF_PRE_DIR  = Path("../ez2use/outputs/animate_temp/transformedcsv")
DEF_PRE_GLOB = "*_transformedtest.csv"
DEF_POST_DIR = Path("scripts/dev/floorZ_files/corrected")
DEF_POST_GLOB = "*_corrected.csv"
DEF_OUTPUT = Path("scripts/dev/outliers_plot")


def get_hists(csv_dir: Path, glob: str, cache: Path | None):
    """npz キャッシュがあれば読み、なければ集計して保存する。"""
    if cache and cache.exists():
        d = np.load(cache)
        H = {k: d[k] for k in ("x", "y", "z", "xy", "xz")}
        print(f"  loaded cache: {cache}  (n_pos={int(d['n_pos']):,}, "
              f"n_sessions={int(d['n_sessions'])})")
        return H, int(d["n_pos"]), int(d["n_sessions"])

    files = sorted(csv_dir.glob(glob))
    if not files:
        raise SystemExit(f"CSV が見つかりません: {csv_dir}/{glob}")
    print(f"  accumulating {len(files)} CSV(s) from {csv_dir} …")
    H, n_pos = accumulate_position_hists(files)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        ex, ey, ez = position_edges()
        np.savez_compressed(cache, x=H["x"], y=H["y"], z=H["z"],
                            xy=H["xy"], xz=H["xz"], ex=ex, ey=ey, ez=ez,
                            n_pos=n_pos, n_sessions=len(files))
        print(f"  saved cache: {cache}")
    return H, n_pos, len(files)


def bbox_rect(j_h, j_v):
    return Rectangle((BBOX_LO[j_h], BBOX_LO[j_v]),
                     BBOX_HI[j_h] - BBOX_LO[j_h], BBOX_HI[j_v] - BBOX_LO[j_v],
                     fill=False, ec="red", lw=2.0, ls="--", label="Blender bbox")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", "--post-dir", dest="post_dir", type=Path, default=DEF_POST_DIR,
                    help=f"補正後 (POST) CSV のディレクトリ (default: {DEF_POST_DIR})")
    ap.add_argument("--pre-dir", type=Path, default=DEF_PRE_DIR,
                    help=f"補正前 (PRE) CSV のディレクトリ (default: {DEF_PRE_DIR})")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=DEF_OUTPUT,
                    help=f"図の出力先 (default: {DEF_OUTPUT})")
    ap.add_argument("--session", default=None, help="(互換用; 全 CSV を集計するため未使用)")
    ap.add_argument("--glob", "--post-glob", dest="post_glob", default=DEF_POST_GLOB)
    ap.add_argument("--pre-glob", default=DEF_PRE_GLOB)
    ap.add_argument("--no-cache", action="store_true",
                    help="npz キャッシュを使わず毎回集計し直す")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.out = args.output_dir / "position_pre_post.png"

    pre_cache  = None if args.no_cache else args.output_dir / "position_hist_pre.npz"
    post_cache = None if args.no_cache else args.output_dir / "position_hist_post.npz"

    print("PRE (before correction):")
    Hpre, npre, spre = get_hists(args.pre_dir, args.pre_glob, pre_cache)
    print("POST (after correction):")
    Hpost, npost, spost = get_hists(args.post_dir, args.post_glob, post_cache)

    ex, ey, ez = position_edges()
    cx = 0.5 * (ex[:-1] + ex[1:]); cy = 0.5 * (ey[:-1] + ey[1:]); cz = 0.5 * (ez[:-1] + ez[1:])

    # fraction (per-bin 確率) に正規化して N 非依存に比較
    def frac2d(h, n):
        f = h / max(1, n)
        f[f == 0] = np.nan
        return f
    fr_pre_xy  = frac2d(Hpre["xy"],  npre);  fr_post_xy  = frac2d(Hpost["xy"],  npost)
    fr_pre_xz  = frac2d(Hpre["xz"],  npre);  fr_post_xz  = frac2d(Hpost["xz"],  npost)
    vmax = np.nanmax([fr_pre_xy, fr_post_xy, fr_pre_xz, fr_post_xz])
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("kp position: PRE vs POST affine correction "
                 f"(PRE {spre} sess / POST {spost} sess; "
                 "affine corrects Z only)", fontsize=14, y=0.995)

    # ── row 0: top view X–Y ───────────────────────────────────────────────────
    for col, (title, fr, n) in enumerate(
            [("PRE", fr_pre_xy, npre), ("POST", fr_post_xy, npost)]):
        ax = axes[0, col]
        pc = ax.pcolormesh(ex, ey, fr.T, norm=norm, cmap="viridis")
        fig.colorbar(pc, ax=ax, label="fraction (log)")
        ax.add_patch(bbox_rect(0, 1))
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
        ax.set_title(f"Top view X–Y · {title}  (N={n:,})")
        ax.legend(loc="upper right", fontsize=8)

    # ── row 1: side view X–Z ──────────────────────────────────────────────────
    for col, (title, fr) in enumerate([("PRE", fr_pre_xz), ("POST", fr_post_xz)]):
        ax = axes[1, col]
        pc = ax.pcolormesh(ex, ez, fr.T, norm=norm, cmap="viridis")
        fig.colorbar(pc, ax=ax, label="fraction (log)")
        ax.add_patch(bbox_rect(0, 2))
        ax.axhline(BBOX_LO[2], color="orange", ls=":", lw=1.2)
        ax.axhline(BBOX_HI[2], color="orange", ls=":", lw=1.2)
        ax.set_xlabel("X (m, horizontal)"); ax.set_ylabel("Z (m, vertical)")
        ax.set_title(f"Side view X–Z · {title}")

    # ── row 2 left: vertical Z overlay ────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(cz, Hpre["z"]  / max(1, npre),  color="#1f77b4", lw=1.8, label="PRE")
    ax.plot(cz, Hpost["z"] / max(1, npost), color="#d62728", lw=1.8, label="POST")
    ax.axvline(BBOX_LO[2], color="orange", ls="--", lw=1.4,
               label=f"floor {BBOX_LO[2]:.3f} / ceiling {BBOX_HI[2]:.3f} m")
    ax.axvline(BBOX_HI[2], color="orange", ls="--", lw=1.4)
    ax.set_xlabel("Z (m, vertical)"); ax.set_ylabel("fraction per bin")
    ax.set_title("Vertical (Z) distribution — overlay")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── row 2 right: horizontal X/Y overlay (補正不変の確認) ──────────────────
    ax = axes[2, 1]
    ax.plot(cx, Hpre["x"]  / max(1, npre),  color="#2ca02c", lw=1.6, label="X · PRE")
    ax.plot(cx, Hpost["x"] / max(1, npost), color="#2ca02c", lw=1.6, ls="--", label="X · POST")
    ax.plot(cy, Hpre["y"]  / max(1, npre),  color="#9467bd", lw=1.6, label="Y · PRE")
    ax.plot(cy, Hpost["y"] / max(1, npost), color="#9467bd", lw=1.6, ls="--", label="Y · POST")
    for j, color in [(0, "#2ca02c"), (1, "#9467bd")]:
        ax.axvline(BBOX_LO[j], color=color, ls=":", lw=1.0)
        ax.axvline(BBOX_HI[j], color=color, ls=":", lw=1.0)
    ax.set_xlabel("Horizontal position (m)"); ax.set_ylabel("fraction per bin")
    ax.set_title("Horizontal (X, Y) — overlay  (Z-only correction -> nearly identical)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(pad=2.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
