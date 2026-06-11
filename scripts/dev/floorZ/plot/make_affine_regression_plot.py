"""
垂直 affine 補正の「線形回帰」を説明する図 (plot)。

左: floor KP の補正前 bZ 分布(KDE)と検出ピーク / 物理高さ目標
右: 回帰 (x=検出ピーク → y=物理高さ) と y=x 基準線

入力 (--input-dir, 既定 outputs/3d):  {input-dir}/{session}/kp3d_*.pickle
出力 (--output-dir, 既定 scripts/dev/floorZ_plot): affine_regression_explained.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_plot")

_CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default="cj425_20210428_090000",
                    help="full_affine になるセッションを選ぶと4点揃う")
    args = ap.parse_args()

    pkl = core.resolve_pickle(args.input_dir / args.session)
    if pkl is None:
        print(f"pickle not found: {args.input_dir / args.session}", file=sys.stderr)
        sys.exit(1)

    vals = core.trim_1_99(core.floor_bz_series(pkl))
    xg, dens = core.kde_curve(vals)
    peaks = core.detect_peaks_kde(vals)
    a, b, status, mp, mh = core.fit_affine(peaks, return_pairs=True)
    print(f"peaks={np.round(peaks, 3)}  a={a:.4f} b={b:+.4f} status={status}")

    labels = ["床 floor", "足場 low perch", "棒 high perch", "天井 ceiling"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.8))
    axL.fill_between(xg, dens, color="0.8", zorder=1)
    axL.plot(xg, dens, color="0.4", lw=1)
    for i, (pkx, ph, c) in enumerate(zip(mp, mh, colors)):
        axL.axvline(pkx, color=c, lw=2, zorder=3, label=f"検出ピーク {pkx:.3f} m")
        axL.axvline(ph, color=c, lw=1.4, ls="--", alpha=0.8, zorder=2)
        axL.annotate("", xy=(ph, dens.max() * (0.55 + 0.1 * i)),
                     xytext=(pkx, dens.max() * (0.55 + 0.1 * i)),
                     arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
    axL.set_title("A. floor KP の高さ分布（補正前 bZ）と検出ピーク\n"
                  "実線=KDE検出ピーク, 破線=物理高さ(目標)", fontsize=11)
    axL.set_xlabel("補正前 bZ (m)  ← min(wrist/ankle) per frame の分布")
    axL.set_ylabel("KDE density")
    axL.legend(fontsize=8, loc="upper right")

    xs = np.linspace(0.65, 1.62, 100)
    axR.plot(xs, xs, color="0.6", ls=":", lw=1.5, label="y = x（無補正の基準線）")
    axR.plot(xs, a * xs + b, color="crimson", lw=2.2,
             label=f"回帰直線  y = {a:.4f}·x + ({b:+.4f})")
    for pkx, ph, c, lb in zip(mp, mh, colors, labels):
        axR.scatter([pkx], [ph], s=90, color=c, zorder=5, edgecolor="k", lw=0.6)
        axR.annotate(lb, (pkx, ph), textcoords="offset points", xytext=(8, -12),
                     fontsize=9, color=c)
        axR.plot([pkx, pkx], [0, ph], color=c, ls=":", lw=0.8, alpha=0.5)
    axR.set_title("B. 線形回帰：x=検出ピーク → y=物理高さ", fontsize=11)
    axR.set_xlabel("x = 補正前 bZ（検出ピーク位置, m）")
    axR.set_ylabel("y = 物理高さ (m)")
    axR.set_xlim(0.66, 1.61); axR.set_ylim(0.66, 1.61)
    axR.set_aspect("equal"); axR.legend(fontsize=9, loc="upper left"); axR.grid(alpha=0.25)

    fig.suptitle(f"垂直 affine 補正の回帰 — {args.session} (a={a:.4f}, b={b:+.4f})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "affine_regression_explained.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
