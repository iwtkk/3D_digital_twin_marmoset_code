"""
1 セッションの床 KP bZ 分布を before / after (affine 補正) の 2 段で描く (plot)。

入力 (--input-dir, 既定 ../ez2use/outputs/animate_temp/transformedcsv):
  {input-dir}/{session}_points.csv_transformedtest.csv  (dailylife_ 接頭辞も許容)
出力 (--output-dir, 既定 scripts/dev/floorZ_plot):
  before_after_{session}.png

使い方:
  OPENBLAS_NUM_THREADS=1 python plot_before_after_session.py --session cj611_20230211_110000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("../ez2use/outputs/animate_temp/transformedcsv")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_plot")


def find_csv(input_dir: Path, session: str) -> Path:
    for cand in (f"{session}_points.csv_transformedtest.csv",
                 f"dailylife_{session}_points.csv_transformedtest.csv"):
        p = input_dir / cand
        if p.exists():
            return p
    raise FileNotFoundError(f"{session} の transformed CSV が見つかりません: {input_dir}")


def _draw_dist(ax, vals, color, peaks, title, xlabel):
    ax.hist(vals, bins=100, color=color, alpha=0.45, density=True)
    if len(vals) > 10:
        xg, dens = core.kde_curve(vals, 500)
        ax.plot(xg, dens, "k-", lw=1.0, alpha=0.7)
    for p in peaks:
        ax.axvline(p, color="red", lw=1.3, alpha=0.9)
    ymax = ax.get_ylim()[1]
    for h, lbl in zip(core.PHYSICAL_HEIGHTS_M, core.PHYS_LABELS):
        ax.axvline(h, color="green", lw=1.0, ls=":", alpha=0.8)
        ax.text(h + 0.004, ymax * 0.96, lbl, fontsize=8, color="darkgreen",
                rotation=90, va="top")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xlim(0.65, 1.62)
    ax.tick_params(labelsize=9)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"transformed CSV のディレクトリ (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"図の出力先 (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", required=True, help="例: cj611_20230211_110000")
    args = ap.parse_args()

    session = re.sub(r"^dailylife_", "", args.session)
    path = find_csv(args.input_dir, session)
    color = core.ANIMAL_COLORS.get(session.split("_")[0], "gray")

    df = pd.read_csv(path)
    z_cols = core.floor_z_cols(df.columns)
    vals = core.trim_1_99(df[z_cols].min(axis=1).values)
    peaks = core.detect_peaks_kde(vals)
    a, b, status = core.fit_affine(peaks)
    vals_c, peaks_c = a * vals + b, a * peaks + b
    fr = (peaks_c[0] - core.PHYSICAL_HEIGHTS_M[0]) * 1000 if len(peaks) else float("nan")
    warn = " !" if core.slope_warn(a) else ""

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _draw_dist(axes[0], vals, color, peaks, f"{session}  BEFORE", "bZ (m)")
    _draw_dist(axes[1], vals_c, color, peaks_c,
               f"AFTER [{status}] a={a:.3f} b={b:+.3f} fr={fr:+.0f}mm{warn}",
               "bZ corrected (m)")
    fig.legend(handles=[
        Line2D([0], [0], color="red", lw=1.3, label="KDE peak"),
        Line2D([0], [0], color="green", lw=1.0, ls=":", label="physical height"),
    ], loc="lower right", fontsize=9)
    fig.suptitle(f"BEFORE vs AFTER affine correction — floor KP bZ ({session})", fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"before_after_{session}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  peaks(before)={np.round(peaks, 3).tolist()} a={a:.3f} b={b:+.3f} "
          f"status={status} fr={fr:+.0f}mm")


if __name__ == "__main__":
    main()
