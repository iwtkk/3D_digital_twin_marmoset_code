"""
補正前 (transformed) CSV の床 KP bZ 分布グリッド (plot)。

plot_corrected.py の補正後グリッドと同レイアウトで、affine 未適用の生分布を
全セッション分描く。セル見出しの a / fr は「この分布に補正が何をするか」の参考値。

入力 (--input-dir, 既定 ../ez2use/outputs/animate_temp/transformedcsv):
  {input-dir}/*_transformedtest.csv  (bZ は Blender メートル, affine 未適用)
出力 (--output-dir, 既定 scripts/dev/floorZ_plot):
  height_dist_grid_before.png
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
NCOL = 6

STATUS_SHORT = {"full_affine": "full", "offset_only": "offset",
                "no_peaks": "none", "no_floor_cols": "nofloor"}


def _short_status(status: str) -> str:
    if status.startswith("partial_affine"):
        return "partial"
    return STATUS_SHORT.get(status, status)


def clean_session(fname: str) -> str:
    s = fname.replace("_points.csv_transformedtest.csv", "")
    return re.sub(r"^dailylife_", "", s)


def compute_session(path: Path) -> dict:
    session = clean_session(path.name)
    animal  = session.split("_")[0]
    df = pd.read_csv(path)
    z_cols = core.floor_z_cols(df.columns)
    if not z_cols:
        return {"session": session, "animal": animal, "status": "no_floor_cols"}
    vals  = core.trim_1_99(df[z_cols].min(axis=1).values)
    peaks = core.detect_peaks_kde(vals)
    slope, intercept, status = core.fit_affine(peaks)
    fr = None
    if len(peaks) and status != "no_peaks":
        fr = (slope * peaks[0] + intercept - core.PHYSICAL_HEIGHTS_M[0]) * 1000
    return {"session": session, "animal": animal, "status": status,
            "slope": slope, "intercept": intercept, "fr": fr,
            "_vals": vals, "_peaks": peaks}


def plot_grid(recs, out_path: Path):
    n = len(recs)
    nrow = (n + NCOL - 1) // NCOL
    fig, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * 4.2, nrow * 3.36))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, recs):
        color = core.ANIMAL_COLORS.get(r["animal"], "gray")
        if "_vals" not in r:
            ax.set_title(f'{r["session"][3:]}\n[{_short_status(r["status"])}]', fontsize=7)
            continue
        vals, peaks = r["_vals"], r["_peaks"]
        ax.hist(vals, bins=100, color=color, alpha=0.45, density=True)
        if len(vals) > 10:
            xg, dens = core.kde_curve(vals, 500)
            ax.plot(xg, dens, "k-", lw=1.0, alpha=0.7)
        for p in peaks:
            ax.axvline(p, color="red", lw=1.3, alpha=0.9)
        ymax = ax.get_ylim()[1]
        for h, lbl in zip(core.PHYSICAL_HEIGHTS_M, core.PHYS_LABELS):
            ax.axvline(h, color="green", lw=1.0, ls=":", alpha=0.8)
            ax.text(h + 0.004, ymax * 0.96, lbl, fontsize=6, color="darkgreen",
                    rotation=90, va="top")
        warn = " !" if core.slope_warn(r["slope"]) else ""
        fr_str = f' fr={r["fr"]:+.0f}mm' if r["fr"] is not None else ""
        ax.set_title(f'{r["session"][3:]}\n[{_short_status(r["status"])}] '
                     f'a={r["slope"]:.3f}{fr_str}{warn}', fontsize=7)
        ax.set_xlabel("bZ (m)", fontsize=7)
        ax.set_xlim(0.65, 1.62)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.legend(handles=[
        Line2D([0], [0], color="red", lw=1.3, label="KDE peak"),
        Line2D([0], [0], color="green", lw=1.0, ls=":", label="physical height"),
    ], loc="lower right", fontsize=9)
    fig.suptitle("Height dist BEFORE affine correction (bZ, m)\n"
                 "red=KDE peak  green=physical heights", fontsize=12)
    plt.tight_layout(rect=[0, 0.005, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}  ({nrow}x{NCOL} cells, {n} sessions)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"transformed CSV のディレクトリ (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"図の出力先 (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="セッション名 (前方一致) で絞る")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*_transformedtest.csv"),
                   key=lambda p: clean_session(p.name))
    if args.session:
        files = [p for p in files if clean_session(p.name).startswith(args.session)]
    print(f"Processing {len(files)} transformed CSV files ...")
    recs = [compute_session(p) for p in files]
    if not recs:
        print("対象 CSV がありません", file=sys.stderr)
        sys.exit(1)
    plot_grid(recs, args.output_dir / "height_dist_grid_before.png")


if __name__ == "__main__":
    main()
