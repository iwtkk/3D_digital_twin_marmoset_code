"""
補正済み CSV から per-session 床高さ推定を棒グラフ化する (plot)。

各セッションで床 KP (wrist/ankle) の Z 最小値を取り、その p01 を床レベル推定値
とする。Blender Z は小さい=床側 / 大きい=天井側。

入力 (--input-dir, 既定 scripts/dev/floorZ_files/corrected):
  {input-dir}/*_corrected.csv  (run/pkl_to_corrected_csv.py の出力)
出力 (--output-dir, 既定 scripts/dev/floorZ_plot):
  floor_estimate_corrected.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("scripts/dev/floorZ_files/corrected")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_plot")

REF_STYLE = [("Floor",   "saddlebrown",    "--"),
             ("Scaffold", "steelblue",     "-."),
             ("Bar",      "mediumseagreen", "-."),
             ("Ceiling",  "gray",          "--")]


def estimate_floor_z(csv_path: Path) -> float | None:
    df = pd.read_csv(csv_path, nrows=1)
    cols = core.floor_z_cols(df.columns)
    if not cols:
        return None
    series = pd.read_csv(csv_path, usecols=cols).min(axis=1).values
    return float(np.percentile(series, 1))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"補正済み CSV のディレクトリ (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"図の出力先 (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None, help="セッション名 (前方一致) で絞る")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*_corrected.csv"))
    if args.session:
        files = [p for p in files if p.stem.startswith(args.session)]
    if not files:
        print(f"対象 CSV がありません: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading {len(files)} corrected CSVs ...")

    records = []
    for fp in files:
        z = estimate_floor_z(fp)
        if z is None:
            continue
        session = fp.stem.replace("_corrected", "")
        records.append({"session": session, "animal": session.split("_")[0],
                        "floor_z_m": round(z, 4)})
    df = pd.DataFrame(records).sort_values("session").reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(max(32, n * 0.22), 7))
    colors = [core.ANIMAL_COLORS.get(a, "gray") for a in df["animal"]]
    ax.bar(range(n), df["floor_z_m"], color=colors, alpha=0.85, width=0.75)

    for (label, color, ls), h in zip(REF_STYLE, core.PHYSICAL_HEIGHTS_M):
        ax.axhline(h, color=color, lw=1.8, ls=ls, label=f"{label} ({h:.3f}m)", zorder=3)

    short = [s[len(s.split("_")[0]) + 1:] for s in df["session"]]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=90, ha="center", fontsize=6.5)

    prev = None
    for i, animal in enumerate(df["animal"]):
        if prev is not None and animal != prev:
            ax.axvline(i - 0.5, color="black", lw=0.8, ls=":", alpha=0.5)
        prev = animal

    ref_handles = [plt.Line2D([0], [0], color=c, lw=1.8, ls=ls, label=f"{lb}")
                   for (lb, c, ls) in REF_STYLE]
    ind_handles = [Patch(color=core.ANIMAL_COLORS.get(a, "gray"), alpha=0.85, label=a)
                   for a in sorted(df["animal"].unique())]
    ax.legend(handles=ref_handles + ind_handles, loc="upper right",
              fontsize=8, ncol=2, framealpha=0.9)

    ax.set_ylabel("Floor Z estimate  [p01(min floor-KP Z per frame)]  (m)", fontsize=10)
    ax.set_title("Per-session floor height estimate (corrected CSV)\n"
                 "Lower = closer to floor / Higher = animal stayed on scaffold or bar",
                 fontsize=11)
    ax.set_xlim(-0.8, n - 0.2); ax.set_ylim(0.60, 1.60)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(axis="y", which="major", alpha=0.3)
    ax.grid(axis="y", which="minor", alpha=0.1)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "floor_estimate_corrected.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  ({n} sessions)")


if __name__ == "__main__":
    main()
