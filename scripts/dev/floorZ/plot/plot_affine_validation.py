"""
垂直 affine 補正の検証プロット (plot)。

run/validate_affine.py と同じ scaletest CSV から床 KP bZ 分布・KDE ピーク・affine を
再計算し、(1) 全セッション分布グリッド (2) affine 係数サマリーを描く。

入力 (--input-dir, 既定 outputs/animate_temp/test_scale):
  {input-dir}/*_scaletest.csv
出力 (--output-dir, 既定 scripts/dev/floorZ_plot):
  height_dist_grid.png
  affine_summary_plot.png
"""
from __future__ import annotations

import argparse
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

DEFAULT_INPUT  = Path("outputs/animate_temp/test_scale")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_plot")


def process_csv(path: Path) -> dict:
    session = path.name.split("_points.csv")[0]
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
        fr = round((slope * peaks[0] + intercept - core.PHYSICAL_HEIGHTS_M[0]) * 1000, 1)
    return {"session": session, "animal": animal, "n_peaks": len(peaks),
            "slope": round(slope, 4), "intercept": round(intercept, 4),
            "status": status, "slope_warn": core.slope_warn(slope),
            "floor_residual_mm": fr, "_vals": vals, "_peaks": peaks}


def plot_grid(records, out_path: Path, ncol: int = 6):
    n = len(records)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 2.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, records):
        color = core.ANIMAL_COLORS.get(r["animal"], "gray")
        if "_vals" not in r:
            ax.set_title(f'{r["session"][3:]}\n{r["status"]}', fontsize=6)
            continue
        vals, peaks = r["_vals"], r["_peaks"]
        ax.hist(vals, bins=100, color=color, alpha=0.5, density=True)
        xg, dens = core.kde_curve(vals, 400)
        ax.plot(xg, dens, "k-", lw=1, alpha=0.6)
        for mu in peaks:
            ax.axvline(mu, color="red", lw=1.2, alpha=0.8)
        for h in core.PHYSICAL_HEIGHTS_M:
            ax.axvline(h, color="green", lw=0.8, ls=":", alpha=0.7)
        warn = " !" if r["slope_warn"] else ""
        ax.set_title(f'{r["session"][3:]}\npeaks={r["n_peaks"]} a={r["slope"]:.3f}{warn}',
                     fontsize=6)
        ax.set_xlabel("bZ (m)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_xlim(0.65, 1.60)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.legend(handles=[
        Line2D([0], [0], color="red", lw=1.5, label="KDE peak"),
        Line2D([0], [0], color="green", lw=0.8, ls=":", label="physical height"),
    ], loc="lower right", fontsize=8)
    fig.suptitle("Height dist per session (bZ m)\nred=KDE peak  green=physical heights",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_affine_summary(records, out_path: Path):
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in records if "slope" in r])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for animal, grp in df.groupby("animal"):
        axes[0].hist(grp["slope"], bins=30, alpha=0.6,
                     color=core.ANIMAL_COLORS.get(animal, "gray"), label=animal)
    axes[0].axvline(1.0, color="k", lw=1.5, ls="--", label="ideal (a=1)")
    axes[0].set_xlabel("slope (a)"); axes[0].set_title("Affine slope distribution")
    axes[0].legend(fontsize=8)
    for animal, grp in df.groupby("animal"):
        axes[1].hist(grp["intercept"], bins=30, alpha=0.6,
                     color=core.ANIMAL_COLORS.get(animal, "gray"), label=animal)
    axes[1].axvline(0.0, color="k", lw=1.5, ls="--", label="ideal (b=0)")
    axes[1].set_xlabel("intercept (b) m"); axes[1].set_title("Affine intercept distribution")
    axes[1].legend(fontsize=8)
    df_valid = df.dropna(subset=["floor_residual_mm"])
    for animal, grp in df_valid.groupby("animal"):
        axes[2].hist(grp["floor_residual_mm"], bins=30, alpha=0.6,
                     color=core.ANIMAL_COLORS.get(animal, "gray"), label=animal)
    axes[2].axvline(0.0, color="k", lw=1.5, ls="--")
    axes[2].set_xlabel("floor residual (mm)")
    axes[2].set_title("Floor peak residual after affine")
    axes[2].legend(fontsize=8)
    fig.suptitle("Affine correction summary (all sessions)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"scaletest CSV のディレクトリ (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"図の出力先 (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None, help="セッション名 (前方一致) で絞る")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*_scaletest.csv"))
    if args.session:
        files = [p for p in files if p.name.startswith(args.session)]
    print(f"Processing {len(files)} CSV files ...")
    records = [process_csv(p) for p in files]
    if not records:
        print("対象 CSV がありません", file=sys.stderr)
        sys.exit(1)
    plot_grid(records, args.output_dir / "height_dist_grid.png")
    plot_affine_summary(records, args.output_dir / "affine_summary_plot.png")


if __name__ == "__main__":
    main()
