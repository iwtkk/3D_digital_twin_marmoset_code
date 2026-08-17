"""
KDE affine 補正結果の可視化 (plot)。

pickle から床 KP (wrist/ankle) の bZ 分布を直接計算し、補正前/補正後を
Blender 物理高さ (床/足場/棒/天井) と重ねてプロットする。

入力 (--input-dir, 既定 outputs/3d):
  {input-dir}/{session}/kp3d_fxdJointLen.pickle  (無ければ kp3d.pickle)
出力 (--output-dir, 既定 scripts/dev/floorZ_plot):
  grid_after_correction.png
  grid_before_after.png   (--before-after 指定時)

使い方:
  OPENBLAS_NUM_THREADS=1 python plot_corrected.py                 # 全セッション
  OPENBLAS_NUM_THREADS=1 python plot_corrected.py --before-after
  OPENBLAS_NUM_THREADS=1 python plot_corrected.py --session cj425_20210427_120000
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

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_plot")


def compute_session(session: str, input_dir: Path, n_frames: int = -1) -> dict | None:
    pkl = core.resolve_pickle(input_dir / session)
    if pkl is None:
        return None
    vals = core.trim_1_99(core.floor_bz_series(pkl, n_frames))
    peaks = core.detect_peaks_kde(vals)
    a, b, status = core.fit_affine(peaks)
    return {"session": session, "animal": session.split("_")[0],
            "vals": vals, "peaks": peaks, "a": a, "b": b, "status": status}


def _draw_dist(ax, vals, color, peaks=None, title="", xlabel=""):
    ax.hist(vals, bins=100, color=color, alpha=0.45, density=True)
    if len(vals) > 10:
        xg, dens = core.kde_curve(vals, 500)
        ax.plot(xg, dens, "k-", lw=1.0, alpha=0.7)
    if peaks is not None:
        for p in peaks:
            ax.axvline(p, color="red", lw=1.3, alpha=0.9)
    ymax = ax.get_ylim()[1]
    for h, lbl in zip(core.PHYSICAL_HEIGHTS_M, core.PHYS_LABELS):
        ax.axvline(h, color="green", lw=1.0, ls=":", alpha=0.8)
        ax.text(h + 0.004, ymax * 0.96, lbl, fontsize=6, color="darkgreen",
                rotation=90, va="top")
    ax.set_title(title, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_xlim(0.65, 1.62)
    ax.tick_params(labelsize=6)


def plot_after_grid(recs, out_path: Path, ncol: int = 6):
    n = len(recs)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, recs):
        color = core.ANIMAL_COLORS.get(r["animal"], "gray")
        vals_c = r["a"] * r["vals"] + r["b"]
        peaks_c = r["a"] * r["peaks"] + r["b"]
        warn = " !" if core.slope_warn(r["a"]) else ""
        _draw_dist(ax, vals_c, color, peaks_c,
                   f'{r["session"][3:]}\n[{r["status"]}] a={r["a"]:.3f} b={r["b"]:+.3f}{warn}',
                   "bZ corrected (m)")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.legend(handles=[
        Line2D([0], [0], color="red", lw=1.3, label="KDE peak (corrected)"),
        Line2D([0], [0], color="green", lw=1.0, ls=":", label="physical height"),
    ], loc="lower right", fontsize=9)
    fig.suptitle("AFTER affine correction — floor KP bZ distribution\n"
                 "red=KDE peak  green=physical heights", fontsize=12)
    plt.tight_layout(rect=[0, 0.005, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}  ({nrow}x{ncol} cells, {n} sessions)")


def plot_before_after_grid(recs, out_path: Path):
    n = len(recs)
    fig, axes = plt.subplots(n, 2, figsize=(9, n * 2.6))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(recs):
        color = core.ANIMAL_COLORS.get(r["animal"], "gray")
        warn = " !" if core.slope_warn(r["a"]) else ""
        _draw_dist(axes[i, 0], r["vals"], color, r["peaks"],
                   f'{r["session"][3:]}  BEFORE', "bZ (m)")
        _draw_dist(axes[i, 1], r["a"] * r["vals"] + r["b"], color,
                   r["a"] * r["peaks"] + r["b"],
                   f'AFTER [{r["status"]}] a={r["a"]:.3f} b={r["b"]:+.3f}{warn}',
                   "bZ corrected (m)")
    fig.suptitle("BEFORE vs AFTER affine correction (floor KP bZ)", fontsize=12)
    plt.tight_layout(rect=[0, 0.003, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}  ({n} sessions, before/after)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"pickle のルート (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"図の出力先 (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="セッション名 (省略時は全セッション)")
    ap.add_argument("--before-after", action="store_true",
                    help="before/after 2 列グリッドも出力")
    ap.add_argument("--n-frames", type=int, default=-1, help="処理フレーム数 (-1=全)")
    args = ap.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(d.name for d in args.input_dir.iterdir() if d.is_dir())
    print(f"{len(sessions)} セッションを処理します")

    recs = []
    for i, s in enumerate(sessions, 1):
        try:
            r = compute_session(s, args.input_dir, args.n_frames)
        except Exception as e:
            print(f"  [ERROR] {s}: {e}", file=sys.stderr)
            r = None
        if r is None:
            print(f"  [SKIP] {s}", file=sys.stderr)
            continue
        recs.append(r)
        if i % 20 == 0 or i == len(sessions):
            print(f"  {i}/{len(sessions)} done")

    if not recs:
        print("有効なセッションがありません", file=sys.stderr)
        sys.exit(1)

    plot_after_grid(recs, args.output_dir / "grid_after_correction.png")
    if args.before_after:
        plot_before_after_grid(recs, args.output_dir / "grid_before_after.png")


if __name__ == "__main__":
    main()
