"""
KDE affine 補正結果の可視化 (plot)。

corrected CSV と affine_log.csv から補正後 / 補正前-補正後を描く。
pickle には触れない。

入力:
  --input-dir  (既定 scripts/dev/floorZ_files/corrected): *_corrected.csv
  --files-dir  (既定 scripts/dev/floorZ_files):            affine_log.csv
出力 (--output-dir, 既定 scripts/dev/floorZ_plot):
  grid_after_correction.png
  grid_before_after.png   (--before-after 指定時)

使い方:
  python plot_corrected.py
  python plot_corrected.py --before-after
  python plot_corrected.py --session cj425_20210427_120000
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

DEFAULT_INPUT  = Path("outputs/dev/floorZ_files/corrected")
DEFAULT_FILES  = Path("outputs/dev/floorZ_files")
DEFAULT_OUTPUT = Path("outputs/dev/floorZ_plot")


def load_session(session: str, corr_dir: Path, log_row: pd.Series) -> dict | None:
    csv_path = corr_dir / f"{session}_corrected.csv"
    if not csv_path.exists():
        return None

    a      = float(log_row["slope"])
    b      = float(log_row["intercept"])
    status = str(log_row["status"])
    peaks_m_raw = log_row.get("peaks_m", "")
    peaks_before = np.array(
        [float(x) for x in str(peaks_m_raw).split(";") if x.strip()],
        dtype=float,
    )

    df = pd.read_csv(csv_path)
    z_cols = core.floor_z_cols(df.columns)
    if not z_cols:
        return None
    vals_c = core.trim_1_99(df[z_cols].min(axis=1).values)

    return {
        "session": session,
        "animal": session.split("_")[0],
        "vals_c": vals_c,
        "peaks_before": peaks_before,
        "a": a, "b": b, "status": status,
    }


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
        peaks_c = r["a"] * r["peaks_before"] + r["b"]
        warn = " !" if core.slope_warn(r["a"]) else ""
        _draw_dist(ax, r["vals_c"], color, peaks_c,
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
        a, b = r["a"], r["b"]
        vals_before = (r["vals_c"] - b) / a
        peaks_c = a * r["peaks_before"] + b
        warn = " !" if core.slope_warn(a) else ""
        _draw_dist(axes[i, 0], vals_before, color, r["peaks_before"],
                   f'{r["session"][3:]}  BEFORE', "bZ (m)")
        _draw_dist(axes[i, 1], r["vals_c"], color, peaks_c,
                   f'AFTER [{r["status"]}] a={a:.3f} b={b:+.3f}{warn}',
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
                    help=f"補正済み CSV のディレクトリ (既定: {DEFAULT_INPUT})")
    ap.add_argument("--files-dir", type=Path, default=DEFAULT_FILES,
                    help=f"affine_log.csv の場所 (既定: {DEFAULT_FILES})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"図の出力先 (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="セッション名 (省略時は全セッション)")
    ap.add_argument("--before-after", action="store_true",
                    help="before/after 2 列グリッドも出力")
    args = ap.parse_args()

    log_path = args.files_dir / "affine_log.csv"
    if not log_path.exists():
        print(f"{log_path} がありません。先に run/pkl_to_corrected_csv.py を実行してください。",
              file=sys.stderr)
        sys.exit(1)
    log_df = pd.read_csv(log_path)

    if args.session:
        log_df = log_df[log_df["session"] == args.session]
        if log_df.empty:
            print(f"{args.session} が affine_log.csv にありません", file=sys.stderr)
            sys.exit(1)

    sessions = log_df["session"].tolist()
    print(f"{len(sessions)} セッションを処理します")

    recs = []
    for i, session in enumerate(sessions, 1):
        try:
            row = log_df[log_df["session"] == session].iloc[0]
            r = load_session(session, args.input_dir, row)
        except Exception as e:
            print(f"  [ERROR] {session}: {e}", file=sys.stderr)
            r = None
        if r is None:
            print(f"  [SKIP] {session}", file=sys.stderr)
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
