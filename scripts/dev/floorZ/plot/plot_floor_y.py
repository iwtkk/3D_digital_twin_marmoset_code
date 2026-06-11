"""
床接地 Y 補正の可視化 (plot)。

run/estimate_floor_y.py が出力した y_corrections.csv を読み、
(1) per-session の floor_y_p99 / median と補正量サマリー
(2) 代表セッションの床 KP Y 分布 before/after
を描く。before/after の生分布は pickle から再計算する。

入力:
  --files-dir (既定 scripts/dev/floorZ_files): y_corrections.csv の場所
  --input-dir (既定 outputs/3d):               pickle のルート
出力 (--output-dir, 既定 scripts/dev/floorZ_plot):
  floor_estimate_summary.png
  y_correction_per_session.png
  floor_y_before_after.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core            # noqa: E402
import estimate_floor_y as efy        # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_FILES  = Path("scripts/dev/floorZ_files")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_plot")


def plot_summary(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, col, title in [
        (axes[0], "floor_y_p99", "Floor Y estimate (p99 of max floor-KP Y)"),
        (axes[1], "floor_y_median", "Floor Y median (typical height of floor KPs)"),
    ]:
        colors = [core.ANIMAL_COLORS.get(a, "gray") for a in df["animal"]]
        ax.bar(range(len(df)), df[col], color=colors, alpha=0.8)
        ax.axhline(efy.GLOBAL_FLOOR_Y, color="brown", lw=2, ls="--",
                   label=f"global floor ({efy.GLOBAL_FLOOR_Y}mm)")
        ax.axhline(efy.FLOOR_Y_CONTACT_THR, color="red", lw=1.5, ls=":",
                   label=f"no-contact threshold ({efy.FLOOR_Y_CONTACT_THR}mm)")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["session"].str[3:], rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("kp3d Y (mm)"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        for i, row in df.reset_index(drop=True).iterrows():
            if row["classification"] == "no_contact":
                ax.annotate("!", (i, df[col].iloc[i]), ha="center", va="bottom",
                            fontsize=8, color="red", fontweight="bold")
    fig.suptitle("Per-session floor estimation & Y-correction", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "floor_estimate_summary.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 4))
    colors = [core.ANIMAL_COLORS.get(a, "gray") for a in df["animal"]]
    ax.bar(range(len(df)), df["y_correction_mm"], color=colors, alpha=0.8)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["session"].str[3:], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Y correction (mm)")
    ax.set_title("Per-session Y correction (= GLOBAL_FLOOR_Y - session floor_y_p99)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "y_correction_per_session.png", dpi=150)
    plt.close(fig)
    print(f"Saved summary plots to {out_dir}")


def plot_before_after(df: pd.DataFrame, input_dir: Path, out_dir: Path):
    showcase = [s for s in ["cj425_20210427_190000", "cj425_20210426_141115",
                            "cj611_20230217_170000", "cj611_20210831_080000"]
                if (input_dir / s).exists() and (df["session"] == s).any()]
    if not showcase:
        showcase = df["session"].head(4).tolist()
    n = len(showcase)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    axes = np.atleast_1d(axes)
    for ax, session in zip(axes, showcase):
        row = df[df["session"] == session].iloc[0]
        pkl = core.resolve_pickle(input_dir / session)
        if pkl is None:
            continue
        with open(pkl, "rb") as f:
            kp3d = pickle.load(f)["kp3d"][0]
        max_y = kp3d[:, efy.FLOOR_KP, 1].max(axis=1)
        max_y_corr = max_y + row["y_correction_mm"]
        ax.hist(max_y, bins=100, alpha=0.5, color="tomato", density=True,
                label=f"before median={np.median(max_y):.0f}mm")
        ax.hist(max_y_corr, bins=100, alpha=0.5, color="steelblue", density=True,
                label=f"after median={np.median(max_y_corr):.0f}mm")
        ax.axvline(efy.GLOBAL_FLOOR_Y, color="brown", ls="--", lw=2,
                   label=f"floor target ({efy.GLOBAL_FLOOR_Y}mm)")
        ax.set_title(f"{session}  correction={row['y_correction_mm']:.1f}mm "
                     f"[{row['classification']}]")
        ax.set_xlabel("max floor-KP Y per frame (mm)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "floor_y_before_after.png", dpi=150)
    plt.close(fig)
    print(f"Saved before/after plot to {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"pickle のルート (既定: {DEFAULT_INPUT})")
    ap.add_argument("--files-dir", type=Path, default=DEFAULT_FILES,
                    help=f"y_corrections.csv の場所 (既定: {DEFAULT_FILES})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"図の出力先 (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="セッション名 (省略時は全件のサマリー)")
    args = ap.parse_args()

    csv = args.files_dir / "y_corrections.csv"
    if not csv.exists():
        print(f"{csv} がありません。先に run/estimate_floor_y.py を実行してください。",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv)
    if args.session:
        df = df[df["session"] == args.session]
        if df.empty:
            print(f"{args.session} が y_corrections.csv にありません", file=sys.stderr)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(df, args.output_dir)
    plot_before_after(df, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
