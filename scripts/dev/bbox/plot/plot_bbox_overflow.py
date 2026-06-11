"""
Blender bbox はみ出しの上位セッション棒グラフ (plot)。

run/analyze_bbox_overflow.py が出力した bbox_overflow.csv を読み、
上下 (z) / 水平 (x,y) の平均はみ出しが大きい上位セッションを横棒で描く。

入力 (--input-dir, 既定 scripts/dev/bbox_files):  bbox_overflow.csv
出力 (--output-dir, 既定 scripts/dev/bbox_plot):  bbox_overflow_top.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT  = Path("scripts/dev/bbox_files")
DEFAULT_OUTPUT = Path("scripts/dev/bbox_plot")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help="bbox_overflow.csv の場所")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="(互換用; 上位ランキングのため未使用)")
    ap.add_argument("--top", type=int, default=25, help="表示件数")
    args = ap.parse_args()

    csv = args.input_dir / "bbox_overflow.csv"
    if not csv.exists():
        print(f"{csv} がありません。先に run/analyze_bbox_overflow.py を実行してください。",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, args.top * 0.28)))
    for ax, key, label, color in [
        (axes[0], "vz_mean_over", "Vertical (z) mean overflow [m]", "#c0392b"),
        (axes[1], "h_mean_over", "Horizontal (x,y) mean overflow [m]", "#2980b9"),
    ]:
        d = df.sort_values(key, ascending=True).tail(args.top)
        ax.barh(d["session"], d[key], color=color)
        ax.set_xlabel(label); ax.tick_params(axis="y", labelsize=7); ax.set_title(label)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "bbox_overflow_top.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
