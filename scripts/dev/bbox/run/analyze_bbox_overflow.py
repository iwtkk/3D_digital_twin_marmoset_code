"""
補正前 transformedcsv の Blender bbox はみ出し分析 (run)。

global scale のみ適用済みの Blender 座標が bbox からどれだけはみ出すかをセッション別に
集計し、CSV 出力＋ランキング表示する。上下 (z) と水平 (x,y) を分ける。可視化は
plot/plot_bbox_overflow.py。

入力 (--input-dir, 既定 ../ez2use/outputs/animate_temp/transformedcsv):
  {input-dir}/*_points.csv_transformedtest.csv  (Blender メートル, affine 未適用)
出力 (--output-dir, 既定 scripts/dev/bbox_files):
  bbox_overflow.csv  セッションごとの全指標 (vertical mean over 降順)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT  = Path("../ez2use/outputs/animate_temp/transformedcsv")
DEFAULT_OUTPUT = Path("scripts/dev/bbox_files")
SUFFIX = "_points.csv_transformedtest.csv"

BLENDER_BBOX_SCALE = np.array([0.86658, 0.88791, 0.7679])
BLENDER_CENTER     = np.array([0.0, -0.11142, 1.11765])
BBOX_LO = BLENDER_CENTER - BLENDER_BBOX_SCALE / 2
BBOX_HI = BLENDER_CENTER + BLENDER_BBOX_SCALE / 2


def session_from_path(path: Path) -> str:
    name = path.name
    return name[:-len(SUFFIX)] if name.endswith(SUFFIX) else name


def coord_cols(df, axis):
    return [c for c in df.columns if c.startswith("pts_") and c.endswith(f"_{axis}")]


def overflow_stats(vals, lo, hi) -> dict:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(frac_out=0.0, frac_below=0.0, frac_above=0.0, mean_over=0.0,
                    p99_over=0.0, max_over=0.0, max_below=0.0, max_above=0.0)
    over_below = np.clip(lo - vals, 0.0, None)
    over_above = np.clip(vals - hi, 0.0, None)
    over = over_below + over_above
    out_mask = over > 0
    over_out = over[out_mask]
    return dict(
        frac_out=float(out_mask.mean()), frac_below=float((over_below > 0).mean()),
        frac_above=float((over_above > 0).mean()), mean_over=float(over.mean()),
        p99_over=float(np.percentile(over_out, 99)) if over_out.size else 0.0,
        max_over=float(over.max()), max_below=float(over_below.max()),
        max_above=float(over_above.max()))


def analyze_session(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    x = df[coord_cols(df, "x")].to_numpy(float).ravel()
    y = df[coord_cols(df, "y")].to_numpy(float).ravel()
    z = df[coord_cols(df, "z")].to_numpy(float).ravel()
    v = overflow_stats(z, BBOX_LO[2], BBOX_HI[2])
    sx = overflow_stats(x, BBOX_LO[0], BBOX_HI[0])
    sy = overflow_stats(y, BBOX_LO[1], BBOX_HI[1])
    ho = np.concatenate([
        np.clip(BBOX_LO[0] - x, 0, None) + np.clip(x - BBOX_HI[0], 0, None),
        np.clip(BBOX_LO[1] - y, 0, None) + np.clip(y - BBOX_HI[1], 0, None)])
    ho = ho[np.isfinite(ho)]
    h_out = ho > 0
    return {
        "session": session_from_path(csv_path), "animal": session_from_path(csv_path).split("_")[0],
        "n_rows": len(df),
        "vz_frac_out": v["frac_out"], "vz_frac_below": v["frac_below"], "vz_frac_above": v["frac_above"],
        "vz_mean_over": v["mean_over"], "vz_p99_over": v["p99_over"],
        "vz_max_below": v["max_below"], "vz_max_above": v["max_above"],
        "h_frac_out": float(h_out.mean()), "h_mean_over": float(ho.mean()),
        "h_p99_over": float(np.percentile(ho[h_out], 99)) if h_out.any() else 0.0,
        "h_max_over": float(ho.max()) if ho.size else 0.0,
        "hx_frac_out": sx["frac_out"], "hx_max_over": sx["max_over"],
        "hy_frac_out": sy["frac_out"], "hy_max_over": sy["max_over"],
    }


def print_ranking(df, sort_key, cols, title, top):
    print(f"\n{'='*78}\n{title}  (sort: {sort_key} 降順, top {top})\n{'='*78}")
    ranked = df.sort_values(sort_key, ascending=False).head(top)
    header = f"{'#':>3} {'session':<28} " + " ".join(f"{c:>11}" for c in cols)
    print(header); print("-" * len(header))
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        cells = " ".join(f"{row[c]:>11.4f}" for c in cols)
        print(f"{i:>3} {row['session']:<28} {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"transformedcsv ディレクトリ (default: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"出力先 (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None, help="単一セッションのみ集計")
    ap.add_argument("--top", type=int, default=25, help="ランキング表示件数")
    args = ap.parse_args()

    files = sorted(Path(p) for p in glob.glob(str(args.input_dir / f"*{SUFFIX}")))
    if args.session:
        files = [f for f in files if session_from_path(f) == args.session]
    if not files:
        print(f"CSV が見つかりません: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"bbox  x:[{BBOX_LO[0]:.5f}, {BBOX_HI[0]:.5f}]  "
          f"y:[{BBOX_LO[1]:.5f}, {BBOX_HI[1]:.5f}]  z:[{BBOX_LO[2]:.5f}, {BBOX_HI[2]:.5f}]")
    print(f"{len(files)} セッションを集計します\n")

    records = []
    for i, f in enumerate(files, 1):
        try:
            rec = analyze_session(f)
            records.append(rec)
            print(f"[{i:>3}/{len(files)}] {rec['session']:<28} "
                  f"v_out={rec['vz_frac_out']*100:5.1f}%  h_out={rec['h_frac_out']*100:5.1f}%")
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}", file=sys.stderr)
    if not records:
        return

    df = pd.DataFrame(records).sort_values("vz_mean_over", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "bbox_overflow.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n保存: {out_csv}")

    print_ranking(df, "vz_mean_over",
                  ["vz_frac_out", "vz_frac_below", "vz_frac_above", "vz_mean_over", "vz_max_below", "vz_max_above"],
                  "■ 上下方向 (z) はみ出し", args.top)
    print_ranking(df, "h_mean_over",
                  ["h_frac_out", "h_mean_over", "h_p99_over", "h_max_over", "hx_frac_out", "hy_frac_out"],
                  "■ 水平方向 (x,y) はみ出し", args.top)


if __name__ == "__main__":
    main()
