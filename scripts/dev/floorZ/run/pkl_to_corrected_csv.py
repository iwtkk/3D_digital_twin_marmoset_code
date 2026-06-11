"""
pickle → KDE affine 補正済み Blender CSV (run)

各フレームの床 KP (wrist/ankle) min(bZ) 分布から KDE ピークを検出し、
物理高さ PHYSICAL_HEIGHTS_M に affine (a, b) でフィットして bZ を補正する。
アルゴリズム本体は floorz_core.py に集約。

入力 (--input-dir, 既定 outputs/3d):
  {input-dir}/{session}/kp3d_fxdJointLen.pickle  (無ければ kp3d.pickle)

出力 (--output-dir, 既定 scripts/dev/floorZ_files):
  {output-dir}/csv/{session}_points.csv          中間 CSV (rotated mm)
  {output-dir}/corrected/{session}_corrected.csv 補正済み CSV
  {output-dir}/affine_log.csv                    session ごとの (a, b, status)

使い方:
  OPENBLAS_NUM_THREADS=1 python pkl_to_corrected_csv.py --session cj425_20210427_120000
  OPENBLAS_NUM_THREADS=1 python pkl_to_corrected_csv.py            # 全セッション
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_files")

# 出力 CSV の座標スケール（全軸に乗算）。1.0=Blenderメートル, 1/1000=mm相当
OUTPUT_SCALE = 1.0


def process_session(session: str, input_dir: Path, csv_dir: Path,
                    corr_dir: Path, n_frames: int = -1) -> dict:
    pkl = core.resolve_pickle(input_dir / session)
    if pkl is None:
        raise FileNotFoundError(f"pickle not found: {input_dir / session}")

    df_raw = core.load_pickle_to_df(pkl, n_frames)
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(csv_dir / f"{session}_points.csv", index=False)

    df_corr, a, b, status = core.scale_and_correct(df_raw, output_scale=OUTPUT_SCALE)
    corr_dir.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(corr_dir / f"{session}_corrected.csv", index=False)

    warn = core.slope_warn(a)
    print(f"  [{status}] a={a:.4f} b={b:.4f}" + (" WARN: |a-1|>0.15" if warn else ""))
    return {
        "session": session, "animal": session.split("_")[0], "n_frames": len(df_raw),
        "slope": round(a, 4), "intercept": round(b, 4),
        "status": status, "slope_warn": warn,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"pickle のルート (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"出力ルート (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="セッション名 (省略時は input-dir 以下の全セッション)")
    ap.add_argument("--n-frames", type=int, default=-1, help="処理フレーム数 (-1=全)")
    args = ap.parse_args()

    csv_dir  = args.output_dir / "csv"
    corr_dir = args.output_dir / "corrected"
    log_path = args.output_dir / "affine_log.csv"

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(d.name for d in args.input_dir.iterdir() if d.is_dir())
    print(f"{len(sessions)} セッションを処理します")

    records, skipped = [], []
    for session in sessions:
        print(f"\n=== {session} ===")
        try:
            records.append(process_session(session, args.input_dir, csv_dir, corr_dir, args.n_frames))
        except Exception as e:
            print(f"  [SKIP] {e}", file=sys.stderr)
            skipped.append(session)

    if records:
        df = pd.DataFrame(records)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(log_path, index=False)
        print(f"\naffine log: {log_path}")
        print(f"\n{'Status':25} {'Count':>6}")
        for s, grp in df.groupby("status"):
            print(f"  {s:23} {len(grp):>6}")
        n_warn = int(df["slope_warn"].sum())
        print(f"\nSlope warnings: {n_warn} ({100 * n_warn / len(df):.0f}%)")
        print(f"Slope  mean={df['slope'].mean():.4f} std={df['slope'].std():.4f} "
              f"min={df['slope'].min():.4f} max={df['slope'].max():.4f}")
    if skipped:
        print(f"\n{len(skipped)} セッションをスキップ: {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
