"""
per_session 水平アライメントを適用した Blender座標 CSV を全フレーム書き出す（Blender取込用）。

floorz の scale_and_correct（回転+global scale+per-session 垂直affine）に、
align_log.csv の per_session 水平補正 (s,tx,ty) を bX,bY へ重ねる。bZ は floorz のまま。
出力列は floorz corrected CSV と同形式（pts_<key>_x/y/z + カメラ姿勢 euler）。

実行:
  python3 scripts/05_create_digital_twin/alignment_3D_keypoint/run/export_aligned_csv.py --session cj357_20221009_090000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))
from align import load_align_log     # noqa: E402
import floorz_core as fz             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--input-dir", default=str(ROOT / "outputs/3d"))
    ap.add_argument("--align-log", default=str(ROOT / "outputs/reproj_eval/align_log.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval/aligned_csv"))
    args = ap.parse_args()

    pp = fz.resolve_pickle(Path(args.input_dir) / args.session)
    if pp is None:
        ap.error(f"pickle が見つかりません: {args.session}")
    df = fz.load_pickle_to_df(pp)              # 全フレーム
    dfc, a, b, status = fz.scale_and_correct(df)

    params = load_align_log(args.align_log).get(args.session)
    if params is None:
        ap.error(f"{args.session} が align_log に無い。先に align.py を実行してください")
    s, tx, ty = params
    for col in dfc.columns:
        if col.endswith("_x"):
            dfc[col] = s * dfc[col] + tx
        elif col.endswith("_y"):
            dfc[col] = s * dfc[col] + ty
    # bZ(_z) と カメラ姿勢(_rx/_ry/_rz) は不変

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.session}_aligned.csv"
    dfc.to_csv(out, index=False)
    print(f"wrote {out}")
    print(f"  frames={len(dfc)} cols={dfc.shape[1]}  "
          f"per_session (s,tx,ty)=({s:.4f},{tx:.4f},{ty:.4f})  "
          f"vertical affine a={a:.4f} b={b:.4f} ({status})")


if __name__ == "__main__":
    main()
