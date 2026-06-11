"""
垂直 affine 補正の検証 (run): scaletest CSV から session ごとの affine を算出。

入力 (--input-dir, 既定 outputs/animate_temp/test_scale):
  {input-dir}/*_scaletest.csv  (bZ は Blender メートル空間, affine 未適用)

出力 (--output-dir, 既定 scripts/dev/floorZ_files):
  {output-dir}/affine_summary.csv  -- session × (n_peaks, peaks, slope, intercept,
                                       status, slope_warn, floor_residual_mm)

可視化は plot/plot_affine_validation.py が同じ入力から行う。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("outputs/animate_temp/test_scale")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_files")


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

    floor_residual = None
    if len(peaks) and status != "no_peaks":
        floor_residual = round((slope * peaks[0] + intercept - core.PHYSICAL_HEIGHTS_M[0]) * 1000, 1)

    return {
        "session": session, "animal": animal, "n_frames": len(df),
        "n_peaks": len(peaks), "peaks_m": np.round(peaks, 4).tolist(),
        "slope": round(slope, 4), "intercept": round(intercept, 4),
        "status": status, "slope_warn": core.slope_warn(slope),
        "floor_residual_mm": floor_residual,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"scaletest CSV のディレクトリ (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"出力ルート (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="セッション名 (前方一致) で 1 件に絞る")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*_scaletest.csv"))
    if args.session:
        files = [p for p in files if p.name.startswith(args.session)]
    print(f"Processing {len(files)} CSV files ...")

    records = [process_csv(p) for p in files]
    df = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "affine_summary.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")

    valid = df[df["status"].notna() & df["status"].str.contains("affine|offset")]
    print(f"\n{'Status':25} {'Count':>6}")
    for status, grp in valid.groupby("status"):
        print(f"  {status:23} {len(grp):>6}")
    if len(valid):
        n_warn = int(valid["slope_warn"].sum())
        print(f"\nSlope warnings (|a-1|>{core.MAX_AFFINE_SLOPE_DEV}): {n_warn} "
              f"({100 * n_warn / len(valid):.0f}%)")
        print(f"Slope mean={valid['slope'].mean():.4f} std={valid['slope'].std():.4f}")
        fr = valid["floor_residual_mm"].dropna()
        if len(fr):
            print(f"Floor residual (mm) mean={fr.abs().mean():.2f} max={fr.abs().max():.2f}")


if __name__ == "__main__":
    main()
