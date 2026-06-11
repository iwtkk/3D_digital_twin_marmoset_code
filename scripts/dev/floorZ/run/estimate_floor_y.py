"""
床接地推定に基づく per-session Y オフセット補正 (run)。

kp3d の Y 軸（大きい=床側 / 小さい=天井側）で動作する旧手法。
各 session で床接地 KP の max Y の p99 を床レベル推定値とし、
グローバル床レベルとの差を補正量とする。

判定:
  p99 >= FLOOR_Y_CONTACT_THR (-50mm) → 床接触あり (contact)
  p99 <  FLOOR_Y_CONTACT_THR          → 床接触なし (no_contact, 要フラグ)

入力 (--input-dir, 既定 outputs/3d):
  {input-dir}/{session}/kp3d_fxdJointLen.pickle  (無ければ kp3d.pickle)

出力 (--output-dir, 既定 scripts/dev/floorZ_files):
  {output-dir}/y_corrections.csv

可視化は plot/plot_floor_y.py が同じ CSV から行う。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_files")

FLOOR_KP            = core.FLOOR_KP_IDX     # L/R wrist, L/R ankle
FLOOR_Y_CONTACT_THR = -50.0                 # mm: p99 がこれ未満なら床接触なし
GLOBAL_FLOOR_Y      = 92.27                 # mm: bbox inlier 上限 ≈ 実際の床


def estimate_floor_y(kp3d: np.ndarray, percentile: float = 99.0) -> float:
    """kp3d (T,n_kp,3) から床高さ(Y)を推定。各フレーム max(floor KP Y) の p99。"""
    return float(np.percentile(kp3d[:, FLOOR_KP, 1].max(axis=1), percentile))


def analyze(input_dir: Path, session: str | None) -> pd.DataFrame:
    if session:
        dirs = [input_dir / session]
    else:
        dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    records = []
    for d in dirs:
        pkl = core.resolve_pickle(d)
        if pkl is None:
            print(f"  [SKIP] pickle not found: {d}", file=sys.stderr)
            continue
        with open(pkl, "rb") as f:
            kp3d = pickle.load(f)["kp3d"][0]   # (T,n_kp,3)

        p99    = estimate_floor_y(kp3d, 99.0)
        median = float(np.median(kp3d[:, FLOOR_KP, 1].max(axis=1)))
        classification = "contact" if p99 >= FLOOR_Y_CONTACT_THR else "no_contact"
        records.append({
            "session": d.name, "animal": d.name.split("_")[0],
            "n_frames": kp3d.shape[0],
            "floor_y_p99": round(p99, 2), "floor_y_median": round(median, 2),
            "classification": classification,
            "y_correction_mm": round(GLOBAL_FLOOR_Y - p99, 2),
        })
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"pickle のルート (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"出力ルート (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="セッション名 (省略時は全セッション)")
    args = ap.parse_args()

    print("Analysing sessions ...")
    df = analyze(args.input_dir, args.session)
    if df.empty:
        print("有効なセッションがありません", file=sys.stderr)
        sys.exit(1)

    n_contact    = int((df["classification"] == "contact").sum())
    n_no_contact = int((df["classification"] == "no_contact").sum())
    print(f"\n{'Sessions analysed':25}: {len(df)}")
    print(f"{'  floor contact':25}: {n_contact}")
    print(f"{'  no floor contact':25}: {n_no_contact}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "y_corrections.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    no_contact = df[df["classification"] == "no_contact"]
    if len(no_contact):
        print(f"\n[WARNING] {len(no_contact)} sessions flagged 'no floor contact':")
        for _, row in no_contact.iterrows():
            print(f"  {row['session']}  floor_y_p99={row['floor_y_p99']:.1f}mm")


if __name__ == "__main__":
    main()
