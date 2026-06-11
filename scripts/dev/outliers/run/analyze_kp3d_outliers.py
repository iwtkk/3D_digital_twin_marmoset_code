"""
3D キーポイントの位置外れ値解析 (run)。

kp3d 位置がグローバル inlier bbox（全 222 セッションの 2–98 パーセンタイル境界,
analysis_log.md 参照）の外に出る (frame, kp) を抽出して JSON 出力する。
可視化は plot/plot_outlier_deviation_dist.py。

入力 (--input-dir, 既定 outputs/3d):  {input-dir}/{session}/kp3d_fxdJointLen.pickle
出力 (--output-dir, 既定 scripts/dev/outliers_files):
  {session}_outlier.json   per-session events (deviation 降順)
  outlier_summary.json     全セッション要約
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/outliers_files")

FPS_3D, FPS_2D = 24.0, 25.0
# グローバル inlier bbox: 全 222 セッションの 2–98 パーセンタイル (mm)
BBOX_X = (681.33, 1580.68)
BBOX_Y = (-665.49, 92.27)
BBOX_Z = (633.38, 1512.50)

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
N_KP = 20


def _excess(vals, lo, hi):
    return np.minimum(vals - lo, 0.0) + np.maximum(vals - hi, 0.0)


def resolve_pickle(session_dir: Path) -> Path | None:
    for name in ("kp3d_fxdJointLen.pickle", "kp3d.pickle"):
        p = session_dir / name
        if p.exists():
            return p
    return None


def analyze_session(session: str, input_dir: Path) -> dict | None:
    pkl = resolve_pickle(input_dir / session)
    if pkl is None:
        return None
    with open(pkl, "rb") as f:
        data3d = pickle.load(f)
    kp   = data3d["kp3d"][0]
    sc3d = data3d["kp3d_score"][0]
    er3d = data3d["kp3d_err"][0]
    T = kp.shape[0]

    out_x = (kp[:, :, 0] < BBOX_X[0]) | (kp[:, :, 0] > BBOX_X[1])
    out_y = (kp[:, :, 1] < BBOX_Y[0]) | (kp[:, :, 1] > BBOX_Y[1])
    out_z = (kp[:, :, 2] < BBOX_Z[0]) | (kp[:, :, 2] > BBOX_Z[1])
    out_any = out_x | out_y | out_z
    if not out_any.any():
        return None

    dev_x = _excess(kp[:, :, 0], *BBOX_X)
    dev_y = _excess(kp[:, :, 1], *BBOX_Y)
    dev_z = _excess(kp[:, :, 2], *BBOX_Z)
    dev_mag = np.sqrt(dev_x**2 + dev_y**2 + dev_z**2)

    events = []
    for t, ki in zip(*[a.tolist() for a in np.where(out_any)]):
        axes = [a for a, m in zip("XYZ", (out_x[t, ki], out_y[t, ki], out_z[t, ki])) if m]
        sc, err = float(sc3d[t, ki]), float(er3d[t, ki])
        events.append({
            "frame_3d": t, "frame_2d": round(t * FPS_2D / FPS_3D),
            "kp_idx": ki, "kp_name": KP_NAMES[ki],
            "pos_xyz": [round(float(kp[t, ki, j]), 2) for j in range(3)],
            "outlier_axes": axes, "deviation_mm": round(float(dev_mag[t, ki]), 2),
            "deviation_vec": [round(float(dev_x[t, ki]), 2), round(float(dev_y[t, ki]), 2),
                              round(float(dev_z[t, ki]), 2)],
            "score3d": round(sc, 4) if np.isfinite(sc) else None,
            "reproj_err_px": round(err, 4) if np.isfinite(err) else None,
        })
    events.sort(key=lambda e: e["deviation_mm"], reverse=True)

    kp_counts = np.zeros(N_KP, dtype=int)
    kp_max_dev = np.zeros(N_KP)
    for ev in events:
        kp_counts[ev["kp_idx"]] += 1
        kp_max_dev[ev["kp_idx"]] = max(kp_max_dev[ev["kp_idx"]], ev["deviation_mm"])

    return {
        "session": session, "n_frames_3d": T, "fps_3d": FPS_3D, "fps_2d": FPS_2D,
        "bbox_x_mm": list(BBOX_X), "bbox_y_mm": list(BBOX_Y), "bbox_z_mm": list(BBOX_Z),
        "pos_x_p05_p95_mm": [round(float(np.percentile(kp[:, :, 0], 5)), 2), round(float(np.percentile(kp[:, :, 0], 95)), 2)],
        "pos_y_p05_p95_mm": [round(float(np.percentile(kp[:, :, 1], 5)), 2), round(float(np.percentile(kp[:, :, 1], 95)), 2)],
        "pos_z_p05_p95_mm": [round(float(np.percentile(kp[:, :, 2], 5)), 2), round(float(np.percentile(kp[:, :, 2], 95)), 2)],
        "n_outlier_events": int(out_any.sum()),
        "n_outlier_frames": int(out_any.any(axis=1).sum()),
        "outlier_frame_rate": round(float(out_any.any(axis=1).mean()), 4),
        "kp_outlier_counts": {KP_NAMES[i]: int(kp_counts[i]) for i in range(N_KP)},
        "kp_max_deviation_mm": {KP_NAMES[i]: round(float(kp_max_dev[i]), 2) for i in range(N_KP)},
        "outlier_events": events,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="セッション名 (省略時は全セッション)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(p.name for p in args.input_dir.iterdir() if p.is_dir())
    print(f"Analyzing {len(sessions)} sessions …")

    summaries = []
    for i, session in enumerate(sessions):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sessions)}] {session}")
        result = analyze_session(session, args.input_dir)
        if result is None:
            continue
        with open(args.output_dir / f"{session}_outlier.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        summaries.append({k: v for k, v in result.items() if k != "outlier_events"})

    print(f"\nSessions with outliers: {len(summaries)} / {len(sessions)}")
    with open(args.output_dir / "outlier_summary.json", "w") as f:
        json.dump({
            "bbox_x_mm": list(BBOX_X), "bbox_y_mm": list(BBOX_Y), "bbox_z_mm": list(BBOX_Z),
            "fps_3d": FPS_3D, "fps_2d": FPS_2D,
            "n_sessions_total": len(sessions), "n_sessions_with_outliers": len(summaries),
            "sessions": summaries,
        }, f, indent=2, ensure_ascii=False)
    print(f"  saved {args.output_dir / 'outlier_summary.json'}")


if __name__ == "__main__":
    main()
