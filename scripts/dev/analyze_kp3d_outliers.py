"""
3D keypoint positional outlier analysis.

Flags (frame, kp) pairs where kp3d position falls outside the
global inlier bounding box (2nd–98th percentile bounds computed
across all 222 sessions; see analysis_log.md section 2).

"95th percentile outliers" here means the ~10 % of (frame, kp)
values that fall outside the global 2–98 % bounding box.

Outputs → outputs/analysis_kp3d_outliers/
  {session}_outlier.json  — per-session events  (same format as
                            analysis_teleport/{session}_teleport.json)
  outlier_summary.json    — all-session summary
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
FPS_3D = 24.0
FPS_2D = 25.0

# Global inlier bbox: 2nd–98th percentile across all 222 sessions
# (analysis_log.md section 2)
BBOX_X = (681.33, 1580.68)   # mm
BBOX_Y = (-665.49, 92.27)    # mm
BBOX_Z = (633.38, 1512.50)   # mm

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
N_KP = 20

BASE_3D = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/3d")
OUT_DIR = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/analysis_kp3d_outliers")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── per-session analysis ───────────────────────────────────────────────────────
def analyze_session(session: str) -> dict | None:
    pkl_path = BASE_3D / session / "kp3d_fxdJointLen.pickle"
    if not pkl_path.exists():
        return None

    with open(pkl_path, "rb") as f:
        data3d = pickle.load(f)

    kp   = data3d["kp3d"][0]        # (T, 20, 3)
    sc3d = data3d["kp3d_score"][0]  # (T, 20)
    er3d = data3d["kp3d_err"][0]    # (T, 20)
    T    = kp.shape[0]

    # outlier mask per axis
    out_x = (kp[:, :, 0] < BBOX_X[0]) | (kp[:, :, 0] > BBOX_X[1])  # (T, 20)
    out_y = (kp[:, :, 1] < BBOX_Y[0]) | (kp[:, :, 1] > BBOX_Y[1])
    out_z = (kp[:, :, 2] < BBOX_Z[0]) | (kp[:, :, 2] > BBOX_Z[1])
    out_any = out_x | out_y | out_z  # (T, 20)

    if not out_any.any():
        return None

    # per-axis signed deviation beyond the bbox boundary (0 if inside)
    def excess(vals, lo, hi):
        below = np.minimum(vals - lo, 0.0)  # negative when below lo
        above = np.maximum(vals - hi, 0.0)  # positive when above hi
        return below + above                 # non-zero only when outside

    dev_x = excess(kp[:, :, 0], *BBOX_X)
    dev_y = excess(kp[:, :, 1], *BBOX_Y)
    dev_z = excess(kp[:, :, 2], *BBOX_Z)
    # Euclidean deviation magnitude
    dev_mag = np.sqrt(dev_x**2 + dev_y**2 + dev_z**2)  # (T, 20)

    # collect events
    t_frames, ki_arr = np.where(out_any)

    events = []
    for t, ki in zip(t_frames.tolist(), ki_arr.tolist()):
        axes = []
        if out_x[t, ki]:
            axes.append("X")
        if out_y[t, ki]:
            axes.append("Y")
        if out_z[t, ki]:
            axes.append("Z")

        sc  = float(sc3d[t, ki])
        err = float(er3d[t, ki])

        events.append({
            "frame_3d":      t,
            "frame_2d":      round(t * FPS_2D / FPS_3D),
            "kp_idx":        ki,
            "kp_name":       KP_NAMES[ki],
            "pos_xyz":       [round(float(kp[t, ki, j]), 2) for j in range(3)],
            "outlier_axes":  axes,
            "deviation_mm":  round(float(dev_mag[t, ki]), 2),
            "deviation_vec": [
                round(float(dev_x[t, ki]), 2),
                round(float(dev_y[t, ki]), 2),
                round(float(dev_z[t, ki]), 2),
            ],
            "score3d":       round(sc, 4)  if np.isfinite(sc)  else None,
            "reproj_err_px": round(err, 4) if np.isfinite(err) else None,
        })

    # sort by deviation magnitude descending
    events.sort(key=lambda e: e["deviation_mm"], reverse=True)

    # per-kp summaries
    kp_counts  = np.zeros(N_KP, dtype=int)
    kp_max_dev = np.zeros(N_KP)
    for ev in events:
        ki = ev["kp_idx"]
        kp_counts[ki] += 1
        kp_max_dev[ki] = max(kp_max_dev[ki], ev["deviation_mm"])

    # position stats for this session
    pos_flat = kp.reshape(-1, 3)
    x_p5, x_p95 = float(np.percentile(kp[:, :, 0], 5)), float(np.percentile(kp[:, :, 0], 95))
    y_p5, y_p95 = float(np.percentile(kp[:, :, 1], 5)), float(np.percentile(kp[:, :, 1], 95))
    z_p5, z_p95 = float(np.percentile(kp[:, :, 2], 5)), float(np.percentile(kp[:, :, 2], 95))

    return {
        "session":            session,
        "n_frames_3d":        T,
        "fps_3d":             FPS_3D,
        "fps_2d":             FPS_2D,
        # bbox used as inlier boundary
        "bbox_x_mm":          list(BBOX_X),
        "bbox_y_mm":          list(BBOX_Y),
        "bbox_z_mm":          list(BBOX_Z),
        # session-level position stats
        "pos_x_p05_p95_mm":  [round(x_p5, 2), round(x_p95, 2)],
        "pos_y_p05_p95_mm":  [round(y_p5, 2), round(y_p95, 2)],
        "pos_z_p05_p95_mm":  [round(z_p5, 2), round(z_p95, 2)],
        # event counts
        "n_outlier_events":   int(out_any.sum()),
        "n_outlier_frames":   int(out_any.any(axis=1).sum()),
        "outlier_frame_rate": round(float(out_any.any(axis=1).mean()), 4),
        # per-kp breakdown (mirrors kp_teleport_counts / kp_max_velocity_mm_s)
        "kp_outlier_counts":  {KP_NAMES[i]: int(kp_counts[i]) for i in range(N_KP)},
        "kp_max_deviation_mm": {KP_NAMES[i]: round(float(kp_max_dev[i]), 2) for i in range(N_KP)},
        # full event list (sorted by deviation descending)
        "outlier_events":     events,
    }


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    sessions = sorted(p.name for p in BASE_3D.iterdir() if p.is_dir())
    print(f"Analyzing {len(sessions)} sessions …")

    all_summaries = []

    for i, session in enumerate(sessions):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sessions)}] {session}")
        result = analyze_session(session)
        if result is None:
            continue

        out_json = OUT_DIR / f"{session}_outlier.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        summary = {k: v for k, v in result.items() if k != "outlier_events"}
        all_summaries.append(summary)

    print(f"\nSessions with outliers: {len(all_summaries)} / {len(sessions)}")

    summary_out = {
        "bbox_x_mm":  list(BBOX_X),
        "bbox_y_mm":  list(BBOX_Y),
        "bbox_z_mm":  list(BBOX_Z),
        "fps_3d":     FPS_3D,
        "fps_2d":     FPS_2D,
        "n_sessions_total":        len(sessions),
        "n_sessions_with_outliers": len(all_summaries),
        "sessions": all_summaries,
    }
    with open(OUT_DIR / "outlier_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    print("  saved outlier_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
