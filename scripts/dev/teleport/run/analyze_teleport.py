"""
3D キーポイントのテレポート解析 (run, v1)。

各セッションで > THRESHOLD_MM/frame のジャンプを検出し、跳んだ KP・速度・加速度・
2D カメラ可視性を集計して JSON 出力する。可視化は plot/plot_teleport.py。

入力:
  --input-dir (既定 outputs/3d):  {input-dir}/{session}/kp3d_*.pickle
  --d2d-dir   (既定 outputs/2d):  2D 検出 JSON のルート
出力 (--output-dir, 既定 scripts/dev/teleport_files):
  {session}_teleport.json   per-session 詳細
  teleport_summary.json     全セッション要約
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import teleport_common as tc  # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_D2D    = Path("outputs/2d")
DEFAULT_OUTPUT = Path("scripts/dev/teleport_files")

THRESHOLD_MM = 100.0


def analyze_session(session: str, input_dir: Path, d2d_dir: Path) -> dict | None:
    pkl = tc.resolve_pickle(input_dir / session)
    if pkl is None:
        return None
    with open(pkl, "rb") as f:
        data3d = pickle.load(f)
    kp   = data3d["kp3d"][0]
    sc3d = data3d["kp3d_score"][0]
    er3d = data3d["kp3d_err"][0]
    T = kp.shape[0]

    disp = np.diff(kp, axis=0)
    dist = np.linalg.norm(disp, axis=-1)
    vel  = dist / tc.DT
    vel_vec = disp / tc.DT
    acc_vec = np.diff(vel_vec, axis=0)
    acc_mag = np.linalg.norm(acc_vec, axis=-1)
    jerk_vec = np.diff(acc_vec, axis=0)
    jerk_mag = np.linalg.norm(jerk_vec, axis=-1)

    tel_mask = dist > THRESHOLD_MM
    tel_frames, tel_kps = np.where(tel_mask)
    if len(tel_frames) == 0:
        return None

    cc = tc.ChunkCache(session, d2d_dir)
    has_2d = len(cc.cameras) > 0

    events = []
    for t_disp, ki in zip(tel_frames.tolist(), tel_kps.tolist()):
        t_before, t_after = t_disp, t_disp + 1
        acc_before = float(acc_mag[max(0, t_disp - 1), ki]) if t_disp >= 1 and t_disp - 1 < acc_mag.shape[0] else None
        acc_after  = float(acc_mag[min(acc_mag.shape[0] - 1, t_disp), ki]) if t_disp < acc_mag.shape[0] else None
        jerk_val   = float(jerk_mag[min(jerk_mag.shape[0] - 1, max(0, t_disp - 1)), ki]) if jerk_mag.shape[0] > 0 else None

        cam_vis: dict[str, dict] = {}
        if has_2d:
            for label, t2d in [("before", tc.t3d_to_t2d(t_before)), ("after", tc.t3d_to_t2d(t_after))]:
                for cam in cc.cameras:
                    dets = cc.frame(cam, t2d)
                    if len(dets) == 0:
                        cam_vis.setdefault(cam, {})[label] = {"detected": False, "kp_score": None, "kp_visible": False}
                    else:
                        best = max(dets, key=lambda d: d[7])
                        kp_score = float(best[5][ki][2])
                        cam_vis.setdefault(cam, {})[label] = {
                            "detected": True, "kp_score": round(kp_score, 4),
                            "kp_visible": kp_score >= tc.VIS_SCORE_THR}
        if cam_vis:
            n_vis = sum(1 for v in cam_vis.values() if v.get("before", {}).get("kp_visible", False))
            occlusion = 1.0 - n_vis / len(cc.cameras)
        else:
            occlusion = None

        events.append({
            "frame_3d_before": t_before, "frame_3d_after": t_after,
            "frame_2d_before": tc.t3d_to_t2d(t_before) if has_2d else None,
            "kp_idx": ki, "kp_name": tc.KP_NAMES[ki],
            "jump_mm": round(float(dist[t_disp, ki]), 2),
            "velocity_mm_s": round(float(vel[t_disp, ki]), 2),
            "displacement_vec": [round(v, 2) for v in disp[t_disp, ki].tolist()],
            "acc_before_mm_s2": round(acc_before, 2) if acc_before is not None else None,
            "acc_after_mm_s2": round(acc_after, 2) if acc_after is not None else None,
            "jerk_mm_s3": round(jerk_val, 2) if jerk_val is not None else None,
            "score3d_before": round(float(sc3d[t_before, ki]), 4),
            "score3d_after": round(float(sc3d[min(T - 1, t_after), ki]), 4),
            "reproj_err_before_px": round(float(er3d[t_before, ki]), 4),
            "reproj_err_after_px": round(float(er3d[min(T - 1, t_after), ki]), 4),
            "occlusion_ratio_before": round(occlusion, 4) if occlusion is not None else None,
            "cameras": cam_vis,
        })
    cc.clear()

    kp_counts = np.zeros(tc.N_KP, dtype=int)
    kp_max_vel = np.zeros(tc.N_KP)
    for ev in events:
        kp_counts[ev["kp_idx"]] += 1
        kp_max_vel[ev["kp_idx"]] = max(kp_max_vel[ev["kp_idx"]], ev["velocity_mm_s"])

    vel_all, acc_all = vel.ravel(), acc_mag.ravel()
    return {
        "session": session, "n_frames_3d": T, "fps_3d": tc.FPS_3D, "fps_2d": tc.FPS_2D,
        "threshold_mm": THRESHOLD_MM, "vis_score_thr": tc.VIS_SCORE_THR, "cameras": cc.cameras,
        "n_teleport_events": int(tel_mask.sum()),
        "vel_all_mean_mm_s": round(float(vel_all.mean()), 2),
        "vel_all_p50_mm_s": round(float(np.percentile(vel_all, 50)), 2),
        "vel_all_p95_mm_s": round(float(np.percentile(vel_all, 95)), 2),
        "vel_all_p99_mm_s": round(float(np.percentile(vel_all, 99)), 2),
        "vel_all_max_mm_s": round(float(vel_all.max()), 2),
        "acc_all_mean_mm_s2": round(float(acc_all.mean()), 2),
        "acc_all_p95_mm_s2": round(float(np.percentile(acc_all, 95)), 2),
        "acc_all_p99_mm_s2": round(float(np.percentile(acc_all, 99)), 2),
        "kp_teleport_counts": {tc.KP_NAMES[i]: int(kp_counts[i]) for i in range(tc.N_KP)},
        "kp_max_velocity_mm_s": {tc.KP_NAMES[i]: round(float(kp_max_vel[i]), 2) for i in range(tc.N_KP)},
        "teleport_events": events,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--d2d-dir", type=Path, default=DEFAULT_D2D, help="2D 検出 JSON のルート")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="セッション名 (省略時は全セッション)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(p.name for p in args.input_dir.iterdir() if p.is_dir())
    print(f"Analyzing {len(sessions)} sessions …")

    all_results, summaries = [], []
    for i, session in enumerate(sessions):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sessions)}] {session}")
        result = analyze_session(session, args.input_dir, args.d2d_dir)
        if result is None:
            continue
        all_results.append(result)
        with open(args.output_dir / f"{session}_teleport.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        summaries.append({k: v for k, v in result.items() if k != "teleport_events"})

    print(f"\nSessions with teleportation: {len(all_results)}")
    with open(args.output_dir / "teleport_summary.json", "w") as f:
        json.dump({
            "threshold_mm": THRESHOLD_MM, "vis_score_thr": tc.VIS_SCORE_THR,
            "fps_3d": tc.FPS_3D, "fps_2d": tc.FPS_2D,
            "n_sessions_total": len(sessions), "n_sessions_with_teleport": len(all_results),
            "sessions": summaries,
        }, f, indent=2, ensure_ascii=False)
    print(f"  saved {args.output_dir / 'teleport_summary.json'}")


if __name__ == "__main__":
    main()
