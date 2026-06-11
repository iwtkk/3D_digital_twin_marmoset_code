"""
3D キーポイントのテレポート解析 (run, v2) — 二段階判定。

  1. 孤立ジャンプ : kp jump > ISOLATED_MIN_MM かつ (jump / 隣接平均) > ISOLATED_RATIO
                    → 単一 KP の 3D 推定失敗
  2. 全体極端    : 全 KP の median 変位 > GLOBAL_MEDIAN_MM
                    → ID スイッチ / 全身トラッキング失敗（物理的に不可能な速度）

入力:
  --input-dir (既定 outputs/3d):  {input-dir}/{session}/kp3d_*.pickle
  --d2d-dir   (既定 outputs/2d):  2D 検出 JSON のルート
出力 (--output-dir, 既定 scripts/dev/teleport_v2_files):
  {session}_teleport_v2.json   per-session 詳細
  teleport_summary_v2.json     全セッション要約
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
DEFAULT_OUTPUT = Path("scripts/dev/teleport_v2_files")

ISOLATED_MIN_MM  = 80.0
ISOLATED_RATIO   = 3.0
GLOBAL_MEDIAN_MM = 250.0
CONTEXT_FRAMES   = 5


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
    acc_vec = np.diff(disp / tc.DT, axis=0)
    acc_mag = np.linalg.norm(acc_vec, axis=-1)
    med_per_frame = np.median(dist, axis=1)

    cc = tc.ChunkCache(session, d2d_dir)

    def build_event(t: int, ki: int, kind: str, ratio: float | None) -> dict:
        ctx_lo = max(0, t - CONTEXT_FRAMES)
        ctx_hi = min(dist.shape[0] - 1, t + CONTEXT_FRAMES + 1)
        vel_context = [round(float(vel[tt, ki]), 1) for tt in range(ctx_lo, ctx_hi)]
        acc_before = float(acc_mag[max(0, t - 1), ki]) if t >= 1 and t - 1 < acc_mag.shape[0] else None
        acc_after  = float(acc_mag[min(acc_mag.shape[0] - 1, t), ki]) if t < acc_mag.shape[0] else None
        err_before = float(er3d[t, ki])
        err_after  = float(er3d[min(T - 1, t + 1), ki])

        t2d = tc.t3d_to_t2d(t)
        cam_vis = {}
        for cam in cc.cameras:
            sc2 = cc.kp_score(cam, t2d, ki)
            cam_vis[cam] = {"kp_score": round(sc2, 4) if sc2 is not None else None,
                            "detected": sc2 is not None,
                            "kp_visible": (sc2 is not None and sc2 >= tc.VIS_SCORE_THR)}
        if cam_vis:
            n_vis = sum(1 for v in cam_vis.values() if v["kp_visible"])
            occlusion = 1.0 - n_vis / len(cc.cameras)
        else:
            occlusion = None

        return {
            "kind": kind, "frame_3d": t, "frame_2d": t2d,
            "kp_idx": ki, "kp_name": tc.KP_NAMES[ki],
            "jump_mm": round(float(dist[t, ki]), 2),
            "velocity_mm_s": round(float(vel[t, ki]), 2),
            "velocity_context_mm_s": vel_context, "context_start_frame": ctx_lo,
            "neighbor_jumps_mm": {tc.KP_NAMES[n]: round(float(dist[t, n]), 2) for n in tc.NEIGHBORS[ki]},
            "isolation_ratio": round(ratio, 2) if ratio is not None else None,
            "acc_before_mm_s2": round(acc_before, 2) if acc_before is not None else None,
            "acc_after_mm_s2": round(acc_after, 2) if acc_after is not None else None,
            "score3d_before": round(float(sc3d[t, ki]), 4),
            "score3d_after": round(float(sc3d[min(T - 1, t + 1), ki]), 4),
            "reproj_err_before_px": None if np.isnan(err_before) else round(err_before, 3),
            "reproj_err_after_px": None if np.isnan(err_after) else round(err_after, 3),
            "global_median_jump_mm": round(float(med_per_frame[t]), 2),
            "global_max_jump_mm": round(float(dist[t].max()), 2),
            "occlusion_ratio": round(occlusion, 4) if occlusion is not None else None,
            "cameras": cam_vis,
        }

    isolated_events, global_events = [], []
    for t in range(dist.shape[0]):
        if med_per_frame[t] > 100:
            continue
        for ki in range(tc.N_KP):
            if dist[t, ki] < ISOLATED_MIN_MM:
                continue
            nbr_mean = max(float(dist[t, tc.NEIGHBORS[ki]].mean()), 1e-6)
            ratio = dist[t, ki] / nbr_mean
            if ratio >= ISOLATED_RATIO:
                isolated_events.append(build_event(t, ki, "isolated", ratio))
    for t in np.where(med_per_frame > GLOBAL_MEDIAN_MM)[0]:
        ki_max = int(dist[t].argmax())
        global_events.append(build_event(int(t), ki_max, "global", None))
    cc.clear()

    if not isolated_events and not global_events:
        return None

    kp_iso_counts = {k: 0 for k in tc.KP_NAMES}
    for ev in isolated_events:
        kp_iso_counts[ev["kp_name"]] += 1
    vel_flat, acc_flat = vel.ravel(), acc_mag.ravel()

    return {
        "session": session, "n_frames_3d": T, "fps_3d": tc.FPS_3D, "fps_2d": tc.FPS_2D,
        "criteria": {"isolated_min_mm": ISOLATED_MIN_MM, "isolated_ratio": ISOLATED_RATIO,
                     "global_median_mm": GLOBAL_MEDIAN_MM},
        "cameras": cc.cameras, "n_isolated": len(isolated_events), "n_global": len(global_events),
        "vel_p50_mm_s": round(float(np.percentile(vel_flat, 50)), 2),
        "vel_p95_mm_s": round(float(np.percentile(vel_flat, 95)), 2),
        "vel_p99_mm_s": round(float(np.percentile(vel_flat, 99)), 2),
        "vel_max_mm_s": round(float(vel_flat.max()), 2),
        "acc_p95_mm_s2": round(float(np.percentile(acc_flat, 95)), 2),
        "acc_p99_mm_s2": round(float(np.percentile(acc_flat, 99)), 2),
        "kp_isolated_counts": kp_iso_counts,
        "isolated_events": isolated_events, "global_events": global_events,
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
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(sessions)}] {session}")
        result = analyze_session(session, args.input_dir, args.d2d_dir)
        if result is None:
            continue
        all_results.append(result)
        with open(args.output_dir / f"{session}_teleport_v2.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        summaries.append({k: v for k, v in result.items()
                          if k not in ("isolated_events", "global_events")})

    total_iso = sum(r["n_isolated"] for r in all_results)
    total_glb = sum(r["n_global"] for r in all_results)
    print(f"\nSessions with artifacts: {len(all_results)}")
    print(f"Isolated jump events:    {total_iso}")
    print(f"Global extreme events:   {total_glb}")
    with open(args.output_dir / "teleport_summary_v2.json", "w") as f:
        json.dump({
            "criteria": {"isolated_min_mm": ISOLATED_MIN_MM, "isolated_ratio": ISOLATED_RATIO,
                         "global_median_mm": GLOBAL_MEDIAN_MM, "vis_score_thr": tc.VIS_SCORE_THR},
            "fps_3d": tc.FPS_3D, "fps_2d": tc.FPS_2D,
            "n_sessions_total": len(sessions), "n_sessions_with_artifacts": len(all_results),
            "total_isolated_events": total_iso, "total_global_events": total_glb,
            "sessions": summaries,
        }, f, indent=2, ensure_ascii=False)
    print(f"  saved {args.output_dir / 'teleport_summary_v2.json'}")


if __name__ == "__main__":
    main()
