"""
Teleportation analysis for 3D keypoints.

For each session with teleportation events (> THRESHOLD mm/frame):
  - Detects which keypoints jump, magnitude, velocity, acceleration
  - Cross-references 2D per-camera data to assess keypoint visibility
  - Saves per-session JSON and summary plots

Outputs → outputs/analysis_teleport/
  teleport_summary.json         — all-session summary
  {session}_teleport.json       — per-session detailed events
  heatmap_session_kp.png        — session × keypoint heatmap
  top_sessions_timeline.png     — velocity timeline for worst sessions
  camera_visibility.png         — occlusion analysis summary
"""

import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
FPS_3D = 24.0
FPS_2D = 25.0
DT = 1.0 / FPS_3D
THRESHOLD_MM = 100.0   # teleport threshold per frame
VIS_SCORE_THR = 0.5    # 2D keypoint "visible" if score >= this

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
N_KP = 20

BASE_3D = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/3d")
BASE_2D = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/2d")
OUT_DIR  = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/analysis_teleport")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 2D data loader (lazy chunk cache) ─────────────────────────────────────────
class ChunkCache:
    """Loads 2D JSON chunks on-demand and caches them in memory."""

    def __init__(self, session: str):
        self.session = session
        self.d2d_dir = BASE_2D / session
        self._cache: dict[tuple, list] = {}
        # discover cameras
        files = sorted(self.d2d_dir.glob("*.json")) if self.d2d_dir.exists() else []
        pat = re.compile(rf"{re.escape(session)}_(\d+)_(\d+)\.json")
        self.cameras: list[str] = []
        for f in files:
            m = pat.match(f.name)
            if m and m.group(1) not in self.cameras:
                self.cameras.append(m.group(1))

    def _load(self, cam: str, chunk: int) -> list:
        key = (cam, chunk)
        if key not in self._cache:
            path = self.d2d_dir / f"{self.session}_{cam}_{chunk:06d}.json"
            if path.exists():
                with open(path) as f:
                    self._cache[key] = json.load(f)
            else:
                self._cache[key] = []
        return self._cache[key]

    def frame(self, cam: str, t2d: int) -> list:
        """Return detections for camera `cam` at 2D frame index `t2d`."""
        chunk = t2d // 10000
        idx   = t2d % 10000
        data  = self._load(cam, chunk)
        if not data or idx >= len(data):
            return []
        return data[idx]

    def clear(self):
        self._cache.clear()


def t3d_to_t2d(t3d: int) -> int:
    return round(t3d * FPS_2D / FPS_3D)


# ── per-session analysis ───────────────────────────────────────────────────────
def analyze_session(session: str) -> dict | None:
    pkl_path = BASE_3D / session / "kp3d_fxdJointLen.pickle"
    if not pkl_path.exists():
        return None

    with open(pkl_path, "rb") as f:
        data3d = pickle.load(f)

    kp   = data3d["kp3d"][0]          # (T, 20, 3)
    sc3d = data3d["kp3d_score"][0]    # (T, 20)
    er3d = data3d["kp3d_err"][0]      # (T, 20)
    T    = kp.shape[0]

    # displacements and derived quantities
    disp = np.diff(kp, axis=0)                    # (T-1, 20, 3)
    dist = np.linalg.norm(disp, axis=-1)           # (T-1, 20) mm/frame
    vel  = dist / DT                               # mm/s

    vel_vec = disp / DT                            # (T-1, 20, 3)
    acc_vec = np.diff(vel_vec, axis=0)             # (T-2, 20, 3)
    acc_mag = np.linalg.norm(acc_vec, axis=-1)     # (T-2, 20) mm/s²

    # jerk: rate of change of acceleration vector
    jerk_vec = np.diff(acc_vec, axis=0)            # (T-3, 20, 3)
    jerk_mag = np.linalg.norm(jerk_vec, axis=-1)   # (T-3, 20) mm/s³

    # teleport mask
    tel_mask = dist > THRESHOLD_MM                 # (T-1, 20)
    tel_frames, tel_kps = np.where(tel_mask)       # frame indices (into disp)

    if len(tel_frames) == 0:
        return None

    # 2D loader
    cc = ChunkCache(session)
    has_2d = len(cc.cameras) > 0

    events = []
    for t_disp, ki in zip(tel_frames.tolist(), tel_kps.tolist()):
        t_before = t_disp        # frame before jump
        t_after  = t_disp + 1   # frame after jump

        jump_mm    = float(dist[t_disp, ki])
        vel_mms    = float(vel[t_disp, ki])
        disp_vec   = disp[t_disp, ki].tolist()

        # acceleration at surrounding frames (clamp to valid range)
        acc_before = float(acc_mag[max(0, t_disp - 1), ki]) if t_disp >= 1 and t_disp - 1 < acc_mag.shape[0] else None
        acc_after  = float(acc_mag[min(acc_mag.shape[0]-1, t_disp), ki]) if t_disp < acc_mag.shape[0] else None

        jerk_val   = float(jerk_mag[min(jerk_mag.shape[0]-1, max(0, t_disp-1)), ki]) if jerk_mag.shape[0] > 0 else None

        sc_before  = float(sc3d[t_before, ki])
        sc_after   = float(sc3d[min(T-1, t_after), ki])
        err_before = float(er3d[t_before, ki])
        err_after  = float(er3d[min(T-1, t_after), ki])

        # 2D visibility per camera
        cam_vis: dict[str, dict] = {}
        if has_2d:
            t2d_before = t3d_to_t2d(t_before)
            t2d_after  = t3d_to_t2d(t_after)
            for cam in cc.cameras:
                for label, t2d in [("before", t2d_before), ("after", t2d_after)]:
                    dets = cc.frame(cam, t2d)
                    if len(dets) == 0:
                        # no detection → not visible
                        cam_vis.setdefault(cam, {})[label] = {
                            "detected": False,
                            "kp_score": None,
                            "kp_visible": False,
                        }
                    else:
                        # pick first (or highest-score) detection
                        best = max(dets, key=lambda d: d[7])
                        kp_score = float(best[5][ki][2])
                        cam_vis.setdefault(cam, {})[label] = {
                            "detected": True,
                            "kp_score": round(kp_score, 4),
                            "kp_visible": kp_score >= VIS_SCORE_THR,
                        }

        # occlusion stats: fraction of cameras where kp is visible (at "before" frame)
        if cam_vis:
            n_visible_before = sum(
                1 for v in cam_vis.values() if v.get("before", {}).get("kp_visible", False)
            )
            occlusion_ratio = 1.0 - n_visible_before / len(cc.cameras)
        else:
            occlusion_ratio = None

        events.append({
            "frame_3d_before":  t_before,
            "frame_3d_after":   t_after,
            "frame_2d_before":  t3d_to_t2d(t_before) if has_2d else None,
            "kp_idx":           ki,
            "kp_name":          KP_NAMES[ki],
            # physical
            "jump_mm":          round(jump_mm, 2),
            "velocity_mm_s":    round(vel_mms, 2),
            "displacement_vec": [round(v, 2) for v in disp_vec],
            "acc_before_mm_s2": round(acc_before, 2) if acc_before is not None else None,
            "acc_after_mm_s2":  round(acc_after,  2) if acc_after  is not None else None,
            "jerk_mm_s3":       round(jerk_val,   2) if jerk_val   is not None else None,
            # 3D quality
            "score3d_before":   round(sc_before,  4),
            "score3d_after":    round(sc_after,   4),
            "reproj_err_before_px": round(err_before, 4),
            "reproj_err_after_px":  round(err_after,  4),
            # 2D visibility
            "occlusion_ratio_before": round(occlusion_ratio, 4) if occlusion_ratio is not None else None,
            "cameras":          cam_vis,
        })

    cc.clear()

    # per-kp summary for this session
    kp_counts  = np.zeros(N_KP, dtype=int)
    kp_max_vel = np.zeros(N_KP)
    for ev in events:
        kp_counts[ev["kp_idx"]] += 1
        kp_max_vel[ev["kp_idx"]] = max(kp_max_vel[ev["kp_idx"]], ev["velocity_mm_s"])

    # overall velocity stats (all frames, all kps)
    vel_all   = vel.ravel()
    acc_all   = acc_mag.ravel()

    return {
        "session":            session,
        "n_frames_3d":        T,
        "fps_3d":             FPS_3D,
        "fps_2d":             FPS_2D,
        "threshold_mm":       THRESHOLD_MM,
        "vis_score_thr":      VIS_SCORE_THR,
        "cameras":            cc.cameras,
        "n_teleport_events":  int(tel_mask.sum()),
        # velocity stats (all frames)
        "vel_all_mean_mm_s":  round(float(vel_all.mean()), 2),
        "vel_all_p50_mm_s":   round(float(np.percentile(vel_all, 50)), 2),
        "vel_all_p95_mm_s":   round(float(np.percentile(vel_all, 95)), 2),
        "vel_all_p99_mm_s":   round(float(np.percentile(vel_all, 99)), 2),
        "vel_all_max_mm_s":   round(float(vel_all.max()), 2),
        "acc_all_mean_mm_s2": round(float(acc_all.mean()), 2),
        "acc_all_p95_mm_s2":  round(float(np.percentile(acc_all, 95)), 2),
        "acc_all_p99_mm_s2":  round(float(np.percentile(acc_all, 99)), 2),
        "kp_teleport_counts": {KP_NAMES[i]: int(kp_counts[i]) for i in range(N_KP)},
        "kp_max_velocity_mm_s": {KP_NAMES[i]: round(float(kp_max_vel[i]), 2) for i in range(N_KP)},
        "teleport_events":    events,
    }


# ── plots ──────────────────────────────────────────────────────────────────────
def plot_heatmap(summaries: list[dict]):
    """Session × keypoint heatmap of teleport counts."""
    sessions = [s["session"] for s in summaries]
    matrix   = np.array([[s["kp_teleport_counts"][k] for k in KP_NAMES] for s in summaries])

    fig, ax = plt.subplots(figsize=(22, max(6, len(sessions) * 0.25 + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=matrix.max()))
    ax.set_xticks(range(N_KP))
    ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sessions)))
    ax.set_yticklabels(sessions, fontsize=6)
    ax.set_xlabel("Keypoint")
    ax.set_ylabel("Session")
    ax.set_title(f"Teleportation events (>{THRESHOLD_MM:.0f}mm/frame) — session × keypoint")
    fig.colorbar(im, ax=ax, label="n events")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_session_kp.png", dpi=150)
    plt.close(fig)
    print("  saved heatmap_session_kp.png")


def plot_timelines(session_results: list[dict], top_n: int = 6):
    """Velocity timeline + teleport markers for top-N sessions by event count."""
    top = sorted(session_results, key=lambda r: r["n_teleport_events"], reverse=True)[:top_n]

    fig, axes = plt.subplots(top_n, 1, figsize=(20, 4 * top_n), sharex=False)
    if top_n == 1:
        axes = [axes]

    for ax, res in zip(axes, top):
        session = res["session"]
        pkl_path = BASE_3D / session / "kp3d_fxdJointLen.pickle"
        with open(pkl_path, "rb") as f:
            kp = pickle.load(f)["kp3d"][0]

        disp = np.diff(kp, axis=0)
        vel  = np.linalg.norm(disp, axis=-1) / DT  # (T-1, 20)
        vel_mean = vel.mean(axis=1)                  # average over kps

        frames = np.arange(len(vel_mean))
        ax.plot(frames, vel_mean, lw=0.5, color="steelblue", label="mean vel (all kps)")

        # mark teleport events
        for ev in res["teleport_events"]:
            ax.axvline(ev["frame_3d_before"], color="red", alpha=0.3, lw=0.8)

        ax.set_ylim(0, min(vel_mean.max() * 1.5, 3000))
        ax.set_ylabel("velocity (mm/s)")
        ax.set_xlabel("frame (3D index)")
        ax.set_title(f"{session}  —  {res['n_teleport_events']} teleport events")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_sessions_timeline.png", dpi=120)
    plt.close(fig)
    print("  saved top_sessions_timeline.png")


def plot_camera_visibility(session_results: list[dict]):
    """Bar chart: mean occlusion ratio at teleport events, per keypoint."""
    kp_occ   = defaultdict(list)
    for res in session_results:
        for ev in res["teleport_events"]:
            if ev["occlusion_ratio_before"] is not None:
                kp_occ[ev["kp_name"]].append(ev["occlusion_ratio_before"])

    means = [np.mean(kp_occ[k]) if kp_occ[k] else 0.0 for k in KP_NAMES]
    counts = [len(kp_occ[k]) for k in KP_NAMES]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    colors = plt.cm.RdYlGn_r(np.array(means))
    ax1.bar(range(N_KP), means, color=colors)
    ax1.set_xticks(range(N_KP))
    ax1.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    ax1.set_ylabel("Mean occlusion ratio at teleport frame\n(1 = all cameras miss the keypoint)")
    ax1.set_ylim(0, 1)
    ax1.set_title("Camera visibility at teleportation events — by keypoint")
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8)

    ax2.bar(range(N_KP), counts, color="steelblue")
    ax2.set_xticks(range(N_KP))
    ax2.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    ax2.set_ylabel("Number of teleport events with 2D data")
    ax2.set_title("Teleport event count with camera visibility info")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "camera_visibility.png", dpi=150)
    plt.close(fig)
    print("  saved camera_visibility.png")


def plot_jump_distribution(session_results: list[dict]):
    """Box plot of jump magnitudes per keypoint."""
    kp_jumps = defaultdict(list)
    for res in session_results:
        for ev in res["teleport_events"]:
            kp_jumps[ev["kp_name"]].append(ev["jump_mm"])

    data   = [kp_jumps[k] for k in KP_NAMES]
    filled = [d if d else [0] for d in data]

    fig, ax = plt.subplots(figsize=(16, 5))
    bp = ax.boxplot(filled, patch_artist=True, showfliers=False,
                    medianprops=dict(color="red", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_xticks(range(1, N_KP + 1))
    ax.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Jump magnitude (mm)")
    ax.set_title(f"Jump size distribution per keypoint (>{THRESHOLD_MM:.0f}mm threshold, outliers hidden)")
    ax.axhline(THRESHOLD_MM, color="orange", ls="--", lw=0.8, label=f"threshold {THRESHOLD_MM:.0f}mm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "jump_distribution.png", dpi=150)
    plt.close(fig)
    print("  saved jump_distribution.png")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    sessions_3d = sorted(p.name for p in BASE_3D.iterdir() if p.is_dir())
    sessions_2d = set(p.name for p in BASE_2D.iterdir() if p.is_dir())

    all_results   = []
    session_summaries = []

    print(f"Analyzing {len(sessions_3d)} sessions …")
    for i, session in enumerate(sessions_3d):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sessions_3d)}] {session}")
        result = analyze_session(session)
        if result is None:
            continue

        all_results.append(result)

        # save per-session JSON (without huge teleport_events list stripped down)
        out_json = OUT_DIR / f"{session}_teleport.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # lightweight summary (no per-event data)
        summary = {k: v for k, v in result.items() if k != "teleport_events"}
        session_summaries.append(summary)

    print(f"\nSessions with teleportation: {len(all_results)}")

    # save summary JSON
    summary_out = {
        "threshold_mm": THRESHOLD_MM,
        "vis_score_thr": VIS_SCORE_THR,
        "fps_3d": FPS_3D,
        "fps_2d": FPS_2D,
        "n_sessions_total": len(sessions_3d),
        "n_sessions_with_teleport": len(all_results),
        "sessions": session_summaries,
    }
    with open(OUT_DIR / "teleport_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    print("  saved teleport_summary.json")

    # plots
    print("Generating plots …")
    plot_heatmap(session_summaries)
    plot_timelines(all_results, top_n=6)
    plot_camera_visibility(all_results)
    plot_jump_distribution(all_results)
    print("Done.")


if __name__ == "__main__":
    main()
