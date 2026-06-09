"""
Teleportation analysis v2 — improved detection criteria.

OLD: any keypoint jump > 100mm/frame (too many false positives: real fast movements)
NEW two-stage criterion:
  1. Isolated jump  : kp jump > ISOLATED_MIN_MM AND (jump / mean_neighbor_jump) > ISOLATED_RATIO
                      - captures single-keypoint 3D estimation failures
  2. Global extreme : median displacement of ALL keypoints > GLOBAL_MEDIAN_MM
                      - captures ID-switch / whole-body tracking failure (physically impossible speed)

Physical basis (marmoset, 24fps):
  - Max realistic sprint: ~5 m/s → ~208 mm/frame peak
  - Global median > 250mm/frame ≈ 6 m/s body-wide → impossible in enclosure

Outputs → outputs/analysis_teleport_v2/
  teleport_summary_v2.json
  {session}_teleport_v2.json
  heatmap_isolated.png
  velocity_distribution.png
  jump_patterns.png       — before/during/after velocity for worst events
  camera_visibility_v2.png
  kp_artifact_frequency.png
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
ISOLATED_MIN_MM  = 80.0
ISOLATED_RATIO   = 3.0
GLOBAL_MEDIAN_MM = 250.0
VIS_SCORE_THR    = 0.5

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
N_KP = 20

# anatomical neighbors (direct joints)
NEIGHBORS = {
    0:[1,2,3,4], 1:[0,2], 2:[0,1], 3:[0,5], 4:[0,6],
    5:[3,6,7,11,17], 6:[4,5,8,12,17],
    7:[5,9], 8:[6,10], 9:[7], 10:[8],
    11:[5,12,13,17], 12:[6,11,14,17],
    13:[11,15], 14:[12,16], 15:[13], 16:[14],
    17:[5,6,11,12,18], 18:[17,19], 19:[18],
}

BASE_3D = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/3d")
BASE_2D = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/2d")
OUT_DIR = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/analysis_teleport_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_FRAMES = 5  # frames before/after to include in event context


# ── 2D loader ─────────────────────────────────────────────────────────────────
class ChunkCache:
    def __init__(self, session: str):
        self.session = session
        self.d2d_dir = BASE_2D / session
        self._cache: dict = {}
        files = sorted(self.d2d_dir.glob("*.json")) if self.d2d_dir.exists() else []
        pat = re.compile(rf"{re.escape(session)}_(\d+)_(\d+)\.json")
        self.cameras = []
        for f in files:
            m = pat.match(f.name)
            if m and m.group(1) not in self.cameras:
                self.cameras.append(m.group(1))

    def frame(self, cam: str, t2d: int) -> list:
        chunk = t2d // 10000
        idx   = t2d % 10000
        key   = (cam, chunk)
        if key not in self._cache:
            path = self.d2d_dir / f"{self.session}_{cam}_{chunk:06d}.json"
            self._cache[key] = json.load(open(path)) if path.exists() else []
        data = self._cache[key]
        return data[idx] if data and idx < len(data) else []

    def kp_score(self, cam: str, t2d: int, ki: int) -> float | None:
        dets = self.frame(cam, t2d)
        if not dets:
            return None
        best = max(dets, key=lambda d: d[7])
        return float(best[5][ki][2])

    def clear(self):
        self._cache.clear()


def t3d_to_t2d(t: int) -> int:
    return round(t * FPS_2D / FPS_3D)


# ── per-session analysis ───────────────────────────────────────────────────────
def analyze_session(session: str) -> dict | None:
    pkl_path = BASE_3D / session / "kp3d_fxdJointLen.pickle"
    if not pkl_path.exists():
        return None

    with open(pkl_path, "rb") as f:
        data3d = pickle.load(f)

    kp   = data3d["kp3d"][0]       # (T, 20, 3)
    sc3d = data3d["kp3d_score"][0] # (T, 20)
    er3d = data3d["kp3d_err"][0]   # (T, 20)
    T    = kp.shape[0]

    disp    = np.diff(kp, axis=0)                     # (T-1, 20, 3)
    dist    = np.linalg.norm(disp, axis=-1)            # (T-1, 20) mm/frame
    vel     = dist / DT                                # mm/s
    vel_vec = disp / DT                                # (T-1, 20, 3)
    acc_vec = np.diff(vel_vec, axis=0)                 # (T-2, 20, 3)
    acc_mag = np.linalg.norm(acc_vec, axis=-1)         # (T-2, 20) mm/s²
    med_per_frame = np.median(dist, axis=1)            # (T-1,)

    cc = ChunkCache(session)

    def build_event(t: int, ki: int, kind: str, ratio: float | None) -> dict:
        jump_mm   = float(dist[t, ki])
        vel_mms   = float(vel[t, ki])
        nbr_vals  = {KP_NAMES[n]: round(float(dist[t, n]), 2) for n in NEIGHBORS[ki]}

        ctx_lo = max(0, t - CONTEXT_FRAMES)
        ctx_hi = min(dist.shape[0] - 1, t + CONTEXT_FRAMES + 1)
        vel_context = [round(float(vel[tt, ki]), 1) for tt in range(ctx_lo, ctx_hi)]

        acc_before = float(acc_mag[max(0, t-1), ki]) if t >= 1 and t-1 < acc_mag.shape[0] else None
        acc_after  = float(acc_mag[min(acc_mag.shape[0]-1, t), ki]) if t < acc_mag.shape[0] else None

        sc_before  = float(sc3d[t, ki])
        sc_after   = float(sc3d[min(T-1, t+1), ki])
        err_before = float(er3d[t, ki])
        err_after  = float(er3d[min(T-1, t+1), ki])

        t2d = t3d_to_t2d(t)
        cam_vis = {}
        for cam in cc.cameras:
            sc2 = cc.kp_score(cam, t2d, ki)
            cam_vis[cam] = {
                "kp_score": round(sc2, 4) if sc2 is not None else None,
                "detected": sc2 is not None,
                "kp_visible": (sc2 is not None and sc2 >= VIS_SCORE_THR),
            }
        if cam_vis:
            n_vis = sum(1 for v in cam_vis.values() if v["kp_visible"])
            occlusion = 1.0 - n_vis / len(cc.cameras)
        else:
            occlusion = None

        return {
            "kind": kind,
            "frame_3d": t,
            "frame_2d": t2d,
            "kp_idx": ki,
            "kp_name": KP_NAMES[ki],
            "jump_mm": round(jump_mm, 2),
            "velocity_mm_s": round(vel_mms, 2),
            "velocity_context_mm_s": vel_context,
            "context_start_frame": ctx_lo,
            "neighbor_jumps_mm": nbr_vals,
            "isolation_ratio": round(ratio, 2) if ratio is not None else None,
            "acc_before_mm_s2": round(acc_before, 2) if acc_before is not None else None,
            "acc_after_mm_s2":  round(acc_after,  2) if acc_after  is not None else None,
            "score3d_before": round(sc_before, 4),
            "score3d_after":  round(sc_after, 4),
            "reproj_err_before_px": None if np.isnan(err_before) else round(err_before, 3),
            "reproj_err_after_px":  None if np.isnan(err_after)  else round(err_after,  3),
            "global_median_jump_mm": round(float(med_per_frame[t]), 2),
            "global_max_jump_mm":    round(float(dist[t].max()), 2),
            "occlusion_ratio": round(occlusion, 4) if occlusion is not None else None,
            "cameras": cam_vis,
        }

    isolated_events = []
    global_events   = []

    # isolated jump detection
    for t in range(dist.shape[0]):
        if med_per_frame[t] > 100:
            continue
        for ki in range(N_KP):
            if dist[t, ki] < ISOLATED_MIN_MM:
                continue
            nbr_mean = float(dist[t, NEIGHBORS[ki]].mean())
            if nbr_mean < 1e-6:
                nbr_mean = 1e-6
            ratio = dist[t, ki] / nbr_mean
            if ratio >= ISOLATED_RATIO:
                isolated_events.append(build_event(t, ki, "isolated", ratio))

    # global extreme detection
    for t in np.where(med_per_frame > GLOBAL_MEDIAN_MM)[0]:
        ki_max = int(dist[t].argmax())
        global_events.append(build_event(int(t), ki_max, "global", None))

    cc.clear()

    if not isolated_events and not global_events:
        return None

    kp_iso_counts = {k: 0 for k in KP_NAMES}
    for ev in isolated_events:
        kp_iso_counts[ev["kp_name"]] += 1

    vel_flat = vel.ravel()
    acc_flat = acc_mag.ravel()

    return {
        "session": session,
        "n_frames_3d": T,
        "fps_3d": FPS_3D,
        "fps_2d": FPS_2D,
        "criteria": {
            "isolated_min_mm": ISOLATED_MIN_MM,
            "isolated_ratio": ISOLATED_RATIO,
            "global_median_mm": GLOBAL_MEDIAN_MM,
        },
        "cameras": cc.cameras,
        "n_isolated": len(isolated_events),
        "n_global": len(global_events),
        "vel_p50_mm_s":  round(float(np.percentile(vel_flat, 50)),  2),
        "vel_p95_mm_s":  round(float(np.percentile(vel_flat, 95)),  2),
        "vel_p99_mm_s":  round(float(np.percentile(vel_flat, 99)),  2),
        "vel_max_mm_s":  round(float(vel_flat.max()), 2),
        "acc_p95_mm_s2": round(float(np.percentile(acc_flat, 95)), 2),
        "acc_p99_mm_s2": round(float(np.percentile(acc_flat, 99)), 2),
        "kp_isolated_counts": kp_iso_counts,
        "isolated_events": isolated_events,
        "global_events":   global_events,
    }


# ── plots ──────────────────────────────────────────────────────────────────────
def plot_heatmaps(summaries: list[dict]):
    sessions = [s["session"] for s in summaries]
    iso_mat  = np.array([[s["kp_isolated_counts"][k] for k in KP_NAMES] for s in summaries])

    fig, ax = plt.subplots(figsize=(22, max(6, len(sessions) * 0.28 + 2)))
    im = ax.imshow(iso_mat, aspect="auto", cmap="YlOrRd",
                   norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=max(iso_mat.max(), 1)))
    ax.set_xticks(range(N_KP)); ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sessions))); ax.set_yticklabels(sessions, fontsize=6)
    ax.set_title(f"Isolated jumps (>{ISOLATED_MIN_MM:.0f}mm, ratio>{ISOLATED_RATIO}x) — session × keypoint")
    fig.colorbar(im, ax=ax, label="n events")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_isolated.png", dpi=150)
    plt.close(fig)
    print("  saved heatmap_isolated.png")


def plot_velocity_distribution(session_results: list[dict]):
    all_vel_normal   = []
    all_vel_artifact = []

    for res in session_results:
        pkl_path = BASE_3D / res["session"] / "kp3d_fxdJointLen.pickle"
        with open(pkl_path, "rb") as f:
            kp = pickle.load(f)["kp3d"][0]
        dist = np.linalg.norm(np.diff(kp, axis=0), axis=-1) / DT  # mm/s

        artifact_frames = set(ev["frame_3d"] for ev in res.get("isolated_events", []))
        artifact_frames |= set(ev["frame_3d"] for ev in res.get("global_events", []))

        for t in range(dist.shape[0]):
            sample = dist[t].ravel()[::5]
            if t in artifact_frames:
                all_vel_artifact.extend(sample.tolist())
            else:
                all_vel_normal.extend(sample.tolist())

    bins = np.linspace(0, 3000, 100)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(all_vel_normal,   bins=bins, alpha=0.6, color="steelblue", label="Normal frames", density=True)
    ax.hist(all_vel_artifact, bins=bins, alpha=0.7, color="tomato",    label="Artifact frames", density=True)
    ax.set_xlabel("Velocity (mm/s)"); ax.set_ylabel("Density")
    ax.set_title("Keypoint velocity distribution: normal vs. artifact frames")
    ax.axvline(ISOLATED_MIN_MM / DT, color="orange", ls="--", lw=1,
               label=f"isolated_min ({ISOLATED_MIN_MM/DT:.0f} mm/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "velocity_distribution.png", dpi=150)
    plt.close(fig)
    print("  saved velocity_distribution.png")


def plot_jump_patterns(session_results: list[dict], top_n: int = 12):
    all_evs = []
    for res in session_results:
        for ev in res.get("isolated_events", []):
            all_evs.append((ev["velocity_mm_s"], res["session"], ev))
    all_evs.sort(key=lambda x: -x[0])
    top_evs = all_evs[:top_n]

    cols = 4
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = axes.ravel()

    for i, (vel, session, ev) in enumerate(top_evs):
        ax = axes[i]
        ctx = ev["velocity_context_mm_s"]
        t0  = ev["context_start_frame"]
        xs  = list(range(t0, t0 + len(ctx)))
        ax.plot(xs, ctx, marker="o", ms=4, color="steelblue")
        ax.axvline(ev["frame_3d"], color="red", lw=1.5)
        ax.set_title(f"{session[-12:]}\n{ev['kp_name']}  {ev['jump_mm']:.0f}mm  "
                     f"ratio={ev['isolation_ratio']:.0f}x", fontsize=8)
        ax.set_xlabel("frame"); ax.set_ylabel("vel (mm/s)")

    for i in range(len(top_evs), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Velocity context around worst isolated-jump events", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "jump_patterns.png", dpi=150)
    plt.close(fig)
    print("  saved jump_patterns.png")


def plot_camera_visibility(session_results: list[dict]):
    kp_occ = defaultdict(list)
    for res in session_results:
        for ev in res.get("isolated_events", []):
            if ev["occlusion_ratio"] is not None:
                kp_occ[ev["kp_name"]].append(ev["occlusion_ratio"])

    means  = [np.mean(kp_occ[k]) if kp_occ[k] else np.nan for k in KP_NAMES]
    counts = [len(kp_occ[k]) for k in KP_NAMES]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    colors = plt.cm.RdYlGn_r(np.nan_to_num(means))
    ax1.bar(range(N_KP), np.nan_to_num(means), color=colors)
    ax1.set_xticks(range(N_KP)); ax1.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    ax1.set_ylabel("Mean occlusion ratio\n(1 = all cameras miss keypoint)")
    ax1.set_ylim(0, 1); ax1.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax1.set_title("Camera visibility at isolated-jump artifact frames")

    ax2.bar(range(N_KP), counts, color="steelblue")
    ax2.set_xticks(range(N_KP)); ax2.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    ax2.set_ylabel("Number of isolated-jump events with 2D data")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "camera_visibility_v2.png", dpi=150)
    plt.close(fig)
    print("  saved camera_visibility_v2.png")


def plot_kp_summary(summaries: list[dict]):
    kp_total = {k: 0 for k in KP_NAMES}
    for s in summaries:
        for k, v in s["kp_isolated_counts"].items():
            kp_total[k] += v

    vals   = [kp_total[k] for k in KP_NAMES]
    thresh = np.percentile(vals, 75)
    colors = ["tomato" if v > thresh else "steelblue" for v in vals]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(N_KP), vals, color=colors)
    ax.set_xticks(range(N_KP)); ax.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Total isolated jump events (all sessions)")
    ax.set_title("Keypoint artifact frequency — anatomically impossible isolated jumps")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "kp_artifact_frequency.png", dpi=150)
    plt.close(fig)
    print("  saved kp_artifact_frequency.png")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    sessions = sorted(p.name for p in BASE_3D.iterdir() if p.is_dir())
    all_results = []
    summaries   = []

    print(f"Analyzing {len(sessions)} sessions …")
    for i, session in enumerate(sessions):
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(sessions)}] {session}")
        result = analyze_session(session)
        if result is None:
            continue
        all_results.append(result)

        with open(OUT_DIR / f"{session}_teleport_v2.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        summaries.append({k: v for k, v in result.items()
                          if k not in ("isolated_events", "global_events")})

    total_iso = sum(r["n_isolated"] for r in all_results)
    total_glb = sum(r["n_global"]   for r in all_results)
    print(f"\nSessions with artifacts: {len(all_results)}")
    print(f"Isolated jump events:    {total_iso}")
    print(f"Global extreme events:   {total_glb}")

    with open(OUT_DIR / "teleport_summary_v2.json", "w") as f:
        json.dump({
            "criteria": {
                "isolated_min_mm": ISOLATED_MIN_MM,
                "isolated_ratio": ISOLATED_RATIO,
                "global_median_mm": GLOBAL_MEDIAN_MM,
                "vis_score_thr": VIS_SCORE_THR,
            },
            "fps_3d": FPS_3D, "fps_2d": FPS_2D,
            "n_sessions_total": len(sessions),
            "n_sessions_with_artifacts": len(all_results),
            "total_isolated_events": total_iso,
            "total_global_events": total_glb,
            "sessions": summaries,
        }, f, indent=2, ensure_ascii=False)
    print("  saved teleport_summary_v2.json")

    print("Generating plots …")
    plot_heatmaps(summaries)
    plot_kp_summary(summaries)
    plot_jump_patterns(all_results, top_n=12)
    plot_camera_visibility(all_results)
    plot_velocity_distribution(all_results)
    print("Done.")


if __name__ == "__main__":
    main()
