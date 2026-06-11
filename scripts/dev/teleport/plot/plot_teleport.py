"""
テレポート解析の可視化 (plot, v1)。

run/analyze_teleport.py が出力した {session}_teleport.json を読み、
heatmap / timeline / camera visibility / jump distribution を描く。
timeline は速度系列の再計算のため pickle も読む。

入力:
  --input-dir  (既定 scripts/dev/teleport_files):  {session}_teleport.json の場所
  --pickle-dir (既定 outputs/3d):                  pickle のルート (timeline 用)
出力 (--output-dir, 既定 scripts/dev/teleport_plot):
  heatmap_session_kp.png / top_sessions_timeline.png
  camera_visibility.png / jump_distribution.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import teleport_common as tc  # noqa: E402

DEFAULT_INPUT  = Path("scripts/dev/teleport_files")
DEFAULT_PICKLE = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/teleport_plot")
THRESHOLD_MM = 100.0
KP, N_KP = tc.KP_NAMES, tc.N_KP


def load_results(input_dir: Path, session: str | None) -> list[dict]:
    pat = f"{session}_teleport.json" if session else "*_teleport.json"
    files = sorted(p for p in input_dir.glob(pat) if not p.name.startswith("teleport_summary"))
    out = []
    for fp in files:
        with open(fp) as f:
            out.append(json.load(f))
    return out


def plot_heatmap(results, out_dir):
    sessions = [s["session"] for s in results]
    matrix = np.array([[s["kp_teleport_counts"][k] for k in KP] for s in results])
    fig, ax = plt.subplots(figsize=(22, max(6, len(sessions) * 0.25 + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=max(matrix.max(), 1)))
    ax.set_xticks(range(N_KP)); ax.set_xticklabels(KP, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sessions))); ax.set_yticklabels(sessions, fontsize=6)
    ax.set_xlabel("Keypoint"); ax.set_ylabel("Session")
    ax.set_title(f"Teleportation events (>{THRESHOLD_MM:.0f}mm/frame) — session × keypoint")
    fig.colorbar(im, ax=ax, label="n events")
    fig.tight_layout(); fig.savefig(out_dir / "heatmap_session_kp.png", dpi=150); plt.close(fig)
    print("  saved heatmap_session_kp.png")


def plot_timelines(results, pickle_dir, out_dir, top_n=6):
    top = sorted(results, key=lambda r: r["n_teleport_events"], reverse=True)[:top_n]
    n = max(1, len(top))
    fig, axes = plt.subplots(n, 1, figsize=(20, 4 * n))
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, top):
        pkl = tc.resolve_pickle(pickle_dir / res["session"])
        if pkl is None:
            continue
        with open(pkl, "rb") as f:
            kp = pickle.load(f)["kp3d"][0]
        vel = np.linalg.norm(np.diff(kp, axis=0), axis=-1) / tc.DT
        vel_mean = vel.mean(axis=1)
        ax.plot(np.arange(len(vel_mean)), vel_mean, lw=0.5, color="steelblue",
                label="mean vel (all kps)")
        for ev in res["teleport_events"]:
            ax.axvline(ev["frame_3d_before"], color="red", alpha=0.3, lw=0.8)
        ax.set_ylim(0, min(vel_mean.max() * 1.5, 3000))
        ax.set_ylabel("velocity (mm/s)"); ax.set_xlabel("frame (3D index)")
        ax.set_title(f"{res['session']} — {res['n_teleport_events']} teleport events")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "top_sessions_timeline.png", dpi=120); plt.close(fig)
    print("  saved top_sessions_timeline.png")


def plot_camera_visibility(results, out_dir):
    kp_occ = defaultdict(list)
    for res in results:
        for ev in res["teleport_events"]:
            if ev["occlusion_ratio_before"] is not None:
                kp_occ[ev["kp_name"]].append(ev["occlusion_ratio_before"])
    means = [np.mean(kp_occ[k]) if kp_occ[k] else 0.0 for k in KP]
    counts = [len(kp_occ[k]) for k in KP]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.bar(range(N_KP), means, color=plt.cm.RdYlGn_r(np.array(means)))
    ax1.set_xticks(range(N_KP)); ax1.set_xticklabels(KP, rotation=45, ha="right")
    ax1.set_ylabel("Mean occlusion ratio at teleport frame\n(1 = all cameras miss the keypoint)")
    ax1.set_ylim(0, 1); ax1.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax1.set_title("Camera visibility at teleportation events — by keypoint")
    ax2.bar(range(N_KP), counts, color="steelblue")
    ax2.set_xticks(range(N_KP)); ax2.set_xticklabels(KP, rotation=45, ha="right")
    ax2.set_ylabel("Number of teleport events with 2D data")
    fig.tight_layout(); fig.savefig(out_dir / "camera_visibility.png", dpi=150); plt.close(fig)
    print("  saved camera_visibility.png")


def plot_jump_distribution(results, out_dir):
    kp_jumps = defaultdict(list)
    for res in results:
        for ev in res["teleport_events"]:
            kp_jumps[ev["kp_name"]].append(ev["jump_mm"])
    filled = [kp_jumps[k] if kp_jumps[k] else [0] for k in KP]
    fig, ax = plt.subplots(figsize=(16, 5))
    bp = ax.boxplot(filled, patch_artist=True, showfliers=False,
                    medianprops=dict(color="red", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_xticks(range(1, N_KP + 1)); ax.set_xticklabels(KP, rotation=45, ha="right")
    ax.set_ylabel("Jump magnitude (mm)")
    ax.set_title(f"Jump size distribution per keypoint (>{THRESHOLD_MM:.0f}mm threshold)")
    ax.axhline(THRESHOLD_MM, color="orange", ls="--", lw=0.8, label=f"threshold {THRESHOLD_MM:.0f}mm")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "jump_distribution.png", dpi=150); plt.close(fig)
    print("  saved jump_distribution.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help="{session}_teleport.json の場所")
    ap.add_argument("--pickle-dir", type=Path, default=DEFAULT_PICKLE, help="timeline 用 pickle のルート")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="セッション名 (前方一致) で絞る")
    args = ap.parse_args()

    results = load_results(args.input_dir, args.session)
    if not results:
        print(f"{args.input_dir} に *_teleport.json がありません。先に run を実行してください。",
              file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(results)} session JSONs. Generating plots …")
    plot_heatmap(results, args.output_dir)
    plot_timelines(results, args.pickle_dir, args.output_dir)
    plot_camera_visibility(results, args.output_dir)
    plot_jump_distribution(results, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
