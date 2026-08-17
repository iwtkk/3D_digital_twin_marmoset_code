"""
テレポート解析の可視化 (plot, v2)。

run/analyze_teleport_v2.py が出力した {session}_teleport_v2.json を読み、
heatmap / kp artifact frequency / jump patterns / camera visibility /
velocity distribution を描く。velocity distribution は pickle も読む。

入力:
  --input-dir  (既定 scripts/dev/teleport_v2_files):  {session}_teleport_v2.json の場所
  --pickle-dir (既定 outputs/3d):                     pickle のルート
出力 (--output-dir, 既定 scripts/dev/teleport_v2_plot):
  heatmap_isolated.png / kp_artifact_frequency.png / jump_patterns.png
  camera_visibility_v2.png / velocity_distribution.png
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

DEFAULT_INPUT  = Path("scripts/dev/teleport_v2_files")
DEFAULT_PICKLE = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/teleport_v2_plot")
KP, N_KP = tc.KP_NAMES, tc.N_KP
ISOLATED_MIN_MM, ISOLATED_RATIO = 80.0, 3.0


def load_results(input_dir: Path, session: str | None) -> list[dict]:
    pat = f"{session}_teleport_v2.json" if session else "*_teleport_v2.json"
    files = sorted(p for p in input_dir.glob(pat) if not p.name.startswith("teleport_summary"))
    return [json.load(open(fp)) for fp in files]


def plot_heatmaps(results, out_dir):
    sessions = [s["session"] for s in results]
    iso_mat = np.array([[s["kp_isolated_counts"][k] for k in KP] for s in results])
    fig, ax = plt.subplots(figsize=(22, max(6, len(sessions) * 0.28 + 2)))
    im = ax.imshow(iso_mat, aspect="auto", cmap="YlOrRd",
                   norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=max(iso_mat.max(), 1)))
    ax.set_xticks(range(N_KP)); ax.set_xticklabels(KP, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sessions))); ax.set_yticklabels(sessions, fontsize=6)
    ax.set_title(f"Isolated jumps (>{ISOLATED_MIN_MM:.0f}mm, ratio>{ISOLATED_RATIO}x) — session × keypoint")
    fig.colorbar(im, ax=ax, label="n events")
    fig.tight_layout(); fig.savefig(out_dir / "heatmap_isolated.png", dpi=150); plt.close(fig)
    print("  saved heatmap_isolated.png")


def plot_kp_summary(results, out_dir):
    kp_total = {k: 0 for k in KP}
    for s in results:
        for k, v in s["kp_isolated_counts"].items():
            kp_total[k] += v
    vals = [kp_total[k] for k in KP]
    thresh = np.percentile(vals, 75)
    colors = ["tomato" if v > thresh else "steelblue" for v in vals]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(N_KP), vals, color=colors)
    ax.set_xticks(range(N_KP)); ax.set_xticklabels(KP, rotation=45, ha="right")
    ax.set_ylabel("Total isolated jump events (all sessions)")
    ax.set_title("Keypoint artifact frequency — anatomically impossible isolated jumps")
    fig.tight_layout(); fig.savefig(out_dir / "kp_artifact_frequency.png", dpi=150); plt.close(fig)
    print("  saved kp_artifact_frequency.png")


def plot_jump_patterns(results, out_dir, top_n=12):
    all_evs = []
    for res in results:
        for ev in res.get("isolated_events", []):
            all_evs.append((ev["velocity_mm_s"], res["session"], ev))
    all_evs.sort(key=lambda x: -x[0])
    top_evs = all_evs[:top_n]
    if not top_evs:
        print("  (no isolated events; skip jump_patterns)")
        return
    cols = 4
    rows = (len(top_evs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for i, (_, session, ev) in enumerate(top_evs):
        ax = axes[i]
        ctx = ev["velocity_context_mm_s"]
        xs = list(range(ev["context_start_frame"], ev["context_start_frame"] + len(ctx)))
        ax.plot(xs, ctx, marker="o", ms=4, color="steelblue")
        ax.axvline(ev["frame_3d"], color="red", lw=1.5)
        ax.set_title(f"{session[-12:]}\n{ev['kp_name']} {ev['jump_mm']:.0f}mm "
                     f"ratio={ev['isolation_ratio']:.0f}x", fontsize=8)
        ax.set_xlabel("frame"); ax.set_ylabel("vel (mm/s)")
    for i in range(len(top_evs), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("Velocity context around worst isolated-jump events", fontsize=11)
    fig.tight_layout(); fig.savefig(out_dir / "jump_patterns.png", dpi=150); plt.close(fig)
    print("  saved jump_patterns.png")


def plot_camera_visibility(results, out_dir):
    kp_occ = defaultdict(list)
    for res in results:
        for ev in res.get("isolated_events", []):
            if ev["occlusion_ratio"] is not None:
                kp_occ[ev["kp_name"]].append(ev["occlusion_ratio"])
    means = [np.mean(kp_occ[k]) if kp_occ[k] else np.nan for k in KP]
    counts = [len(kp_occ[k]) for k in KP]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.bar(range(N_KP), np.nan_to_num(means), color=plt.cm.RdYlGn_r(np.nan_to_num(means)))
    ax1.set_xticks(range(N_KP)); ax1.set_xticklabels(KP, rotation=45, ha="right")
    ax1.set_ylabel("Mean occlusion ratio\n(1 = all cameras miss keypoint)")
    ax1.set_ylim(0, 1); ax1.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax1.set_title("Camera visibility at isolated-jump artifact frames")
    ax2.bar(range(N_KP), counts, color="steelblue")
    ax2.set_xticks(range(N_KP)); ax2.set_xticklabels(KP, rotation=45, ha="right")
    ax2.set_ylabel("Number of isolated-jump events with 2D data")
    fig.tight_layout(); fig.savefig(out_dir / "camera_visibility_v2.png", dpi=150); plt.close(fig)
    print("  saved camera_visibility_v2.png")


def plot_velocity_distribution(results, pickle_dir, out_dir):
    all_normal, all_artifact = [], []
    for res in results:
        pkl = tc.resolve_pickle(pickle_dir / res["session"])
        if pkl is None:
            continue
        with open(pkl, "rb") as f:
            kp = pickle.load(f)["kp3d"][0]
        dist = np.linalg.norm(np.diff(kp, axis=0), axis=-1) / tc.DT
        artifact_frames = set(ev["frame_3d"] for ev in res.get("isolated_events", []))
        artifact_frames |= set(ev["frame_3d"] for ev in res.get("global_events", []))
        for t in range(dist.shape[0]):
            sample = dist[t].ravel()[::5]
            (all_artifact if t in artifact_frames else all_normal).extend(sample.tolist())
    bins = np.linspace(0, 3000, 100)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(all_normal, bins=bins, alpha=0.6, color="steelblue", label="Normal frames", density=True)
    ax.hist(all_artifact, bins=bins, alpha=0.7, color="tomato", label="Artifact frames", density=True)
    ax.set_xlabel("Velocity (mm/s)"); ax.set_ylabel("Density")
    ax.set_title("Keypoint velocity distribution: normal vs. artifact frames")
    ax.axvline(ISOLATED_MIN_MM / tc.DT, color="orange", ls="--", lw=1,
               label=f"isolated_min ({ISOLATED_MIN_MM/tc.DT:.0f} mm/s)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "velocity_distribution.png", dpi=150); plt.close(fig)
    print("  saved velocity_distribution.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help="{session}_teleport_v2.json の場所")
    ap.add_argument("--pickle-dir", type=Path, default=DEFAULT_PICKLE, help="pickle のルート")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="セッション名 (前方一致) で絞る")
    args = ap.parse_args()

    results = load_results(args.input_dir, args.session)
    if not results:
        print(f"{args.input_dir} に *_teleport_v2.json がありません。先に run を実行してください。",
              file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(results)} session JSONs. Generating plots …")
    plot_heatmaps(results, args.output_dir)
    plot_kp_summary(results, args.output_dir)
    plot_jump_patterns(results, args.output_dir)
    plot_camera_visibility(results, args.output_dir)
    plot_velocity_distribution(results, args.pickle_dir, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
