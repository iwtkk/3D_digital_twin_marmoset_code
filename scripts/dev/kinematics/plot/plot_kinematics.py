"""
運動学・信号品質解析の可視化 (plot)。

run/analyze_kinematics.py が出力した kinematics_accum.npz を読み、
速度/加速度/ジャークのヒストグラム・PSD・ボーン長変動・代表セッション時系列を描く。

入力 (--input-dir, 既定 scripts/dev/kinematics_files):  kinematics_accum.npz
出力 (--output-dir, 既定 scripts/dev/kinematics_plot):
  velocity_distribution.png / speed|accel|jerk_hist_per_kp.png
  psd_pos_vel_acc.png / psd_velocity_per_kp.png / bone_length_cv.png
  timeseries_speed|accel|jerk.png / timeseries_bone_length.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import kinematics_common as kc  # noqa: E402

DEFAULT_INPUT  = Path("scripts/dev/kinematics_files")
DEFAULT_OUTPUT = Path("scripts/dev/kinematics_plot")
KP, N_KP = kc.KP_NAMES, kc.N_KP


def _grid_axes(nrows=4, ncols=5, figsize=(20, 14)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes.ravel()


def plot_overall_velocity(counts_kp, bins, out_dir):
    counts = counts_kp.sum(axis=0)
    centers = np.sqrt(bins[:-1] * bins[1:])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(centers, counts, width=np.diff(bins), align="center", color="steelblue", alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Speed (mm/s)"); ax.set_ylabel("Count (all keypoints, all sessions)")
    cdf = np.cumsum(counts) / counts.sum()
    for q, c in [(50, "green"), (95, "orange"), (99, "red")]:
        v = float(np.interp(q / 100.0, cdf, centers))
        ax.axvline(v, color=c, ls="--", lw=1, label=f"p{q} = {v:.0f} mm/s")
    ax.axvline(5000, color="black", ls=":", lw=1, label="biol. limit ~5 m/s")
    ax.legend(); ax.set_title("Keypoint speed distribution (3D anipose-smoothed)")
    fig.tight_layout(); fig.savefig(out_dir / "velocity_distribution.png", dpi=150); plt.close(fig)
    print("  saved velocity_distribution.png")


def plot_hist_per_kp(counts_kp, bins, fname, xlabel, color, out_dir):
    centers = np.sqrt(bins[:-1] * bins[1:])
    fig, axes = _grid_axes()
    for ki in range(N_KP):
        ax = axes[ki]
        ax.bar(centers, counts_kp[ki], width=np.diff(bins), align="center", color=color, alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        p50 = kc.hist_percentile(counts_kp[ki], bins, 50)
        p95 = kc.hist_percentile(counts_kp[ki], bins, 95)
        p99 = kc.hist_percentile(counts_kp[ki], bins, 99)
        ax.axvline(p95, color="orange", ls="--", lw=0.8)
        ax.axvline(p99, color="red", ls="--", lw=0.8)
        ax.set_title(f"{KP[ki]}\np50={p50:.0f} p99={p99:.0f}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Per-keypoint {xlabel} histogram (all sessions, log-log)", fontsize=13)
    fig.supxlabel(xlabel)
    fig.tight_layout(); fig.savefig(out_dir / fname, dpi=150); plt.close(fig)
    print(f"  saved {fname}")


def plot_psd(d, out_dir):
    f = d["psd_freqs"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, key, name, unit in [
        (axes[0], "psd_pos_mean", "Position", "mm^2/Hz"),
        (axes[1], "psd_vel_mean", "Velocity", "(mm/s)^2/Hz"),
        (axes[2], "psd_acc_mean", "Acceleration", "(mm/s^2)^2/Hz"),
    ]:
        m = d[key]
        lo, hi = np.percentile(m, [10, 90], axis=0)
        ax.fill_between(f, lo, hi, color="steelblue", alpha=0.25, label="10-90% kp")
        ax.plot(f, m.mean(axis=0), color="navy", lw=1.5, label="mean over kp")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axvline(kc.HF_CUTOFF_HZ, color="red", ls=":", lw=1, label=f"{kc.HF_CUTOFF_HZ:.0f} Hz")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(f"PSD ({unit})")
        ax.set_title(f"{name} PSD"); ax.legend(fontsize=8)
    fig.suptitle(f"Power spectral density — 3D is anipose-smoothed (Nyquist = {kc.NYQUIST:.0f} Hz).",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir / "psd_pos_vel_acc.png", dpi=150); plt.close(fig)
    print("  saved psd_pos_vel_acc.png")


def plot_psd_heatmap(d, out_dir):
    m, f = d["psd_vel_mean"], d["psd_freqs"]
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.pcolormesh(f, np.arange(N_KP), np.log10(m + 1e-12), cmap="magma", shading="auto")
    ax.set_xscale("log")
    ax.set_yticks(range(N_KP)); ax.set_yticklabels(KP, fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.axvline(kc.HF_CUTOFF_HZ, color="cyan", ls=":", lw=1)
    ax.set_title("Velocity PSD per keypoint (log10) — bright high-freq tail = noisier keypoint")
    fig.colorbar(im, ax=ax, label="log10 PSD")
    fig.tight_layout(); fig.savefig(out_dir / "psd_velocity_per_kp.png", dpi=150); plt.close(fig)
    print("  saved psd_velocity_per_kp.png")


def plot_bone_cv(d, out_dir):
    mean, std, cv = d["bone_mean"], d["bone_std"], d["bone_cv"]
    order = np.argsort(cv)[::-1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    x = np.arange(kc.N_BONE)
    colors = ["tomato" if cv[i] > 0.05 else "steelblue" for i in order]
    ax1.bar(x, cv[order] * 100, color=colors)
    ax1.set_xticks(x); ax1.set_xticklabels([kc.BONE_NAMES[i] for i in order], rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("Coefficient of variation (%)")
    ax1.axhline(5, color="gray", ls="--", lw=0.8, label="5% (fixed-length should be ~0)")
    ax1.set_title("Per-bone length variability — high CV flags unconstrained / unstable bones")
    ax1.legend()
    ax2.bar(x - 0.2, mean[order], width=0.4, color="navy", label="measured mean length")
    ax2.errorbar(x - 0.2, mean[order], yerr=std[order], fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax2.set_xticks(x); ax2.set_xticklabels([kc.BONE_NAMES[i] for i in order], rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("Bone length (mm)"); ax2.set_title("Mean bone length ± std"); ax2.legend()
    fig.tight_layout(); fig.savefig(out_dir / "bone_length_cv.png", dpi=150); plt.close(fig)
    print("  saved bone_length_cv.png")


def plot_timeseries(metric, fname, ylabel, color, out_dir):
    t = np.arange(metric.shape[0]) / kc.FPS
    fig, axes = _grid_axes()
    for ki in range(N_KP):
        axes[ki].plot(t, metric[:, ki], color=color, lw=0.4)
        axes[ki].set_title(KP[ki], fontsize=8); axes[ki].tick_params(labelsize=6)
    fig.suptitle(f"{ylabel} time series — representative session", fontsize=13)
    fig.supxlabel("Time (s)"); fig.supylabel(ylabel)
    fig.tight_layout(); fig.savefig(out_dir / fname, dpi=130); plt.close(fig)
    print(f"  saved {fname}")


def plot_bone_timeseries(bone_len, out_dir):
    t = np.arange(bone_len.shape[0]) / kc.FPS
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = plt.cm.tab20(np.linspace(0, 1, kc.N_BONE))
    for bi in range(kc.N_BONE):
        ax.plot(t, bone_len[:, bi], lw=0.5, color=cmap[bi], label=kc.BONE_NAMES[bi])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Bone length (mm)")
    ax.set_title("Bone lengths over time — representative session")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(out_dir / "timeseries_bone_length.png", dpi=130); plt.close(fig)
    print("  saved timeseries_bone_length.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help="kinematics_accum.npz の場所")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="(互換用; npz 内の代表セッションを使用)")
    args = ap.parse_args()

    npz_path = args.input_dir / "kinematics_accum.npz"
    if not npz_path.exists():
        print(f"{npz_path} がありません。先に run/analyze_kinematics.py を実行してください。",
              file=sys.stderr)
        sys.exit(1)
    d = np.load(npz_path, allow_pickle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {npz_path}. Generating plots …")

    plot_overall_velocity(d["speed_counts"], d["speed_bins"], args.output_dir)
    plot_hist_per_kp(d["speed_counts"], d["speed_bins"], "speed_hist_per_kp.png",
                     "Speed (mm/s)", "steelblue", args.output_dir)
    plot_hist_per_kp(d["accel_counts"], d["accel_bins"], "accel_hist_per_kp.png",
                     "Acceleration (mm/s^2)", "darkorange", args.output_dir)
    plot_hist_per_kp(d["jerk_counts"], d["jerk_bins"], "jerk_hist_per_kp.png",
                     "Jerk (mm/s^3)", "seagreen", args.output_dir)
    plot_psd(d, args.output_dir)
    plot_psd_heatmap(d, args.output_dir)
    plot_bone_cv(d, args.output_dir)

    if "ts_speed" in d.files:
        plot_timeseries(d["ts_speed"], "timeseries_speed.png", "Speed (mm/s)", "steelblue", args.output_dir)
        plot_timeseries(d["ts_accel"], "timeseries_accel.png", "Acceleration (mm/s^2)", "darkorange", args.output_dir)
        plot_timeseries(d["ts_jerk"], "timeseries_jerk.png", "Jerk (mm/s^3)", "seagreen", args.output_dir)
        plot_bone_timeseries(d["ts_bone_len"], args.output_dir)
    else:
        print("  WARNING: npz に時系列データがありません (run で代表セッションが見つからなかった)")
    print("Done.")


if __name__ == "__main__":
    main()
