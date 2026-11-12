"""
Tool: plot_affine_err_by_camera

Purpose:
    For every session, measure per camera the pixel distance between the real-camera projection
    and the Blender-camera projection of the same 3D keypoints, before (initial placement A0,t0)
    and after the reprojection-fitted affine (A,t), and draw a per-camera box plot with the
    initial (left) and fitted (right) boxes side by side.

    distance (one keypoint, one frame, one camera) =
        || omnidir_cam(kp3d_mm) - equisolid_blender_cam(P) ||_2   [px]
        P = A0 @ X_m + t0 (initial)  or  A @ X_m + t (fit),  X_m = (kp3d_mm @ rot.T) / 1000
    Only keypoints with score > --score-thr that are valid in all three projections are used,
    so initial and fit are compared on the same points. Frames are sampled at random among
    frames NOT used to fit the affine (affine_align.build_targets: seed=1, 300 frames with
    >= 16 KPs of score > 0.4), i.e. this is a held-out evaluation.
    Each box shows the distribution over sessions of the per-session median distance.

Inputs:
    npz:
        type: file
        path: {npz-dir}/{session}.npz   (path_registry.outputs.reproj_aligned_npz), all sessions
        keys: kp3d (T,23,3) m, kp3d_mm (T,20,3) mm, rot (3,3), kp3d_score (T,20), camera_id (8,)
    cameras:
        config/marmo_cj425m/{config.yaml, cam_intrinsic.h5, cam_extrinsic_optim.h5, blender_cam.json}

Outputs:
    per-session table:
        type: csv, one row per (session, camera)
        path: {output-dir}/affine_err_by_camera.csv
        columns: session, animal, camera_id, n_points, init_median_px, init_p90_px,
                 fit_median_px, fit_p90_px
    figure:
        type: png (300 dpi) + pdf
        path: {output-dir}/affine_err_by_camera.{png,pdf}
        content: x = camera, two boxes per camera (initial / affine fit), y = per-session median
                 distance [px] on a log axis

Registry:
    .agent_docs/docs/tool_registry.md#plot_affine_err_by_camera

Usage:
    NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \\
        scripts/05_create_digital_twin/alignment_3D_keypoint/plot/plot_affine_err_by_camera.py --jobs 16
    # redraw from the existing CSV only
    .venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/plot/plot_affine_err_by_camera.py --plot-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/plot"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))

CFG_DIR = ROOT / "config/marmo_cj425m"
N_KP = 20
# fit frames of affine_align.build_targets (must match the run that produced affine_log.npz)
FIT_SEED, FIT_N, FIT_THR, FIT_MIN_KP = 1, 300, 0.4, 16

COL_INIT = "#eb6834"   # categorical slot 2 (orange)
COL_FIT = "#2a78d6"    # categorical slot 1 (blue)

_W = {}


def _init_worker(score_thr, n_frames, seed):
    from blender_fisheye import load_cameras
    from affine_align import _floorz_init
    from plot_real_vs_blender_2d import load_omnidir
    cfg_y = yaml.safe_load(open(CFG_DIR / "config.yaml"))
    cam_ids = [str(c) for c in cfg_y["camera_id"]]
    _W.update(cam_ids=cam_ids, W=cfg_y["img_size"][0], H=cfg_y["img_size"][1],
              omni={c: load_omnidir(c) for c in cam_ids}, bcams=load_cameras(),
              A0t0=_floorz_init(), thr=score_thr, n=n_frames, seed=seed)


def _q(d):
    if len(d) == 0:
        return np.nan, np.nan
    return float(np.median(d)), float(np.percentile(d, 90))


def process(npz_path):
    z = np.load(npz_path)
    session = Path(npz_path).stem
    mm, score = z["kp3d_mm"], z["kp3d_score"]
    assert [str(c) for c in z["camera_id"]] == _W["cam_ids"], session

    # exclude the frames used for the fit
    fin = np.isfinite(mm[:, :, 0])
    fit_pool = np.where((fin & (score > FIT_THR)).sum(1) >= FIT_MIN_KP)[0]
    fit_fr = np.random.default_rng(FIT_SEED).choice(fit_pool, min(FIT_N, len(fit_pool)), replace=False) \
        if len(fit_pool) else np.array([], int)
    pool = np.setdiff1d(np.where((fin & (score > _W["thr"])).sum(1) >= 10)[0], fit_fr)
    if len(pool) == 0:
        return []
    sel = np.sort(np.random.default_rng(_W["seed"]).choice(pool, min(_W["n"], len(pool)), replace=False))

    X_mm = mm[sel].reshape(-1, 3)
    ok = np.isfinite(X_mm[:, 0]) & (score[sel].reshape(-1) > _W["thr"])
    X_m = (X_mm @ z["rot"].T) / 1000.0
    A0, t0 = _W["A0t0"]
    P_init = X_m @ A0.T + t0
    P_fit = z["kp3d"][sel, :N_KP].reshape(-1, 3)
    W, H = _W["W"], _W["H"]

    rows = []
    animal = session.split("_")[0]
    for c in _W["cam_ids"]:
        real = _W["omni"][c](X_mm)
        vr = ok & np.isfinite(real).all(1) & (real[:, 0] >= 0) & (real[:, 0] < W) & \
            (real[:, 1] >= 0) & (real[:, 1] < H)
        init, vi = _W["bcams"][c].project(np.nan_to_num(P_init))
        fit, vf = _W["bcams"][c].project(np.nan_to_num(P_fit))
        m = vr & vi & vf
        di = np.linalg.norm(init[m] - real[m], axis=1)
        df = np.linalg.norm(fit[m] - real[m], axis=1)
        rows.append([session, animal, c, int(m.sum()), *_q(di), *_q(df)])
    return rows


def plot(df, cam_ids, out_png, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_sess = df["session"].nunique()
    fs = 12                                     # base font size (1.2x matplotlib default)
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs * 0.9,
                         "ytick.labelsize": fs, "legend.fontsize": fs * 0.9})
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    w, gap = 0.34, 0.04
    for k, c in enumerate(cam_ids):
        d = df[df["camera_id"] == c]
        for vals, off, col in [(d["init_median_px"].dropna(), -(w + gap) / 2, COL_INIT),
                               (d["fit_median_px"].dropna(), (w + gap) / 2, COL_FIT)]:
            ax.boxplot([vals], positions=[k + off], widths=w, patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=col, edgecolor=col, alpha=0.35, linewidth=1),
                       medianprops=dict(color=col, linewidth=2),
                       whiskerprops=dict(color=col, linewidth=1), capprops=dict(color=col, linewidth=1),
                       flierprops=dict(marker="o", markersize=2.5, markerfacecolor=col,
                                       markeredgecolor="none", alpha=0.6))
    ax.set_yscale("log")
    ax.set_xticks(range(len(cam_ids)))
    ax.set_xticklabels(cam_ids)
    ax.set_xlabel("Camera ID")
    ax.set_ylabel("Real vs. digital-twin projection\nper-session median distance [px]")
    ax.set_xlim(-0.6, len(cam_ids) - 0.4)
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.6)
    ax.grid(axis="y", which="minor", color="#eeeeee", linewidth=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    mi, mf = df["init_median_px"].median(), df["fit_median_px"].median()
    ax.legend(handles=[Patch(facecolor=COL_INIT, alpha=0.6, label=f"Initial placement (median {mi:.1f} px)"),
                       Patch(facecolor=COL_FIT, alpha=0.6, label=f"Affine fit (median {mf:.1f} px)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    ax.text(1.0, 1.0, f"n = {n_sess} sessions per box", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=fs * 0.8, color="#555555")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=str(ROOT / "outputs/reproj_eval/aligned_npz"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval/affine_err_by_camera"))
    ap.add_argument("--n-frames", type=int, default=500, help="held-out frames per session")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true", help="redraw from the existing CSV")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "affine_err_by_camera.csv"
    cam_ids = [str(c) for c in yaml.safe_load(open(CFG_DIR / "config.yaml"))["camera_id"]]

    if not args.plot_only:
        import multiprocessing as mp
        paths = sorted(Path(args.npz_dir).glob("*.npz"))
        rows = []
        with mp.Pool(args.jobs, initializer=_init_worker,
                     initargs=(args.score_thr, args.n_frames, args.seed)) as pool:
            for k, r in enumerate(pool.imap_unordered(process, paths), 1):
                rows += r
                if k % 20 == 0 or k == len(paths):
                    print(f"  {k}/{len(paths)} sessions", flush=True)
        df = pd.DataFrame(rows, columns=["session", "animal", "camera_id", "n_points",
                                         "init_median_px", "init_p90_px", "fit_median_px", "fit_p90_px"])
        df = df.sort_values(["session", "camera_id"]).round(3)
        df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path, dtype={"camera_id": str})

    plot(df, cam_ids, out / "affine_err_by_camera.png", out / "affine_err_by_camera.pdf")
    summ = df.groupby("camera_id")[["init_median_px", "fit_median_px"]].median().reindex(cam_ids)
    print(summ.round(2).to_string())
    print(f"sessions: {df['session'].nunique()}  output: {out}")


if __name__ == "__main__":
    main()
