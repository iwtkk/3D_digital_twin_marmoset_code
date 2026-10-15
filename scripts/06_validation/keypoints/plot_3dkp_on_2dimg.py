# plotting rotated rays about X axis by -90 degrees (no Z=0 plane),
# marking camera centers with 'x' and labeling by camera_id.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reprojection_error"))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# load_config.py
import yaml
import pickle
import os
import argparse
from utils.aniposelib.cameras import CameraGroup
import plot_ray_repr_error as prre
import utils.frame_to_imgs as fti

def plot_p2d_flat(ax, observed_points, reprojected_points=None, highlight_idx=None):
    """
    Overlay detected 2D keypoints (and optional reprojected points) on the
    currently displayed image.
    """
    pts_obs = np.asarray(observed_points, dtype=float)
    if pts_obs.ndim != 2 or pts_obs.shape[1] != 2:
        raise ValueError(f"observed_points must have shape (N, 2); got {pts_obs.shape}")

    obs_valid = ~np.isnan(pts_obs).any(axis=1)
    if obs_valid.any():
        ax.scatter(pts_obs[obs_valid, 0],
                   pts_obs[obs_valid, 1],
                   s=35,
                   c="#50fa7b",
                   label="2D detection")
        for kp_idx in np.where(obs_valid)[0]:
            ax.text(pts_obs[kp_idx, 0] + 2,
                    pts_obs[kp_idx, 1] + 2,
                    str(kp_idx),
                    fontsize=8,
                    color="#f8f8f2",
                    alpha=0.8)

    if reprojected_points is not None:
        pts_reproj = np.asarray(reprojected_points, dtype=float)
        if pts_reproj.shape != pts_obs.shape:
            raise ValueError(
                f"reprojected_points must match observed shape {pts_obs.shape}; got {pts_reproj.shape}"
            )
        reproj_valid = ~np.isnan(pts_reproj).any(axis=1)
        if reproj_valid.any():
            ax.scatter(pts_reproj[reproj_valid, 0],
                       pts_reproj[reproj_valid, 1],
                       s=25,
                       c="#ffb86c",
                       marker="x",
                       label="reprojection")

    if highlight_idx is not None and 0 <= highlight_idx < pts_obs.shape[0]:
        hx, hy = pts_obs[highlight_idx]
        if not np.isnan(hx) and not np.isnan(hy):
            ax.scatter([hx],
                       [hy],
                       s=130,
                       facecolors="none",
                       edgecolors="#f1fa8c",
                       linewidths=2.0,
                       label="target kp")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Avoid duplicated labels when the function is called multiple times on the same axes.
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        ax.legend(list(unique.values()), list(unique.keys()), loc="upper right", fontsize=8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="Results folder (e.g. ../marmo3Dpose/results/3d_v0p8_dark_fix_20/dailylife_...)")
    ap.add_argument("--vidbase", required=True,
                    help="Results folder (e.g. ../marmo3Dpose/results/3d_v0p8_dark_fix_20/dailylife_...)")
    ap.add_argument("--config_yaml", required=True,
                    help="YAML that defines the camera ID order (e.g. ./config/marmo_cj425m/config.yaml)")
    ap.add_argument("--frame", type=int, default=70)
    ap.add_argument("--animal", type=int, default=0)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--kpn", type=int, default=19)
    ap.add_argument("--plot-camera", default=None,
                    help="Visualization target camera (index or camera_id). Defaults to 0.")
    ap.add_argument("--output", default=None, help="Output PNG path. Defaults to outputs/validation/keypoints/.")
    ap.add_argument("--show", action="store_true", help="Show an interactive window after saving.")
    args = ap.parse_args()

    exception_cam_ids = set()
    
    base = args.base.rstrip("/")

    calib_toml = os.path.join(base, "calibration.toml")
    if not os.path.isfile(calib_toml):
        raise FileNotFoundError(f"not found: {calib_toml}")
    
    # Load CameraGroup from calibration.toml (intrinsics scaled, extrinsics optimized)
    cgroup = CameraGroup.load(calib_toml)

    # Fix the camera order to match reconst
    ids = prre.load_ids_from_yaml(args.config_yaml)
    cgroup = cgroup.subset_cameras_names(ids)

    # Load 2D keypoints (after Viterbi filtering)
    kp2d_f = prre.load_kp2d_f(base)
    p2d, scores = prre.build_points2d_for_frame(
        kp2d_f, frame=args.frame, animal=args.animal, score_thresh=args.score_threshold
    )

    n_cam, n_kp, _ = p2d.shape
    print(f"frame={args.frame}, animal={args.animal}")
    print(f"n_cam={n_cam}, n_kp={n_kp}")
    print("camera order:", ids)

    # Reshape to match aniposelib (n_cam, N, 2)
    p2d_flat = p2d.reshape(n_cam, -1, 2)

    with open(os.path.join(base, 'kp3d_fxdJointLen.pickle'), 'rb') as f:
        pts3d = pickle.load(f)
    pts3d = pts3d['kp3d'][0][args.frame]
    
    # Triangulation (no optimization) -> reprojection error
    # pts3d = cgroup.triangulate(p2d_flat, progress=False)        # shape: (N,3)
    err_mean = cgroup.reprojection_error(pts3d, p2d_flat, mean=True)  # (N,)

    print("\n[recomputed mean reprojection error per keypoint (px)]")
    print(err_mean)
    
    pts3d_prj = cgroup.project(pts3d)
    
    tmp = "tmp"
    img_paths = fti.base_frame_to_imgs(
        args.frame,
        args.vidbase,
        tmp,
        session_id=os.path.basename(base),
        camera_ids=ids,
        result_dir=base,
    )
    if not img_paths:
        print("No frame images were extracted; skipping 2D overlay.")
        return

    if len(img_paths) != n_cam:
        print(f"[warn] extracted {len(img_paths)} images for {n_cam} cameras. "
              "Proceeding with the overlapping subset.")

    if args.kpn < 0 or args.kpn >= n_kp:
        raise IndexError(f"kpn {args.kpn} out of range for {n_kp} keypoints")

    nan_mask = np.isnan(p2d_flat[:, args.kpn, :]).any(axis=1)
    
    target_err = float(err_mean[args.kpn]) if 0 <= args.kpn < len(err_mean) else np.nan
    target_camera_idx = None
    if args.plot_camera is not None:
        if str(args.plot_camera).isdigit() and int(args.plot_camera) < n_cam:
            target_camera_idx = int(args.plot_camera)
        elif str(args.plot_camera) in ids:
            target_camera_idx = ids.index(str(args.plot_camera))
        else:
            raise ValueError(f"plot-camera must be a camera index or one of {ids}: {args.plot_camera}")

    plotted = False

    for cam_idx, (img_file, is_nan) in enumerate(zip(img_paths, nan_mask)):
        if target_camera_idx is not None and cam_idx != target_camera_idx:
            continue
        if is_nan:
            continue
        img_data = plt.imread(img_file)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.imshow(img_data)
        reprojected = pts3d_prj[cam_idx] if pts3d_prj is not None else None
        plot_p2d_flat(
            ax,
            observed_points=p2d_flat[cam_idx],
            reprojected_points=reprojected,
            highlight_idx=args.kpn,
        )

        try:
            cam_id = ids[cam_idx]
            cam_int = int(cam_id)
            cam_label = prre.cam_map.get(cam_int, f"Cam {cam_id}")
        except (ValueError, TypeError, IndexError):
            cam_label = f"Cam {cam_idx}"

        if np.isnan(target_err):
            title_err = "err: n/a"
        else:
            title_err = f"err: {target_err:.2f}px"

        ax.set_title(f"{cam_label} frame {args.frame} kp#{args.kpn} ({title_err})")

        fig.tight_layout()
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = (
                Path("outputs")
                / "validation"
                / "keypoints"
                / f"{os.path.basename(base)}_cam{ids[cam_idx]}_frame{args.frame:07d}_kp{args.kpn:02d}.png"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"saved: {output_path}")
        if args.show:
            plt.show()
        plt.close(fig)
        plotted = True
        break

    if not plotted:
        print(f"Keypoint {args.kpn} is NaN for every camera; nothing to plot.")
    


if __name__ == "__main__":
    main()
