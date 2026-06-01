# plotting rotated rays about X axis by -90 degrees (no Z=0 plane),
# marking camera centers with 'x' and labeling by camera_id.

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from utils.aniposelib.cameras import CameraGroup
from marmo3d.validation.utils import (
    cam_map, load_ids_from_yaml, load_kp2d_f, build_points2d_for_frame, rodrigues_to_R,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = [abs(x_limits[1]-x_limits[0]),
              abs(y_limits[1]-y_limits[0]),
              abs(z_limits[1]-z_limits[0])]
    centers = [(x_limits[0]+x_limits[1])/2.0,
               (y_limits[0]+y_limits[1])/2.0,
               (z_limits[0]+z_limits[1])/2.0]
    max_range = max(ranges)
    for ctr, getlim, setlim in zip(centers,
                                   [ax.get_xlim3d, ax.get_ylim3d, ax.get_zlim3d],
                                   [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        setlim(ctr - max_range/2, ctr + max_range/2)


def plot_camera_rays(pts2d, pts3d, all_results, cam_label_map,
                     exception_cam_ids=None, highlight_idx=None, repr_error=None,
                     output=None, show=False):
    """
    Visualize camera rays and 3D keypoints. Expects:
      pts2d: iterable with per-camera arrays shaped (N, 2)
      pts3d: array-like shaped (N, 3)
      all_results: iterable of dicts with keys `C`, `d`, `camera_id`,
                   optionally `lambda_Z0_orig`
      cam_label_map: dict camera_id -> display label
    """
    pts3d_arr = np.asarray(pts3d, dtype=float)
    if pts3d_arr.ndim != 2 or pts3d_arr.shape[1] != 3:
        raise ValueError("pts3d must have shape (N, 3)")

    num_cams = len(pts2d)
    num_points = pts2d[0].shape[0] if num_cams and hasattr(pts2d[0], "shape") else pts3d_arr.shape[0]

    angle = -np.pi / 2.0
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle),  np.cos(angle)]])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title(f"n_cameras={num_cams}, n_points={num_points}")
    ax.set_title(f"Keypoint number:{highlight_idx+1}, Reprojection error:{repr_error:.4f}")

    xyz_stack = []

    color_cycle = plt.rcParams.get('axes.prop_cycle')
    colors = color_cycle.by_key().get('color') if color_cycle else None

    for idx, res in enumerate(all_results):
        C = np.asarray(res["C"], dtype=float).reshape(3)
        d = np.asarray(res["d"], dtype=float).reshape(3)

        C_r = (Rx @ C).ravel()
        d_r = (Rx @ d).ravel()

        L_back, L_fwd = 1000.0, 3000.0
        lam = np.linspace(-L_back, L_fwd, 200)
        ray = C_r[None, :] + lam[:, None] * d_r[None, :]

        color = colors[idx % len(colors)] if colors else None

        # ax.scatter([C_r[0]], [C_r[1]], [C_r[2]], marker='x', s=40, color=color)
        ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color=color)

        cam_id = res.get("camera_id")
        cam_label = cam_label_map.get(cam_id, str(cam_id)) if cam_label_map else str(cam_id)
        ax.text(C_r[0], C_r[1], C_r[2], cam_label, fontsize=15, color=color)

        xyz_stack.extend([C_r, ray[0], ray[-1]])

    highlight_idx = highlight_idx if highlight_idx is not None else -1
    for idx, kp in enumerate(pts3d_arr):
        v_rot = (Rx @ kp.reshape(3))
        if idx == highlight_idx:
            ax.scatter(v_rot[0], v_rot[1], v_rot[2], marker='o', s=70, alpha=1.0)
        else:
            ax.scatter(v_rot[0], v_rot[1], v_rot[2], marker='o', s=50, alpha=0.2)
        xyz_stack.append(v_rot)

    if xyz_stack:
        xyz = np.vstack(xyz_stack)
        mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
        center = (mins + maxs) / 2.0
        span = max(1.0, (maxs - mins).max())
        ax.set_xlim(center[0]-span/2, center[0]+span/2)
        ax.set_ylim(center[1]-span/2, center[1]+span/2)
        ax.set_zlim(center[2]-span/2, center[2]+span/2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)

    # Preserve manual limits from previous visualization for consistency.
    ax.set_xlim([800, 1600])
    ax.set_ylim([600, 1400])
    ax.set_zlim([0, 800])

    if exception_cam_ids:
        rows = []
        for res in all_results:
            if res.get("camera_id") in exception_cam_ids:
                C_r = (Rx @ np.asarray(res["C"], dtype=float).reshape(3)).ravel()
                d_r = (Rx @ np.asarray(res["d"], dtype=float).reshape(3)).ravel()
                lambda_z0 = res.get("lambda_Z0_orig")
                try:
                    lambda_val = float(lambda_z0)
                except (TypeError, ValueError):
                    lambda_val = np.nan
                rows.append({
                    "camera_id": res["camera_id"],
                    "C_x_rot": float(C_r[0]),
                    "C_y_rot": float(C_r[1]),
                    "C_z_rot": float(C_r[2]),
                    "d_x_rot": float(d_r[0]),
                    "d_y_rot": float(d_r[1]),
                    "d_z_rot": float(d_r[2]),
                    "lambda_Z0_orig": lambda_val if not np.isnan(lambda_val) else np.nan
                })
        if rows:
            df = pd.DataFrame(rows)
            print(df.head())

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"saved: {output}")

    if show or output is None:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="結果フォルダ（例: ../marmo3Dpose/results/3d_v0p8_dark_fix_20/dailylife_...）")
    ap.add_argument("--config_yaml", required=True,
                    help="カメラID順序を読む YAML（例: ./config/marmo_cj425m/config.yaml）")
    ap.add_argument("--frame", type=int, default=70)
    ap.add_argument("--animal", type=int, default=0)
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--kpn", type=int, default=0)
    ap.add_argument("--output", default=None, help="Output PNG path.")
    ap.add_argument("--show", action="store_true", help="Show an interactive window after saving.")
    args = ap.parse_args()

    exception_cam_ids = set()
    
    base = args.base.rstrip("/")

    calib_toml = os.path.join(base, "calibration.toml")
    if not os.path.isfile(calib_toml):
        raise FileNotFoundError(f"not found: {calib_toml}")
    
    # CameraGroup を calibration.toml からロード（内参スケール済み・外参最適化済み）
    cgroup = CameraGroup.load(calib_toml)

    # カメラ順序を reconst と同じに固定
    ids = load_ids_from_yaml(args.config_yaml)
    cgroup = cgroup.subset_cameras_names(ids)

    # 2D（Viterbiフィルタ後）をロード
    kp2d_f = load_kp2d_f(base)
    p2d, scores = build_points2d_for_frame(
        kp2d_f, frame=args.frame, animal=args.animal, score_thresh=args.score_threshold
    )

    n_cam, n_kp, _ = p2d.shape
    print(f"frame={args.frame}, animal={args.animal}")
    print(f"n_cam={n_cam}, n_kp={n_kp}")
    print("camera order:", ids)

    # 形を aniposelib に合わせる (n_cam, N, 2)
    p2d_flat = p2d.reshape(n_cam, -1, 2)

    with open(os.path.join(base, 'kp3d_fxdJointLen.pickle'), 'rb') as f:
        pts3d = pickle.load(f)
    pts3d = pts3d['kp3d'][0][args.frame]
    
    # 三角測量（最適化なし）→ 投影誤差
    # pts3d = cgroup.triangulate(p2d_flat, progress=False)        # shape: (N,3)
    err_mean = cgroup.reprojection_error(pts3d, p2d_flat, mean=True)  # (N,)

    print("\n[recomputed mean reprojection error per keypoint (px)]")
    print(err_mean)
    
    # ------------------------------------------------------------
    # Build rays for all cameras using CameraGroup (Omnidir-aware)
    # ------------------------------------------------------------
    all_results = []
    pts3d_arr = np.asarray(pts3d, dtype=np.float64)

    for cam_idx, cam in enumerate(cgroup.cameras):
        cam_name = ids[cam_idx]
        try:
            cam_id = int(cam_name)
        except (TypeError, ValueError):
            cam_id = cam_name

        p2d_cam = p2d_flat[cam_idx]
        u, v = p2d_cam[args.kpn]
        conf = scores[cam_idx, args.kpn]

        if np.isnan(u) or np.isnan(v):
            exception_cam_ids.add(cam_id)
            continue

        undist = cam.undistort_points(np.array([[u, v]], dtype=np.float64)).ravel()
        d_cam = np.array([undist[0], undist[1], 1.0], dtype=float)
        d_cam /= np.linalg.norm(d_cam)

        rvec = np.asarray(cam.get_rotation(), dtype=float)
        tvec = np.asarray(cam.get_translation(), dtype=float)
        R = rodrigues_to_R(rvec)

        C_w = (-R.T @ tvec).ravel()
        d_w = (R.T @ d_cam).ravel()
        d_w /= np.linalg.norm(d_w)

        lam_z0 = -C_w[2] / d_w[2] if abs(d_w[2]) > 1e-8 else np.nan

        all_results.append({
            "camera_id": cam_id,
            "C": C_w,
            "d": d_w,
            "u": float(u),
            "v": float(v),
            "conf": float(conf),
            "lambda_Z0_orig": float(lam_z0) if np.isfinite(lam_z0) else np.nan
        })

    plot_camera_rays(
        pts2d=p2d_flat,
        pts3d=pts3d_arr,
        all_results=all_results,
        cam_label_map=cam_map,
        exception_cam_ids=exception_cam_ids,
        highlight_idx=args.kpn,
        repr_error=err_mean[args.kpn],
        output=args.output,
        show=args.show,
    )

if __name__ == "__main__":
    main()
