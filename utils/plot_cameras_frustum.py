#!/usr/bin/env python3
"""
plot_cameras_frustum.py

Visualize camera centers and frustums from an HDF5 file containing OpenCV-style
extrinsics per camera ID group with datasets:
  - rvec: (3,) Rodrigues rotation vector
  - tvec: (3, 1) or (3,) translation vector

By default, this renders a Blender-like external view with an optional
rotation of the world around the X axis (e.g., -90 degrees).

Usage:
  python plot_cameras_frustum.py --h5 cam_extrinsic_optim.h5 --rotate-x -90 --show
  python plot_cameras_frustum.py --h5 cam_extrinsic_optim.h5 --save camera_frustums.png

Dependencies:
  pip install numpy h5py matplotlib
"""
import argparse
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues vector (3,) -> rotation matrix (3,3)."""
    rvec = rvec.reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=float)
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R

def load_cameras(h5_path: str):
    """
    Returns a list of (cam_id, C, fwd, right, up), where:
      C    : camera center in world
      fwd  : camera +Z in world
      right: camera +X in world
      up   : camera +Y in world
    """
    cams = []
    with h5py.File(h5_path, "r") as f:
        for cam_id in f.keys():
            g = f[cam_id]
            if "rvec" not in g or "tvec" not in g:
                continue
            rvec = np.array(g["rvec"]).reshape(3)
            tvec = np.array(g["tvec"]).reshape(3)
            R = rodrigues_to_R(rvec)
            # OpenCV convention: x_c = R X + t  => camera center C = - R^T t
            C = - R.T @ tvec
            fwd = R.T @ np.array([0.0, 0.0, 1.0])   # camera +Z axis in world
            right = R.T @ np.array([1.0, 0.0, 0.0]) # camera +X
            up = R.T @ np.array([0.0, 1.0, 0.0])    # camera +Y
            # Normalize axes
            fwd /= np.linalg.norm(fwd) + 1e-12
            right /= np.linalg.norm(right) + 1e-12
            up /= np.linalg.norm(up) + 1e-12
            cams.append((str(cam_id), C, fwd, right, up))
    return cams

def rotation_matrix_x(deg: float) -> np.ndarray:
    """World rotation about X by given degrees."""
    ang = math.radians(deg)
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)

def apply_world_rotation(cams, Rx):
    """Rotate all camera positions and axes by Rx (3x3)."""
    out = []
    for cam_id, C, fwd, right, up in cams:
        Cr = Rx @ C
        fwdr = Rx @ fwd
        rightr = Rx @ right
        upr = Rx @ up
        # re-normalize
        fwdr /= np.linalg.norm(fwdr) + 1e-12
        rightr /= np.linalg.norm(rightr) + 1e-12
        upr /= np.linalg.norm(upr) + 1e-12
        out.append((cam_id, Cr, fwdr, rightr, upr))
    return out

def plot_frustums(cams, near=150.0, half_w=120.0, half_h=80.0, axis_len=70.0,
                  title="Cameras — frustum view"):
    """Plot camera frustums and centers."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    all_pts = []
    for cam_id, C, fwd, right, up in cams:
        # image plane center
        Pc = C + fwd * near
        # rectangle corners
        c1 = Pc - right * half_w - up * half_h
        c2 = Pc + right * half_w - up * half_h
        c3 = Pc + right * half_w + up * half_h
        c4 = Pc - right * half_w + up * half_h

        # pyramid edges
        for p in (c1, c2, c3, c4):
            ax.plot([C[0], p[0]], [C[1], p[1]], [C[2], p[2]], linewidth=1.0)
        # rectangle outline
        rect = np.stack([c1, c2, c3, c4, c1])
        ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], linewidth=1.2)
        # center marker
        ax.scatter([C[0]], [C[1]], [C[2]], s=40, marker="^")
        # ax.text(C[0], C[1], C[2], cam_id, fontsize=8)

        # local axes (short)
        ax.plot([C[0], C[0] + right[0] * axis_len],
                [C[1], C[1] + right[1] * axis_len],
                [C[2], C[2] + right[2] * axis_len], color="k", linewidth=1.1)
        ax.plot([C[0], C[0] + up[0] * axis_len],
                [C[1], C[1] + up[1] * axis_len],
                [C[2], C[2] + up[2] * axis_len], color="k", linewidth=1.1, linestyle="--")
        ax.plot([C[0], C[0] + fwd[0] * axis_len],
                [C[1], C[1] + fwd[1] * axis_len],
                [C[2], C[2] + fwd[2] * axis_len], color="k", linewidth=1.1, linestyle=":")

        all_pts.extend([C, c1, c2, c3, c4])

    if all_pts:
        P = np.array(all_pts)
        max_range = max(P[:, 0].max() - P[:, 0].min(),
                        P[:, 1].max() - P[:, 1].min(),
                        P[:, 2].max() - P[:, 2].min()) / 2.0
        mx, my, mz = P[:, 0].mean(), P[:, 1].mean(), P[:, 2].mean()
        pad = max(100.0, 0.2 * max_range)
        ax.set_xlim(mx - max_range - pad, mx + max_range + pad)
        ax.set_ylim(my - max_range - pad, my + max_range + pad)
        ax.set_zlim(mz - max_range - pad, mz + max_range + pad)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig, ax

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to HDF5 file with rvec/tvec per camera group.")
    ap.add_argument("--rotate-x", type=float, default=-90.0,
                    help="Rotate world around X (degrees). Default: -90 (Blender-like). Use 0 for no rotation.")
    ap.add_argument("--near", type=float, default=150.0, help="Frustum depth from camera center to image plane.")
    ap.add_argument("--half-w", type=float, default=120.0, help="Half width of image plane rectangle.")
    ap.add_argument("--half-h", type=float, default=80.0, help="Half height of image plane rectangle.")
    ap.add_argument("--axis-len", type=float, default=70.0, help="Length of the drawn local camera axes.")
    ap.add_argument("--save", type=str, default=None, help="Path to save PNG.")
    ap.add_argument("--show", action="store_true", help="Open an interactive window.")
    args = ap.parse_args()

    cams = load_cameras(args.h5)

    if args.rotate_x != 0.0:
        Rx = rotation_matrix_x(args.rotate_x)
        cams = apply_world_rotation(cams, Rx)

    fig, ax = plot_frustums(
        cams,
        near=args.near,
        half_w=args.half_w,
        half_h=args.half_h,
        axis_len=args.axis_len,
        title=f"Cameras — frustum view (rotate-x={args.rotate_x}°)"
    )

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print("Saved:", args.save)

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
