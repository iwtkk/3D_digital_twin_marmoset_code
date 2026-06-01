#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import argparse
import json
import os
import pickle

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import utils.plot_cameras_frustum as pcf
from marmo3d.validation.utils import rodrigues_to_R

def load_extrinsics(h5_path):
    # "Return dict cam_id -> (R, t, C) using OpenCV convention x_c = R X + t."
    out = {}
    with h5py.File(h5_path, "r") as f:
        for cam_id in f.keys():
            if isinstance(f[cam_id], h5py.Group):
                g = f[cam_id]
                if "rvec" in g and "tvec" in g:
                    rvec = np.array(g["rvec"]).reshape(3)
                    tvec = np.array(g["tvec"]).reshape(3)
                    R = rodrigues_to_R(rvec)
                    C = - R.T @ tvec  # camera center in world
                    out[str(cam_id)] = (R, tvec, C)
    return out

def load_intrinsics(h5_path):
    out = {}
    with h5py.File(h5_path, "r") as f:
        for cam_id in f.keys():
            if isinstance(f[cam_id], h5py.Group):
                g = f[cam_id]
                # 理想内パラ
                K = np.array(g["K"]) if "K" in g else np.array(g["mtx"])
                size = tuple(map(int, np.array(g["size"]).tolist())) if "size" in g else None
                # 歪みパラ（omni対応）
                D = np.array(g["D"]).ravel() if "D" in g else (np.array(g["dist"]).ravel() if "dist" in g else None)
                xi = float(np.array(g["xi"]).ravel()[0]) if "xi" in g else None
                out[str(cam_id)] = (K, size, D, xi)
    return out


def kp2d_json_path(kp2d_dir, scene, cam_id, frame):
    shard = frame // 10000
    return os.path.join(
        kp2d_dir,
        f"{scene}_{cam_id}_{shard:06d}.json"
    ), frame % 10000  # local_idx

def load_ui_from_json_any(kp2d_dir, scene, ref_cam, frame, kp_index):
    path, local_idx = kp2d_json_path(kp2d_dir, scene, ref_cam, frame)
    with open(path, "r") as f:
        j = json.load(f)

    # --- Case A: 配列ベース [[[id,cx,cy,w,h, [[x,y,s],...]]], ...] の形 ---
    if isinstance(j, list):
        # フレーム順に list が並んでいる場合（1フレーム=1要素 or 複数検出）
        #   - local_idx 番目が「そのフレームの検出群」パターン
        #   - もしくは「全要素が検出」でフレーム概念が無いパターン
        # 両方をカバーするため、まず local_idx を優先して安全に拾い、
        # それが「検出ひとつ」ならそのまま、配列の配列なら候補群として扱う。
        elt = None
        if 0 <= local_idx < len(j):
            elt = j[local_idx]
        else:
            # フレーム概念が無ければ全体を候補群とみなす
            elt = j

        # 候補をリスト化
        cand = elt if (isinstance(elt, list) and len(elt) > 0 and isinstance(elt[0], list)) else [elt]

        # スコア最大の検出を選ぶ
        best = None
        best_score = -1.0
        for det in cand:
            # det 例: [id, cx, cy, w, h, [[x,y,score],...]]
            if not (isinstance(det, list) and len(det) >= 6 and isinstance(det[5], list)):
                continue
            kps = det[5]
            if not (0 <= kp_index < len(kps)):
                continue
            # 平均scoreで良い個体を選ぶ
            scores = [float(p[2]) for p in kps if isinstance(p, list) and len(p) >= 3]
            mean_s = sum(scores) / max(1, len(scores)) if scores else 0.0
            if mean_s > best_score:
                best_score = mean_s
                best = det
        if best is None:
            raise RuntimeError(f"No valid detection in {os.path.basename(path)} for local_idx={local_idx}")

        u, v, _ = best[5][kp_index]
        return np.array([float(u), float(v)], dtype=float)

    else:
        raise TypeError(f"Unsupported JSON structure in {os.path.basename(path)}")


def fundamental_from_pose(Ki, Kj, Ri, ti, Rj, tj):
    """
    F_j_i: 点(カメラi) -> 線(カメラj)
    OpenCV: x_i = R_i X + t_i, x_j = R_j X + t_j

    相対姿勢（i→j）:
      R_ji = R_j R_i^T
      t_ji = t_j - R_j R_i^T t_i
    （※ ここが重要。t = tj - Rj @ ti ではダメ）
    E = [t_ji]_x R_ji
    F = K_j^{-T} E K_i^{-1}
    """
    R_ji = Rj @ Ri.T
    t_ji = tj - Rj @ Ri.T @ ti
    tx = np.array([[0, -t_ji[2], t_ji[1]],
                   [t_ji[2], 0, -t_ji[0]],
                   [-t_ji[1], t_ji[0], 0]], dtype=float)
    E = tx @ R_ji
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    return F

def backproject_ray_world(K, R, t, u, v):
    """
    OpenCV: x_c = R X + t
    画像点(u,v,1)を、ワールドでの方向ベクトルに直す:
      d_cam = K^{-1} [u,v,1]^T
      d_world = R^T d_cam   （単位化）
    カメラ中心 C = -R^T t
    """
    d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=float)
    d_world = R.T @ d_cam
    d_world = d_world / (np.linalg.norm(d_world) + 1e-12)
    C = - R.T @ t
    return C, d_world

def undistort_to_ideal_pixel(u, v, K, D=None, xi=None):
    """
    入力: 歪みありの画素座標 (u,v) @ 2048x1536
    出力: 歪み無し「理想ピクセル」u',v' （= K * 正規化座標）
    """
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)  # shape (1,1,2)
    if xi is not None and D is not None and hasattr(cv2, "omnidir"):
        # omnidir: 出力は正規化座標 (x,y)、z=1 と解釈
        xy = cv2.omnidir.undistortPoints(pts, K, D, xi, R=np.eye(3), P=np.eye(3))  # (1,1,2)
        x, y = xy[0,0,0], xy[0,0,1]
        uvw = K @ np.array([x, y, 1.0])
        return float(uvw[0]), float(uvw[1])
    else:
        # pinhole: 正規化座標へ -> K を掛けて理想ピクセルへ
        xy = cv2.undistortPoints(pts, K, D if D is not None else None, P=np.eye(3))  # (1,1,2) 正規化
        x, y = xy[0,0,0], xy[0,0,1]
        uvw = K @ np.array([x, y, 1.0])
        return float(uvw[0]), float(uvw[1])


def line_image_bounds_intersections(l, width, height):
    """Given line l=[a,b,c]^T in pixel coords (ax+by+c=0), return up to two intersections."""
    a, b, c = l
    pts = []
    if abs(b) > 1e-12:
        y = -c / b
        if 0 <= y <= height: pts.append((0.0, float(y)))
        y = -(a * width + c) / b
        if 0 <= y <= height: pts.append((float(width), float(y)))
    if abs(a) > 1e-12:
        x = -c / a
        if 0 <= x <= width: pts.append((float(x), 0.0))
        x = -(b * height + c) / a
        if 0 <= x <= width: pts.append((float(x), float(height)))
    uniq = []
    for p in pts:
        if p not in uniq:
            uniq.append(p)
    if len(uniq) > 2:
        dmax = -1
        keep = (uniq[0], uniq[1])
        for i in range(len(uniq)):
            for j in range(i+1, len(uniq)):
                d = (uniq[i][0]-uniq[j][0])**2 + (uniq[i][1]-uniq[j][1])**2
                if d > dmax:
                    dmax = d; keep = (uniq[i], uniq[j])
        return keep
    return tuple(uniq)

def pixel_to_frustum_plane_xy(u, v, width, height, half_w, half_h):
    """Map pixel (u,v) to plane coords used by frustum plot (linear mapping)."""
    x = (u / width) * (2*half_w) - half_w
    y = -((v / height) * (2*half_h) - half_h)  # invert y (image down -> plane up)
    return x, y

def draw_line_on_frustum(ax, C, right, up, fwd, p1_pix, p2_pix, width, height, near, half_w, half_h, **plot_kwargs):
    x1, y1 = pixel_to_frustum_plane_xy(p1_pix[0], p1_pix[1], width, height, half_w, half_h)
    x2, y2 = pixel_to_frustum_plane_xy(p2_pix[0], p2_pix[1], width, height, half_w, half_h)
    Pc = C + fwd * near
    P1 = Pc + right * x1 + up * y1
    P2 = Pc + right * x2 + up * y2
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], **plot_kwargs)

def load_kp3d(pickle_path):
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)
    X = np.array(d["kp3d"])  # could be (frames,K,3) or (N,frames,K,3)
    if X.ndim == 4:
        X = X[0]
    return X

def project_point(K, R, t, Xw):
    x = R @ Xw + t
    if abs(x[2]) < 1e-9:
        x[2] = 1e-9
    uvw = K @ (x / x[2])
    return uvw[:2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extrin_h5", required=True, help="Path to cam_extrinsic_optim.h5")
    ap.add_argument("--intrin_h5", required=True, help="Path to cam_intrinsic.h5")
    ap.add_argument("--pickle3d", required=True, help="Path to kp3d_fxdJointLen.pickle or kp3d.pickle")
    ap.add_argument("--frame", type=int, default=100, help="Frame index (0-based)")
    ap.add_argument("--kp_index", type=int, default=9, help="Keypoint index for epipolar lines")
    ap.add_argument("--ref_cam", type=str, default=None, help="Reference camera ID (e.g. '23506237')")
    ap.add_argument("--width", type=int, default=2048, help="Image width")
    ap.add_argument("--height", type=int, default=1536, help="Image height")
    ap.add_argument("--rotate_x", type=float, default=-90.0, help="World rotation (deg) for view consistency")
    ap.add_argument("--near", type=float, default=150.0)
    ap.add_argument("--half_w", type=float, default=120.0)
    ap.add_argument("--half_h", type=float, default=80.0)
    ap.add_argument("--axis_len", type=float, default=70.0)
    ap.add_argument("--use_2d", action="store_true", help="参照2DをJSONから使う")
    ap.add_argument("--kp2d_dir", type=str, default=None, help="2D JSONディレクトリ")
    ap.add_argument("--scene", type=str, default=None, help="シーン名")
    ap.add_argument("--undistort_2d", action="store_true", help="2Dを歪み補正してから使う")
    ap.add_argument("--save", type=str, default="outputs/validation/reprojection_error/frustum_kps_epi.png")
    ap.add_argument("--show", action="store_true", help="Open an interactive window.")
    args = ap.parse_args()

    # Load cams for frustum plotting
    cams = pcf.load_cameras(args.extrin_h5)
        
    if args.rotate_x != 0.0:
        Rx = pcf.rotation_matrix_x(args.rotate_x)
        cams = pcf.apply_world_rotation(cams, Rx)

    # Create base frustum figure
    fig, ax = pcf.plot_frustums(cams, near=args.near, half_w=args.half_w, half_h=args.half_h,
                                axis_len=args.axis_len, title=f"Cameras — frustum + 3D KPs (f={args.frame})")


    # Load 3D
    X3d = load_kp3d(args.pickle3d)  # (frames, K, 3)
    if not (0 <= args.frame < X3d.shape[0]):
        raise ValueError(f"Frame {args.frame} out of range 0..{X3d.shape[0]-1}")
    if not (0 <= args.kp_index < X3d.shape[1]):
        raise ValueError(f"kp_index {args.kp_index} out of range 0..{X3d.shape[1]-1}")
    P3_world = X3d[args.frame]  # (K,3)
    P3 = P3_world.copy()
    
    if abs(args.rotate_x) > 1e-9:
        ang = math.radians(args.rotate_x)
        Rx_pts = np.array([[1,0,0],
                        [0, math.cos(ang), -math.sin(ang)],
                        [0, math.sin(ang),  math.cos(ang)]])
        P3 = (Rx_pts @ P3.T).T

    # Overlay 3D keypoints
    ax.scatter(P3[:,0], P3[:,1], P3[:,2], s=12, depthshade=False)
    Xref = P3[args.kp_index]
    Xref_world = P3_world[args.kp_index]
    ax.scatter([Xref[0]],[Xref[1]],[Xref[2]], s=60, marker="o")  # highlight

    # Intrinsics & extrinsics for F
    extr = load_extrinsics(args.extrin_h5)
    intr = load_intrinsics(args.intrin_h5)

    # Choose reference camera
    cam_ids = [cid for cid, *_ in cams]
    ref_cam = args.ref_cam if args.ref_cam is not None else cam_ids[0]
    if ref_cam not in extr or ref_cam not in intr:
        raise ValueError(f"Reference camera {ref_cam} not found. Available: {list(extr.keys())}")

    ref_cam = str(args.ref_cam if args.ref_cam is not None else cam_ids[0])
    Ki, size_i, Di, xi_i = intr[ref_cam]
    Ri, ti, _Ci = extr[ref_cam]

    # Ki, Ri, ti は参照カメラ
    if args.use_2d:
        ui = load_ui_from_json_any(args.kp2d_dir, args.scene, ref_cam, args.frame, args.kp_index)
    else:
        ui = project_point(Ki, Ri, ti, Xref_world)

    Kj_ref, _, Dj_ref, xi_ref = intr[ref_cam]
    ui0, ui1 = undistort_to_ideal_pixel(ui[0], ui[1], Kj_ref, Dj_ref, xi_ref)
    print(ui0,ui1)
    ui_h = np.array([ui0, ui1, 1.0])
    
    # 参照カメラの逆投影レイ（見やすい長さで可視化）
    Ci, di = backproject_ray_world(Ki, Ri, ti, ui0, ui1)
    L = 2000.0  # レイの長さ（任意で調整）
    P1 = Ci
    P2 = Ci + di * L
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], linewidth=2.0)
    
    # Draw epipolar lines on each camera's image plane
    for cam_id, C, fwd, right, up in cams:
        cam_key = str(cam_id)
        Kj, size_j, Dj, xi_j = intr.get(cam_key, (None, None, None, None))
        Rj, tj, _Cj = extr.get(cam_key, (None, None, None))
        if Kj is None or Rj is None:
            print(f"[WARN] missing K/R for cam_id={cam_key}")
            continue
        Fji = fundamental_from_pose(Ki, Kj, Ri, ti, Rj, tj)
        l = Fji @ ui_h  # a*x + b*y + c = 0
        seg = line_image_bounds_intersections(l, args.width-1, args.height-1)
        if len(seg) == 2:
            draw_line_on_frustum(ax, C, right, up, fwd,
                                 seg[0], seg[1],
                                 args.width, args.height,
                                 args.near, args.half_w, args.half_h,
                                 linewidth=2.0)
            
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print("Saved:", save_path)
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
