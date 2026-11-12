"""
Tool: plot_real_vs_blender_2d

Purpose:
    Overlay, on real camera frames, the 2D keypoints obtained by projecting the same 3D keypoints
    (1) with the real camera parameters and (2) with the Blender cameras, before and after the
    reprojection-fitted affine alignment. Visualises how far the initial placement of the
    animal in the digital twin was off, and how the per-session affine corrects it.

    red      real camera   : kp3d_mm projected with cv2.omnidir (cam_intrinsic.h5 +
                             cam_extrinsic_optim.h5), i.e. the cameras used for triangulation
    cyan     Blender, init : P = A0 @ X_m + t0  (fixed rotation + fixed per-axis scale +
                             fixed translation, common to all sessions; affine_align._floorz_init)
    magenta  Blender, fit  : P = A  @ X_m + t   (per-session affine fitted by minimising the
                             reprojection error; affine_A / affine_t in the NPZ)
    where X_m = (kp3d_mm @ rot.T) / 1000. Blender projection uses the FISHEYE_EQUISOLID model
    (blender_fisheye.py). red and magenta never coincide exactly: the real (omnidir) and Blender
    (equisolid) lens models differ, leaving ~7 px even after the fit.

Inputs:
    npz:
        type: file
        path: {npz-dir}/{session}.npz   (path_registry.outputs.reproj_aligned_npz)
        keys: kp3d (T,23,3) float64 m, kp3d_mm (T,20,3) float64 mm, rot (3,3), affine_A (3,3),
              affine_t (3,), kp3d_score (T,20), camera_id (C,), frame_num (C,T) int64,
              frame_min (C,) int64
    video:
        type: directory
        path: {vid-dir}/{session}.{camera_id}/metadata.yaml   (imgstore, 2048x1536,
              frames are returned in BGR order like cv2.VideoCapture; used as is)
    cameras:
        config/marmo_cj425m/{config.yaml, cam_intrinsic.h5, cam_extrinsic_optim.h5, blender_cam.json}

Outputs:
    overlay images:
        type: jpg, 2048x1536 BGR uint8
        path: {output-dir}/{session}/{session}_frame{i:06d}_cam{camera_id}.jpg
    grid:
        type: png, 2x4 crops around the animal (one per camera, layout --grid-order), cell 600x600
        path: {output-dir}/{session}/{session}_frame{i:06d}_grid.png
        legend: "error" = mean distance [px] to the real-camera projection over all cameras x KPs of the
                frame (KPs with score > --score-thr valid in all three projections)
    clean crop (--clean-cams):
        type: png 600x600, the grid cell of that camera without overlays
        path: {output-dir}/{session}/{session}_frame{i:06d}_cam{camera_id}_clean.png
              and ..._clean_real.png (same crop with the real-camera projection, red)
    external crop (--extra-image CAM=PATH):
        type: png 600x600, PATH cropped with the same window as the grid cell of CAM
        path: {output-dir}/{session}/{session}_frame{i:06d}_cam{camera_id}_{extra-tag}.png
              and ..._{extra-tag}_init.png / ..._{extra-tag}_fit.png (same crop with the Blender-camera
              projection of the initial placement, cyan / of the affine fit, magenta)
    stats:
        type: csv, one row per camera + "all"
        path: {output-dir}/{session}/{session}_px_diff.csv
        columns: camera_id, n_points, init_median_px, init_p90_px, fit_median_px, fit_p90_px
        content: distance between red and cyan (init) / red and magenta (fit) over
                 --n-stat-frames randomly sampled frames (KPs with score > --score-thr,
                 inside both images)

Registry:
    .agent_docs/docs/tool_registry.md#plot_real_vs_blender_2d

Usage:
    .venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/plot/plot_real_vs_blender_2d.py \\
        --session cj357_20221009_090000 --n-frames 3
    # specific kp3d frame indices
    .venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/plot/plot_real_vs_blender_2d.py \\
        --session cj357_20221009_090000 --frames 3000 12000
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import h5py
import imgstore
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))
import _config as cfg                        # noqa: E402
from blender_fisheye import load_cameras      # noqa: E402
from affine_align import _floorz_init         # noqa: E402

CFG_DIR = ROOT / "config/marmo_cj425m"
BONES = cfg.BONES
N_KP = 20

# BGR
COL_REAL = (0, 0, 255)
COL_INIT = (255, 255, 0)
COL_FIT = (255, 0, 255)
# 2x4 grid layout (row-major). Right-most column: 23511613 (top) / 23506239 (bottom).
GRID_ORDER = ["23506237", "23506236", "23511607", "23511613",
              "23511614", "23506214", "23506226", "23506239"]
CELL = 600   # grid cell size [px]
LEGEND_FONT = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"   # Helvetica clone


def load_omnidir(cam_id):
    with h5py.File(CFG_DIR / "cam_intrinsic.h5", "r") as f:
        K = f[f"/{cam_id}/K"][()]; xi = f[f"/{cam_id}/xi"][()]; D = f[f"/{cam_id}/D"][()]
    with h5py.File(CFG_DIR / "cam_extrinsic_optim.h5", "r") as f:
        rv = f[f"/{cam_id}/rvec"][()]; tv = f[f"/{cam_id}/tvec"][()]

    def project(p3d):
        p = np.reshape(p3d, (-1, 1, 3)).astype(np.float64)
        out = np.full((len(p), 2), np.nan)
        ok = np.isfinite(p[:, 0, 0])
        if ok.any():
            pts, _ = cv2.omnidir.projectPoints(p[ok], rv, tv, K, xi[0][0], D)
            out[ok] = pts[:, 0, :]
        return out
    return project


def in_frame(pts, w, h):
    return np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & \
        (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)


def draw_skeleton(img, pts2d, valid, color, radius=5, thick=3):
    for a, b in BONES:
        if a < len(valid) and b < len(valid) and valid[a] and valid[b]:
            cv2.line(img, tuple(np.round(pts2d[a]).astype(int)),
                     tuple(np.round(pts2d[b]).astype(int)), color, thick, cv2.LINE_AA)
    for j in range(len(pts2d)):
        if valid[j]:
            cv2.circle(img, tuple(np.round(pts2d[j]).astype(int)), radius, color, -1, cv2.LINE_AA)


def crop_box(pts_list, W, H, margin=0.35):
    """Square crop window (x0, y0, side) in image pixels around the union of the 2D point sets."""
    pts = np.concatenate([p for p in pts_list if len(p)], axis=0) if any(len(p) for p in pts_list) \
        else np.zeros((0, 2))
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) == 0:
        side = min(H, W)
        return (W - side) // 2, (H - side) // 2, side
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    half = max(hi - lo) * (0.5 + margin)
    half = int(np.clip(half, 120, min(H, W) / 2))
    x0 = int(np.clip(c[0] - half, 0, W - 2 * half)); y0 = int(np.clip(c[1] - half, 0, H - 2 * half))
    return x0, y0, 2 * half


def crop_resize(img, box, size=CELL):
    x0, y0, side = box
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


def legend_bar(width, entries, font_px=56, height=110, swatch=100, gap=60, pad=30):
    """White legend bar: a line+dot swatch in each colour followed by its label (black text),
    centred, in Nimbus Sans (metric-compatible Helvetica clone). The font shrinks until it fits."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    while True:
        font = ImageFont.truetype(LEGEND_FONT, font_px)
        widths = [swatch + 20 + d.textlength(t, font=font) for t, _ in entries]
        if sum(widths) + gap * (len(entries) - 1) <= width - 2 * pad or font_px <= 20:
            break
        font_px -= 2
    x = (width - (sum(widths) + gap * (len(entries) - 1))) / 2
    cy = height / 2
    for (txt, bgr), w in zip(entries, widths):
        rgb = tuple(int(c) for c in bgr[::-1])
        d.line([(x, cy), (x + swatch, cy)], fill=rgb, width=8)
        for px in (x + 8, x + swatch - 8):
            d.ellipse([px - 9, cy - 9, px + 9, cy + 9], fill=rgb)
        d.text((x + swatch + 20, cy), txt, font=font, fill="black", anchor="lm")
        x += w + gap
    return np.asarray(im)[..., ::-1].copy()   # RGB -> BGR for cv2


def put_label(img, lines, org=(15, 35), scale=0.9):
    for k, (txt, col) in enumerate(lines):
        y = org[1] + k * int(36 * scale / 0.9)
        cv2.putText(img, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


class Projector:
    """Projects one kp3d frame with real / Blender-init / Blender-fit cameras."""

    def __init__(self, z, cam_ids, W, H, score_thr):
        self.mm = z["kp3d_mm"]
        self.score = z["kp3d_score"]
        self.X_m = (self.mm @ z["rot"].T) / 1000.0
        A0, t0 = _floorz_init()
        self.P_init = self.X_m @ A0.T + t0
        self.P_fit = z["kp3d"][:, :N_KP]                 # = X_m @ affine_A.T + affine_t
        self.omni = {c: load_omnidir(c) for c in cam_ids}
        self.bcams = load_cameras()
        self.W, self.H, self.thr = W, H, score_thr

    def __call__(self, i, cam_id):
        ok = np.isfinite(self.mm[i, :, 0]) & (self.score[i] > self.thr)
        real = self.omni[cam_id](self.mm[i])
        init, v_init = self.bcams[cam_id].project(self.P_init[i])
        fit, v_fit = self.bcams[cam_id].project(self.P_fit[i])
        return (real, ok & in_frame(real, self.W, self.H)), (init, ok & v_init), (fit, ok & v_fit)


def write_stats(proj, cam_ids, n_frame, n_stat, seed, path):
    rng = np.random.default_rng(seed)
    good = np.where((np.isfinite(proj.mm[:, :, 0]) & (proj.score > proj.thr)).sum(1) >= 10)[0]
    sel = rng.choice(good, size=min(n_stat, len(good)), replace=False)
    rows, d_all = [], {"init": [], "fit": []}
    for cam_id in cam_ids:
        d = {"init": [], "fit": []}
        for i in sel:
            (real, vr), (init, vi), (fit, vf) = proj(i, cam_id)
            m = vr & vi & vf                      # same KP set for both distances
            d["init"].append(np.linalg.norm(init[m] - real[m], axis=1))
            d["fit"].append(np.linalg.norm(fit[m] - real[m], axis=1))
        di, df = np.concatenate(d["init"]), np.concatenate(d["fit"])
        d_all["init"].append(di); d_all["fit"].append(df)
        rows.append([cam_id, len(di), *_q(di), *_q(df)])
    di, df = np.concatenate(d_all["init"]), np.concatenate(d_all["fit"])
    rows.append(["all", len(di), *_q(di), *_q(df)])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera_id", "n_points", "init_median_px", "init_p90_px",
                    "fit_median_px", "fit_p90_px"])
        w.writerows(rows)
    return rows


def _q(d):
    if len(d) == 0:
        return [np.nan, np.nan]
    return [round(float(np.median(d)), 2), round(float(np.percentile(d, 90)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="cj357_20221009_090000")
    ap.add_argument("--npz-dir", default=str(ROOT / "outputs/reproj_eval/aligned_npz"))
    ap.add_argument("--vid-dir", default=str(ROOT / "vid"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval/real_vs_blender"))
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="kp3d frame indices to draw (default: --n-frames evenly spaced)")
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--n-stat-frames", type=int, default=500)
    ap.add_argument("--grid-order", nargs=8, default=GRID_ORDER,
                    help="camera IDs of the 2x4 grid, row-major (top row left->right, then bottom row)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-stats", action="store_true", help="do not recompute {session}_px_diff.csv")
    ap.add_argument("--clean-cams", nargs="*", default=[],
                    help="also save the grid crop of these cameras without any overlay")
    ap.add_argument("--extra-image", nargs="*", default=[], metavar="CAM=PATH",
                    help="crop an external full-size image (e.g. a Blender render of the same frame "
                         "and camera, 2048x1536) with the same window as the grid cell of CAM; "
                         "requires exactly one --frames value")
    ap.add_argument("--extra-tag", default="blender",
                    help="filename tag of the --extra-image crops (*_{tag}.png, *_{tag}_init.png, *_{tag}_fit.png)")
    args = ap.parse_args()
    extra = dict(e.split("=", 1) for e in args.extra_image)
    if extra and (not args.frames or len(args.frames) != 1):
        ap.error("--extra-image requires exactly one --frames value")

    cfg_y = yaml.safe_load(open(CFG_DIR / "config.yaml"))
    W, H = cfg_y["img_size"]
    z = np.load(Path(args.npz_dir) / f"{args.session}.npz")
    cam_ids = [str(c) for c in z["camera_id"]]
    frame_num, frame_min = z["frame_num"], z["frame_min"]
    n_frame = z["kp3d_mm"].shape[0]
    out = Path(args.output_dir) / args.session
    out.mkdir(parents=True, exist_ok=True)

    proj = Projector(z, cam_ids, W, H, args.score_thr)

    if not args.skip_stats:
        rows = write_stats(proj, cam_ids, n_frame, args.n_stat_frames, args.seed,
                           out / f"{args.session}_px_diff.csv")
        print("camera_id  n  init_med  init_p90  fit_med  fit_p90")
        for r in rows:
            print("  ".join(str(x) for x in r))

    if args.frames:
        sel = np.asarray(args.frames, int)
    else:
        good = np.where((np.isfinite(proj.mm[:, :, 0]) & (proj.score > args.score_thr)).sum(1) >= 15)[0]
        sel = good[np.linspace(0, len(good) - 1, args.n_frames + 2).astype(int)[1:-1]]

    stores = {c: imgstore.new_for_filename(str(Path(args.vid_dir) / f"{args.session}.{c}/metadata.yaml"))
              for c in cam_ids}
    legend = [("real camera (omnidir)", COL_REAL), ("Blender cam, initial", COL_INIT),
              ("Blender cam, affine fit", COL_FIT)]

    for i in sel:
        cells, err = {}, {"init": [], "fit": []}
        for ci, cam_id in enumerate(cam_ids):
            fnum = int(frame_num[ci, i])
            if fnum < 0:
                print(f"  skip frame {i} cam {cam_id}: no frame_num"); continue
            img, _ = stores[cam_id].get_image(frame_number=fnum)
            # imgstore (cv2.VideoCapture) already returns BGR: do not swap channels
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else np.ascontiguousarray(img)

            (real, vr), (init, vi), (fit, vf) = proj(i, cam_id)
            box = crop_box([real[vr], init[vi], fit[vf]], W, H)
            k = vr & vi & vf
            err["init"].append(np.linalg.norm(init[k] - real[k], axis=1))
            err["fit"].append(np.linalg.norm(fit[k] - real[k], axis=1))
            stem = f"{args.session}_frame{int(i):06d}_cam{cam_id}"
            if cam_id in args.clean_cams:
                cv2.imwrite(str(out / f"{stem}_clean.png"), crop_resize(img, box))
                ov = img.copy(); draw_skeleton(ov, real, vr, COL_REAL)
                cv2.imwrite(str(out / f"{stem}_clean_real.png"), crop_resize(ov, box))
                print(f"  clean crop {cam_id}: x0={box[0]} y0={box[1]} side={box[2]}")
            if cam_id in extra:
                ext = cv2.imread(extra[cam_id], cv2.IMREAD_COLOR)
                assert ext is not None and ext.shape[:2] == (H, W), \
                    f"{extra[cam_id]}: expected {W}x{H}, got {None if ext is None else ext.shape}"
                cv2.imwrite(str(out / f"{stem}_{args.extra_tag}.png"), crop_resize(ext, box))
                for tag, (pts, val, col) in {"init": (init, vi, COL_INIT), "fit": (fit, vf, COL_FIT)}.items():
                    ov = ext.copy(); draw_skeleton(ov, pts, val, col)
                    cv2.imwrite(str(out / f"{stem}_{args.extra_tag}_{tag}.png"), crop_resize(ov, box))
            draw_skeleton(img, init, vi, COL_INIT)
            draw_skeleton(img, fit, vf, COL_FIT)
            draw_skeleton(img, real, vr, COL_REAL)

            full = img.copy()
            put_label(full, [(f"{args.session}  frame {i}  cam {cam_id}", (255, 255, 255))] + legend)
            cv2.imwrite(str(out / f"{args.session}_frame{int(i):06d}_cam{cam_id}.jpg"), full,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            cell = crop_resize(img, box)
            put_label(cell, [(f"cam {cam_id}", (255, 255, 255))], scale=0.8)
            cells[cam_id] = cell

        if len(cells) == len(cam_ids):
            order = [cells[c] for c in args.grid_order]
            grid = np.vstack([np.hstack(order[:4]), np.hstack(order[4:8])])
            # mean distance to the real-camera projection over all cameras x KPs of this frame
            ei, ef = (float(np.concatenate(err[t]).mean()) for t in ("init", "fit"))
            bar = legend_bar(grid.shape[1], [
                ("Real camera", COL_REAL),
                (f"Blender camera, initial (error {ei:.1f} px)", COL_INIT),
                (f"Blender camera, affine fit (error {ef:.1f} px)", COL_FIT)])
            print(f"  frame {i}: mean error initial {ei:.1f} px, affine fit {ef:.1f} px")
            cv2.imwrite(str(out / f"{args.session}_frame{int(i):06d}_grid.png"), np.vstack([bar, grid]))
        print(f"frame {i}: wrote {len(cells)} camera overlays")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
