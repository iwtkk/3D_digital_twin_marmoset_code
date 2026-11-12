"""
per_session affine (affine_log.npz) を実カメラ動画に重ねた検証動画。

  赤 = 実カメラ omnidir 再投影 ( raw kp3d )
  黄 = per_session affine 投影 ( A·(raw@Rᵀ/1000)+t を Blender equisolid 投影 )

実行:
  python3 scripts/05_create_digital_twin/alignment_3D_keypoint/plot/make_affine_video.py \
      --session cj611_20230226_110000 --cameras 23506237        # 既定で全フレーム
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))
import _config as cfg                                  # noqa: E402
from blender_fisheye import load_cameras                # noqa: E402
from frame_map import build_frame_map                   # noqa: E402
import align                                            # noqa: E402
import affine_align as aa                               # noqa: E402
import floorz_core as fz                                # noqa: E402

BONES = cfg.BONES
CFG = ROOT / "config/marmo_cj425m/config.yaml"


def draw(img, pts, valid, color, r=4, th=2):
    for a, b in BONES:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(np.round(pts[a]).astype(int)),
                     tuple(np.round(pts[b]).astype(int)), color, th, cv2.LINE_AA)
    for j in range(len(pts)):
        if valid[j]:
            cv2.circle(img, tuple(np.round(pts[j]).astype(int)), r, color, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--cameras", nargs="+", default=["23506237"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=-1, help="-1 で全フレーム")
    ap.add_argument("--score-thr", type=float, default=0.4)
    ap.add_argument("--affine-log", default=str(ROOT / "outputs/reproj_eval/affine_log.npz"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval/affine_video"))
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--scale", type=float, default=0.5, help="出力解像度の倍率")
    args = ap.parse_args()

    cfg_y = yaml.safe_load(open(CFG))
    cam_ids = [str(c) for c in cfg_y["camera_id"]]
    W, H = cfg_y["img_size"]
    bcams = load_cameras()
    proj_omni = align._omnidir_projectors(cam_ids)
    R = fz.KP3D_ROT

    params = dict(np.load(args.affine_log, allow_pickle=True)).get(args.session)
    if params is None:
        ap.error(f"{args.session} が affine_log に無い")
    A = params[:9].reshape(3, 3); t = params[9:]

    pp = fz.resolve_pickle(Path(ROOT / "outputs/3d") / args.session)
    data = pickle.load(open(pp, "rb"))
    raw = data["kp3d"][0]; score = data["kp3d_score"][0]
    fmap, nT = build_frame_map(ROOT / "vid", args.session, cam_ids)
    n_frame = min(raw.shape[0], nT)

    end = n_frame if args.n < 0 else min(args.start + args.n, n_frame)
    frames = range(args.start, end)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    OW, OH = int(W * args.scale), int(H * args.scale)

    for cam in args.cameras:
        fnum_arr, fmin, store = fmap[cam]
        vp = out / f"{args.session}_cam{cam}_affine.mp4"
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (OW, OH))
        nw = 0
        for i in frames:
            try:
                img, _ = store.get_image(frame_number=int(fnum_arr[i]))
            except Exception:
                continue
            img = np.ascontiguousarray(img) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # imgstore already returns BGR
            base = (score[i] > args.score_thr) & np.isfinite(raw[i, :, 0])
            red = proj_omni(cam, raw[i])
            Pbl = aa.apply_affine((raw[i] @ R.T) / 1000.0, A, t)
            aff, av = bcams[cam].project(Pbl)
            draw(img, red, base & np.isfinite(red[:, 0]), (0, 0, 255))      # 赤=実omnidir
            draw(img, aff, base & av, (0, 255, 255))                        # 黄=affine
            cv2.putText(img, f"{args.session} cam{cam} f{i}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            for k, (txt, col) in enumerate([("red: real omnidir", (0, 0, 255)),
                                            ("yellow: per_session affine", (0, 255, 255))]):
                cv2.putText(img, txt, (20, 80 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
            vw.write(cv2.resize(img, (OW, OH)))
            nw += 1
        vw.release()
        print(f"wrote {vp} ({nw} frames)")


if __name__ == "__main__":
    main()
