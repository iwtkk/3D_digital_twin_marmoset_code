"""
per_session と per_frame の水平アライメントを実カメラ動画に重ねた比較動画。

各フレームに重畳:
  赤      = 実カメラ omnidir 再投影 ( P_3d )           ← 合わせたい目標
  マゼンタ = per_session 補正 ( 1組の s,tx,ty を全フレーム共通 )
  黄       = per_frame 補正  ( フレームごとに s,tx,ty を最適化 )

狙い: 背後の3Dが一貫していれば per_session と per_frame は一致するはず。
per_frame だけがフレームごとにガタつくなら、それが過適合（座標変換にノイズを吸わせている）。

実行:
  python3 scripts/05_create_digital_twin/alignment_3D_keypoint/plot/make_align_video.py \
      --session cj357_20221009_090000 --cameras 23506237 --n 500
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
    ap.add_argument("--session", default="cj357_20221009_090000")
    ap.add_argument("--cameras", nargs="+", default=["23506237"])
    ap.add_argument("--start", type=int, default=-1, help="-1 で最大動き区間を自動選択")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--score-thr", type=float, default=0.4)
    ap.add_argument("--align-log", default=str(ROOT / "outputs/reproj_eval/align_log.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval/align_video"))
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    cfg_y = yaml.safe_load(open(CFG))
    cam_ids = [str(c) for c in cfg_y["camera_id"]]
    W, H = cfg_y["img_size"]
    bcams = load_cameras()
    proj_omni = align._omnidir_projectors(cam_ids)

    pp = fz.resolve_pickle(Path(ROOT / "outputs/3d") / args.session)
    data = pickle.load(open(pp, "rb"))
    raw = data["kp3d"][0]; score = data["kp3d_score"][0]
    bcoords = align.blender_coords(pp)
    n_frame = min(raw.shape[0], bcoords.shape[0])

    # per_session params（無ければその場で fit）
    ps_params = align.load_align_log(args.align_log).get(args.session)
    if ps_params is None:
        items = align.build_targets(args.session, cam_ids, bcams, proj_omni, W, H)
        ps_params, *_ = align.fit(items, cam_ids, bcams)
    ps_params = np.asarray(ps_params)

    fmap, nT = build_frame_map(ROOT / "vid", args.session, cam_ids)
    n_frame = min(n_frame, nT)

    # クリップ選択（最大動き区間）
    if args.start < 0:
        ctr = np.nanmean(raw[:n_frame], axis=1)
        disp = np.r_[0, np.linalg.norm(np.diff(ctr, axis=0), axis=1)]
        win = np.convolve(np.nan_to_num(disp), np.ones(args.n), "valid")
        start = int(np.argmax(win))
    else:
        start = args.start
    frames = list(range(start, min(start + args.n, n_frame)))
    print(f"clip start={start} n={len(frames)}, per_session={ps_params}")

    def build_item(i):
        rd, vd = {}, {}
        for c in cam_ids:
            r = proj_omni(c, raw[i]); _, bv = bcams[c].project(bcoords[i])
            v = (bv & (score[i] > args.score_thr) & np.isfinite(raw[i, :, 0]) & np.isfinite(r[:, 0])
                 & (r[:, 0] >= 0) & (r[:, 0] < W) & (r[:, 1] >= 0) & (r[:, 1] < H))
            rd[c] = r; vd[c] = v
        return (bcoords[i], rd, vd)

    # per_frame fit（全カメラで拘束）
    pf_params = {}
    from scipy.optimize import least_squares
    for i in frames:
        it = [build_item(i)]
        sol = least_squares(align._residual, ps_params.copy(),
                            args=(it, cam_ids, bcams), method="lm", max_nfev=200)
        pf_params[i] = sol.x
    pfa = np.array([pf_params[i] for i in frames])
    print(f"per_frame param spread: s std={pfa[:,0].std():.4f} "
          f"tx std={pfa[:,1].std()*1000:.1f}mm ty std={pfa[:,2].std()*1000:.1f}mm")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    OW, OH = W // 2, H // 2
    for cam in args.cameras:
        fnum_arr, fmin, store = fmap[cam]
        vp = out / f"{args.session}_cam{cam}_{start}_{len(frames)}.mp4"
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (OW, OH))
        for i in frames:
            try:
                img, _ = store.get_image(frame_number=int(fnum_arr[i]))
            except Exception:
                continue
            img = np.ascontiguousarray(img) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # imgstore already returns BGR
            sc = score[i]
            base_valid = (sc > args.score_thr) & np.isfinite(raw[i, :, 0])
            red = proj_omni(cam, raw[i])
            ps2, psv = bcams[cam].project(align.apply_horizontal(bcoords[i], ps_params))
            pf2, pfv = bcams[cam].project(align.apply_horizontal(bcoords[i], pf_params[i]))
            draw(img, red, base_valid & np.isfinite(red[:, 0]), (0, 0, 255))
            draw(img, ps2, base_valid & psv, (255, 0, 255))
            draw(img, pf2, base_valid & pfv, (0, 255, 255))
            cv2.putText(img, f"{args.session} cam{cam} f{i}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            for k, (t, col) in enumerate([("red: real omnidir", (0, 0, 255)),
                                          ("magenta: per_session", (255, 0, 255)),
                                          ("yellow: per_frame", (0, 255, 255))]):
                cv2.putText(img, t, (20, 80 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
            vw.write(cv2.resize(img, (OW, OH)))
        vw.release()
        print(f"wrote {vp}")


if __name__ == "__main__":
    main()
