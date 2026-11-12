"""
A方式の水平アライメント最適化。

cyan = Blenderカメラ投影( T(P_3d) に水平補正を加えたもの ) を
red  = 実カメラ omnidir 投影( P_3d ) に近づける per-session 変換 (s, tx, ty) を求める。

水平補正（Blender世界 m, bZ=高さ は触らない）:
    bX' = s * bX + tx
    bY' = s * bY + ty
target は red（omnidir再投影）。同じ3Dから両側を作るので3D復元ノイズが入らず、
T＋レンズ差だけを最小化できる。ロバスト損失(soft_l1/Huber)で検出・三角測量の外れ値を抑える。

粒度（検証で per-session を推奨。global/animal/frame は比較用）:
  per-frame  : 過適合（パラメータがほぼ動かず、レンズ則差を各フレームで吸収するだけ）
  per-animal : セッション間ドリフト>個体差 のため global と大差なく時に悪化
  global     : ドリフトしたセッションを取りこぼす
  per-session: 過適合せず最良。既存の垂直補正(floorZ)と同粒度

CLI:
  python3 scripts/05_create_digital_twin/alignment_3D_keypoint/run/align.py --session cj357_20221009_090000
  python3 scripts/05_create_digital_twin/alignment_3D_keypoint/run/align.py --all      # outputs/3d 全 session
出力: <output-dir>/align_log.csv （session,s,tx,ty,before_px,after_px,n_frame,n_obs）
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))
import _config as cfg                       # noqa: E402
from blender_fisheye import load_cameras     # noqa: E402
import floorz_core as fz                     # noqa: E402

CFG = ROOT / "config/marmo_cj425m/config.yaml"
INTRIN = ROOT / "config/marmo_cj425m/cam_intrinsic.h5"
EXTRIN = ROOT / "config/marmo_cj425m/cam_extrinsic_optim.h5"
KEYS = cfg.KP_KEYS[:20]


def _omnidir_projectors(cam_ids):
    fi = h5py.File(INTRIN, "r"); fe = h5py.File(EXTRIN, "r")
    pr = {}
    for c in cam_ids:
        K = fi[f"/{c}/K"][()]; xi = fi[f"/{c}/xi"][()][0, 0]; D = fi[f"/{c}/D"][()]
        rv = fe[f"/{c}/rvec"][()]; tv = fe[f"/{c}/tvec"][()]
        pr[c] = (K, xi, D, rv, tv)
    fi.close(); fe.close()

    def project(c, P):
        K, xi, D, rv, tv = pr[c]
        pts, _ = cv2.omnidir.projectPoints(np.reshape(P, (-1, 1, 3)).astype(np.float64), rv, tv, K, xi, D)
        return pts[:, 0, :]
    return project


def blender_coords(pickle_path):
    df = fz.load_pickle_to_df(pickle_path)
    dfc, *_ = fz.scale_and_correct(df)
    return np.stack([dfc[[f"pts_{k}_x", f"pts_{k}_y", f"pts_{k}_z"]].to_numpy(float)
                     for k in KEYS], axis=1)


def build_targets(session, cam_ids, bcams, proj_omni, W, H,
                  n_frames=400, score_thr=0.4, input_dir=None, seed=1):
    """session の (bcoords_frame, red_dict, valid_dict) のリストを返す（画像不要）。"""
    input_dir = Path(input_dir or (ROOT / "outputs/3d"))
    pp = fz.resolve_pickle(input_dir / session)
    data = pickle.load(open(pp, "rb"))
    raw = data["kp3d"][0]; score = data["kp3d_score"][0]
    bc = blender_coords(pp)
    n = min(raw.shape[0], bc.shape[0])
    good = np.where((np.isfinite(raw[:n, :, 0]) & (score[:n] > score_thr)).sum(1) >= 16)[0]
    if len(good) == 0:
        return []
    fr = np.random.default_rng(seed).choice(good, min(n_frames, len(good)), replace=False)
    items = []
    for i in fr:
        rd, vd = {}, {}
        for c in cam_ids:
            r = proj_omni(c, raw[i]); _, bv = bcams[c].project(bc[i])
            v = (bv & (score[i] > score_thr) & np.isfinite(raw[i, :, 0]) & np.isfinite(r[:, 0])
                 & (r[:, 0] >= 0) & (r[:, 0] < W) & (r[:, 1] >= 0) & (r[:, 1] < H))
            rd[c] = r; vd[c] = v
        items.append((bc[i], rd, vd))
    return items


def _residual(params, items, cam_ids, bcams):
    s, tx, ty = params
    out = []
    for bc, rd, vd in items:
        P = bc.copy(); P[:, 0] = s * P[:, 0] + tx; P[:, 1] = s * P[:, 1] + ty
        for c in cam_ids:
            v = vd[c]
            if v.sum() == 0:
                continue
            b, _ = bcams[c].project(P[v], clip_fov=False)
            out.append((b - rd[c][v]).ravel())
    return np.concatenate(out) if out else np.zeros(1)


def median_px(params, items, cam_ids, bcams):
    r = _residual(params, items, cam_ids, bcams)
    return float(np.median(np.linalg.norm(r.reshape(-1, 2), axis=1)))


def fit(items, cam_ids, bcams, robust=True):
    """(params, before_px, after_px, n_obs) を返す。robust=True で soft_l1(Huber系)。"""
    before = median_px([1.0, 0.0, 0.0], items, cam_ids, bcams)
    kw = dict(method="trf", loss="soft_l1", f_scale=8.0) if robust else dict(method="lm")
    sol = least_squares(_residual, [1.0, 0.0, 0.0],
                        args=(items, cam_ids, bcams), max_nfev=400, **kw)
    after = median_px(sol.x, items, cam_ids, bcams)
    return sol.x, before, after, sol.fun.size // 2


def apply_horizontal(bcoords, params):
    """floorz Blender座標 (..., 3) に水平補正 (s,tx,ty) を適用（bZ は不変）。"""
    s, tx, ty = params
    out = np.asarray(bcoords, float).copy()
    out[..., 0] = s * out[..., 0] + tx
    out[..., 1] = s * out[..., 1] + ty
    return out


def load_align_log(path):
    """align_log.csv → {session: (s,tx,ty)}。"""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(open(path)):
        out[row["session"]] = (float(row["s"]), float(row["tx"]), float(row["ty"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--all", action="store_true", help="outputs/3d の全 session")
    ap.add_argument("--input-dir", default=str(ROOT / "outputs/3d"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval"))
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--score-thr", type=float, default=0.4)
    ap.add_argument("--no-robust", action="store_true")
    args = ap.parse_args()

    cfg_y = yaml.safe_load(open(CFG))
    cam_ids = [str(c) for c in cfg_y["camera_id"]]
    W, H = cfg_y["img_size"]
    bcams = load_cameras()
    proj_omni = _omnidir_projectors(cam_ids)

    in_dir = Path(args.input_dir)
    if args.all:
        sessions = sorted(d.name for d in in_dir.iterdir()
                          if d.is_dir() and fz.resolve_pickle(d) is not None)
    elif args.session:
        sessions = [args.session]
    else:
        ap.error("--session または --all を指定してください")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "align_log.csv"
    rows = []
    for S in sessions:
        items = build_targets(S, cam_ids, bcams, proj_omni, W, H,
                              args.n_frames, args.score_thr, in_dir)
        if not items:
            print(f"{S}: 追跡フレーム不足、スキップ"); continue
        p, before, after, nobs = fit(items, cam_ids, bcams, robust=not args.no_robust)
        rows.append(dict(session=S, s=f"{p[0]:.5f}", tx=f"{p[1]:.5f}", ty=f"{p[2]:.5f}",
                         before_px=f"{before:.2f}", after_px=f"{after:.2f}",
                         n_frame=len(items), n_obs=nobs))
        print(f"{S}: s={p[0]:.3f} tx={p[1]:+.3f} ty={p[2]:+.3f}  {before:.1f} -> {after:.1f} px")

    if rows:
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {log_path} ({len(rows)} sessions)")


if __name__ == "__main__":
    main()
