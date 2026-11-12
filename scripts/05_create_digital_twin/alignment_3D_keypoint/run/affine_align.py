"""
回転だけ残し、その上に 3Dアフィン A·X+t を再投影誤差最小化で per_session 直接当てはめる。

floorz の global scale + 垂直affine を使わず、回転済み計測座標 (m) から Blender座標へ
1つのアフィンで写す:
    X_m = (raw @ KP3D_ROT.T) / 1000          # 回転後・メートル
    P_blender = A @ X_m + t                    # A:3x3, t:3
目的: Σ_frame Σ_cam Σ_kp ρ( || B_c(P_blender) − O_c(raw) || )   (soft_l1, per_session)

検証: full affine(12) は train=test=6.78px（cj357）で過適合せず、similarity(7)とほぼ同じ。
floorz の固定回転が ~3.75° ずれていたのが主因で、これを直接補正すると omnidir/equisolid
のレンズ床(~6.8px)まで到達する。

CLI:
  python3 scripts/05_create_digital_twin/alignment_3D_keypoint/run/affine_align.py --session cj357_20221009_090000 --export
出力: <out>/affine_log.npz (session->A,t), --export で aligned_csv/<session>_affine.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))
import _config as cfg                       # noqa: E402
from blender_fisheye import load_cameras     # noqa: E402
import align as _al                          # noqa: E402
import floorz_core as fz                     # noqa: E402

CFG = ROOT / "config/marmo_cj425m/config.yaml"
R = fz.KP3D_ROT


def _floorz_init():
    hs = np.array(fz.BLENDER_BBOX_SCALE) / 2
    NS = np.array(fz.KP3D_NORM_SCALE); C = np.array(fz.KP3D_CENTER); BC = np.array(fz.BLENDER_CENTER)
    A0 = np.diag(1000.0 * hs / NS); t0 = BC - C * hs / NS
    return A0, t0


def build_targets(session, cam_ids, bcams, proj_omni, W, H,
                  n_frames=300, score_thr=0.4, input_dir=None, seed=1):
    """(X_m_rotated_meters, red_dict, valid_dict) を返す。"""
    pp = fz.resolve_pickle(Path(input_dir or (ROOT / "outputs/3d")) / session)
    data = fz.load_kp3d(pp); raw = data["kp3d"][0]; score = data["kp3d_score"][0]
    n = raw.shape[0]
    good = np.where((np.isfinite(raw[:n, :, 0]) & (score[:n] > score_thr)).sum(1) >= 16)[0]
    if len(good) == 0:
        return []
    fr = np.random.default_rng(seed).choice(good, min(n_frames, len(good)), replace=False)
    items = []
    for i in fr:
        Xm = (raw[i] @ R.T) / 1000.0
        rd, vd = {}, {}
        for c in cam_ids:
            r = proj_omni(c, raw[i])
            v = (np.isfinite(raw[i, :, 0]) & (score[i] > score_thr) & np.isfinite(r[:, 0])
                 & (r[:, 0] >= 0) & (r[:, 0] < W) & (r[:, 1] >= 0) & (r[:, 1] < H))
            rd[c] = r; vd[c] = v
        items.append((Xm, rd, vd))
    return items


def _residual(p, items, cam_ids, bcams):
    A = p[:9].reshape(3, 3); t = p[9:]
    out = []
    for Xm, rd, vd in items:
        P = (A @ Xm.T).T + t
        for c in cam_ids:
            v = vd[c]
            if v.sum() == 0:
                continue
            b, _ = bcams[c].project(P[v], clip_fov=False)
            out.append((b - rd[c][v]).ravel())
    return np.concatenate(out) if out else np.zeros(1)


def fit(items, cam_ids, bcams):
    A0, t0 = _floorz_init()
    p0 = np.concatenate([A0.ravel(), t0])
    before = float(np.median(np.linalg.norm(_residual(p0, items, cam_ids, bcams).reshape(-1, 2), axis=1)))
    sol = least_squares(_residual, p0, args=(items, cam_ids, bcams),
                        method="trf", loss="soft_l1", f_scale=8.0, max_nfev=800)
    after = float(np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)))
    return sol.x[:9].reshape(3, 3), sol.x[9:], before, after


def apply_affine(rotated_m, A, t):
    """回転後メートル座標 (...,3) に A·X+t。"""
    X = np.asarray(rotated_m, float)
    return (A @ X.reshape(-1, 3).T).T.reshape(X.shape) + t


def build_aligned_df(session, A, t, input_dir=None):
    """DataFrame of affine-aligned Blender coordinates (shared by CSV and NPZ export)."""
    pp = fz.resolve_pickle(Path(input_dir or (ROOT / "outputs/3d")) / session)
    df = fz.load_pickle_to_df(pp)        # 回転済み mm（20+派生3点）
    coord = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    keys = sorted({c.rsplit("_", 1)[0] for c in coord})
    for k in keys:
        Xm = df[[f"{k}_x", f"{k}_y", f"{k}_z"]].to_numpy(float) / 1000.0
        P = apply_affine(Xm, A, t)
        df[f"{k}_x"], df[f"{k}_y"], df[f"{k}_z"] = P[:, 0], P[:, 1], P[:, 2]
    return fz.add_cam_pose_with_euler(df)


def export_csv(session, A, t, out_dir, input_dir=None):
    """floorz と同形式の Blender座標 CSV を全フレーム書き出す（affine適用）。"""
    df = build_aligned_df(session, A, t, input_dir)
    out = Path(out_dir) / f"{session}_affine.csv"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out, len(df)


# ── 並列ワーカー（プロセスごとにカメラ等を1度だけ構築） ───────────────────────────
_W: dict = {}


def _init_worker(input_dir, n_frames, export, out_dir):
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg_y = yaml.safe_load(open(CFG))
    _W["cam_ids"] = [str(c) for c in cfg_y["camera_id"]]
    _W["W"], _W["H"] = cfg_y["img_size"]
    _W["bcams"] = load_cameras()
    _W["proj_omni"] = _al._omnidir_projectors(_W["cam_ids"])
    _W["input_dir"] = Path(input_dir); _W["n_frames"] = n_frames
    _W["export"] = export; _W["out_dir"] = Path(out_dir)


def _process(session):
    """1 session を fit。(session, params|None, msg) を返す。"""
    try:
        items = build_targets(session, _W["cam_ids"], _W["bcams"], _W["proj_omni"],
                              _W["W"], _W["H"], _W["n_frames"], input_dir=_W["input_dir"])
        if not items:
            return session, None, f"{session}: 追跡不足 skip"
        A, t, before, after = fit(items, _W["cam_ids"], _W["bcams"])
        if _W["export"]:
            export_csv(session, A, t, _W["out_dir"] / "aligned_csv", _W["input_dir"])
        msg = (f"{session}: {before:.1f}->{after:.1f}px "
               f"scale={np.linalg.norm(A, axis=0).round(3).tolist()}")
        return session, np.concatenate([A.ravel(), t]), msg
    except Exception as e:
        return session, None, f"{session}: ERROR {e}"


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--all", action="store_true", help="outputs/3d の全 session を処理")
    ap.add_argument("--input-dir", default=str(ROOT / "outputs/3d"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval"))
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--resume", action="store_true", help="affine_log.npz に既にある session はスキップ")
    ap.add_argument("--jobs", type=int, default=1,
                    help="並列プロセス数。ワークステーションは1、GPUサーバ等は 8〜 を推奨")
    ap.add_argument("--verbose", action="store_true",
                    help="各 session の詳細を表示（既定はプログレスバーのみ）")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if args.all:
        sessions = sorted(d.name for d in in_dir.iterdir()
                          if d.is_dir() and fz.resolve_pickle(d) is not None)
    elif args.session:
        sessions = [args.session]
    else:
        ap.error("--session または --all を指定してください")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "affine_log.npz"
    store = dict(np.load(npz)) if npz.exists() else {}   # 値は float 配列のみ
    if args.resume:
        sessions = [s for s in sessions if s not in store]
    if not sessions:
        print("処理対象なし（--resume で全て済）"); return

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    log = open(out_dir / "affine_fit.log", "a")
    init_args = (args.input_dir, args.n_frames, args.export, str(out_dir))

    def handle(result):
        session, params, msg = result
        log.write(msg + "\n"); log.flush()
        if params is not None:
            store[session] = params
            np.savez(npz, **store)        # 親プロセスのみが書く（競合なし）・逐次保存
        if args.verbose:
            (tqdm.write if tqdm else print)(msg)

    jobs = max(1, args.jobs)
    if jobs == 1:
        _init_worker(*init_args)
        it = (tqdm(sessions, desc="affine fit") if (tqdm and not args.verbose) else sessions)
        for s in it:
            handle(_process(s))
    else:
        with mp.Pool(jobs, initializer=_init_worker, initargs=init_args) as pool:
            gen = pool.imap_unordered(_process, sessions)
            it = (tqdm(gen, total=len(sessions), desc=f"affine fit x{jobs}")
                  if (tqdm and not args.verbose) else gen)
            for result in it:
                handle(result)
    log.close()
    print(f"done. {len(store)} sessions in {npz}")


if __name__ == "__main__":
    main()
