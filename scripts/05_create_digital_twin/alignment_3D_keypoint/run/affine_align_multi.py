"""
Tool: affine_align_multi

Purpose:
    affine_align.py（再投影誤差最小化による 3Dアフィン A·X+t の per-session フィット）を
    複数個体に拡張する。個体ごとに kp3d[i_animal] を使って A,t をフィットし、
    それぞれ Blender 取込用 CSV（他 transform と同形式）に書き出す。
    フィット本体（_residual/fit）と初期値（_floorz_init）は affine_align.py をそのまま再利用し、
    個体選択（build_targets / load_df / export）だけを i_animal 対応にした薄いラッパー。

Inputs:
    kp3d_pickle:
        type: file
        path: {input_dir}/{session}/kp3d_fxdJointLen.pickle (無ければ kp3d.pickle)
        content: dict{kp3d:[n_animal,n_frame,n_kp,3], kp3d_score:[n_animal,n_frame,n_kp]}
    camera_calibration:
        type: files (固定 rig, 個体・セッション非依存)
        path: config/marmo_cj425m/{config.yaml, cam_intrinsic.h5, cam_extrinsic_optim.h5}
              + blender_fisheye.load_cameras() の Blender カメラ

Outputs:
    aligned_csv:
        type: file(s)
        path: {output_dir}/{session}_animal{i}_affine.csv  (有効個体ごと)
        content: A·X+t 適用済みの Blender 座標 CSV（pts_{kp}_{x,y,z} + 頭部 Euler 列）
    affine_log_npz:
        type: file
        path: {output_dir}/affine_log_multi.npz  (key "{session}#a{i}" -> [A.ravel(9), t(3)])

Registry:
    .agent_docs/docs/tool_registry.yml#affine_align_multi

使い方:
    .venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/run/affine_align_multi.py \
        --session foodcomp_cj753_cj813_20240111_132306 \
        --input-dir ../crop_pe/outputs/multiview_pose3d_hybrid \
        --output-dir outputs/animate_temp_multi/reproj_csv --export
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))

import affine_align as aa          # noqa: E402  再利用: fit / _residual / _floorz_init / apply_affine
import align as _al                # noqa: E402  _omnidir_projectors
import floorz_core as fz           # noqa: E402  回転・KP 名・頭部姿勢
from blender_fisheye import load_cameras  # noqa: E402

CFG = ROOT / "config/marmo_cj425m/config.yaml"
R = fz.KP3D_ROT


# ── 個体選択ユーティリティ ─────────────────────────────────────────────────────

def valid_animals(pp: Path) -> list[int]:
    """全 NaN / 全ゼロの空スロットを除いた有効個体インデックス一覧。"""
    data = pickle.load(open(pp, "rb"))
    X = data["kp3d"]
    out = []
    for i in range(X.shape[0]):
        finite = np.isfinite(X[i]).all(axis=-1)
        nonzero = np.abs(X[i]).sum(axis=-1) > 1e-9
        if float((finite & nonzero).mean()) > 0.0:
            out.append(i)
    return out


def build_targets_animal(session, i_animal, cam_ids, bcams, proj_omni, W, H,
                         n_frames=300, score_thr=0.4, input_dir=None, seed=1):
    """affine_align.build_targets の i_animal 版。

    ターゲット 2D は kp3d[i_animal] の omnidir 再投影。個体ごとに自己完結する。
    """
    pp = fz.resolve_pickle(Path(input_dir or (ROOT / "outputs/3d")) / session)
    data = pickle.load(open(pp, "rb"))
    raw = data["kp3d"][i_animal]
    score = data["kp3d_score"][i_animal]
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
            rd[c] = r
            vd[c] = v
        items.append((Xm, rd, vd))
    return items


def load_df_animal(pp: Path, i_animal: int):
    """floorz_core.load_pickle_to_df の i_animal 版（回転済み mm, 20KP + 派生3点）。"""
    import pandas as pd

    data = pickle.load(open(pp, "rb"))
    X = data["kp3d"]
    S = data["kp3d_score"]
    n_frame = X.shape[1]
    Xa = X[i_animal, :n_frame].copy()
    Sa = S[i_animal, :n_frame].copy()
    kp_names = fz.KP_NAMES[:]
    for (i1, i2, name) in [(5, 6, "shoulder_mid"), (11, 12, "base"), (1, 2, "center_eye")]:
        if max(i1, i2) < Xa.shape[1]:
            mid = (Xa[:, i1] + Xa[:, i2]) / 2
            Xa = np.concatenate([Xa, mid[:, None, :]], axis=1)
            Sa = np.concatenate([Sa, ((Sa[:, i1] + Sa[:, i2]) / 2)[:, None]], axis=1)
            kp_names.append(name)
    Xa = Xa @ R.T
    kp_keys = fz.unique_keys(kp_names)
    n = Xa.shape[0]
    d = {"i_frame": np.arange(n, dtype=np.int64),
         "i_animal": np.full(n, i_animal, dtype=np.int64)}
    for ki, key in enumerate(kp_keys):
        d[f"pts_{key}_x"] = Xa[:, ki, 0]
        d[f"pts_{key}_y"] = Xa[:, ki, 1]
        d[f"pts_{key}_z"] = Xa[:, ki, 2]
    return pd.DataFrame(d)


def export_csv_animal(session, i_animal, A, t, out_dir, input_dir, out_name):
    """1 個体分の A·X+t を全フレームに適用し Blender 座標 CSV を書き出す。"""
    pp = fz.resolve_pickle(Path(input_dir or (ROOT / "outputs/3d")) / session)
    df = load_df_animal(pp, i_animal)
    coord = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    keys = sorted({c.rsplit("_", 1)[0] for c in coord})
    for k in keys:
        Xm = df[[f"{k}_x", f"{k}_y", f"{k}_z"]].to_numpy(float) / 1000.0
        P = aa.apply_affine(Xm, A, t)
        df[f"{k}_x"], df[f"{k}_y"], df[f"{k}_z"] = P[:, 0], P[:, 1], P[:, 2]
    df = fz.add_cam_pose_with_euler(df)
    out = Path(out_dir) / f"{out_name}.csv"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out, len(df)


def process_session(session, cam_ids, bcams, proj_omni, W, H,
                    input_dir, out_dir, n_frames, export, store):
    pp = fz.resolve_pickle(Path(input_dir) / session)
    if pp is None:
        raise FileNotFoundError(f"pickle not found: {Path(input_dir) / session}")
    animals = valid_animals(pp)
    print(f"\n=== {session} (有効個体={animals}) ===")
    for i_animal in animals:
        items = build_targets_animal(session, i_animal, cam_ids, bcams, proj_omni,
                                     W, H, n_frames, input_dir=input_dir)
        if not items:
            print(f"  animal{i_animal}: 追跡不足 skip")
            continue
        A, t, before, after = aa.fit(items, cam_ids, bcams)
        store[f"{session}#a{i_animal}"] = np.concatenate([A.ravel(), t])
        scale = np.linalg.norm(A, axis=0).round(3).tolist()
        print(f"  animal{i_animal}: reproj {before:.1f}->{after:.1f}px  scale={scale}")
        if export:
            out_name = f"{session}_animal{i_animal}_affine"
            out, n = export_csv_animal(session, i_animal, A, t, out_dir, input_dir, out_name)
            print(f"    [OK] wrote {out}  ({n} frames)")


def main():
    ap = argparse.ArgumentParser(
        description="複数個体の 3Dアフィン再投影アライメント → Blender CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None)
    ap.add_argument("--all", action="store_true", help="--input-dir の全 session を処理")
    ap.add_argument("--input-dir", default=str(ROOT / "outputs/3d"),
                    help="pickle のルート（複数個体なら ../crop_pe/outputs/multiview_pose3d_hybrid 等）")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/animate_temp_multi/reproj_csv"))
    ap.add_argument("--n-frames", type=int, default=300, help="フィットに使うサンプルフレーム数")
    ap.add_argument("--export", action="store_true", help="Blender CSV を書き出す")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if args.all:
        sessions = sorted(d.name for d in in_dir.iterdir()
                          if d.is_dir() and fz.resolve_pickle(d) is not None)
    elif args.session:
        sessions = [args.session]
    else:
        ap.error("--session または --all を指定してください")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # カメラ・投影器は 1 度だけ構築（全 session/個体で共有）
    cfg_y = yaml.safe_load(open(CFG))
    cam_ids = [str(c) for c in cfg_y["camera_id"]]
    W, H = cfg_y["img_size"]
    bcams = load_cameras()
    proj_omni = _al._omnidir_projectors(cam_ids)

    npz = out_dir / "affine_log_multi.npz"
    store = dict(np.load(npz, allow_pickle=True)) if npz.exists() else {}

    skipped = []
    for session in sessions:
        try:
            process_session(session, cam_ids, bcams, proj_omni, W, H,
                            args.input_dir, out_dir, args.n_frames, args.export, store)
        except Exception as e:
            print(f"  [SKIP] {session}: {e}", file=sys.stderr)
            skipped.append(session)

    np.savez(npz, **store)
    print(f"\ndone. affine log: {npz}")
    if skipped:
        print(f"{len(skipped)} セッションをスキップ: {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
