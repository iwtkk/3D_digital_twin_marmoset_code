"""
pickle → KDE affine 補正済み Blender CSV

Phase 1 (validate_affine_correction.py) で検証した
KDE + Hungarian affine 補正を pickle パイプラインに統合したスクリプト。

test_scale_kp3d_to_blender.py との違い:
  旧: p99 ベースの offset のみ (a=1.0 固定)
  新: KDE ピーク検出 + affine 回帰 (a, b 両方補正)

使い方:
  # 1セッション
  OPENBLAS_NUM_THREADS=1 python scripts/dev/affine_pkl_csv/pkl_to_corrected_csv.py \\
      --session cj425_20210427_120000

  # 全セッション
  OPENBLAS_NUM_THREADS=1 python scripts/dev/affine_pkl_csv/pkl_to_corrected_csv.py --all

  # テスト用: 最初の1000フレームのみ
  OPENBLAS_NUM_THREADS=1 python scripts/dev/affine_pkl_csv/pkl_to_corrected_csv.py \\
      --session cj425_20210427_120000 --n-frames 1000

入力:
  outputs/3d/{session}/kp3d_fxdJointLen.pickle  (または kp3d.pickle)

出力:
  outputs/dev/affine_pkl_csv/csv/{session}_points.csv          中間 CSV (rotated mm)
  outputs/dev/affine_pkl_csv/corrected/{session}_corrected.csv 補正済み CSV (座標 × 1/1000)
  outputs/dev/affine_pkl_csv/affine_log.csv                    session ごとの (a, b, status)

座標スケール:
  KDE affine 補正は Blender bZ (m) 空間で計算。
  出力 CSV には OUTPUT_SCALE=1/1000 を全軸に乗じた値を書き出す。
  例) 床 bZ=0.7337m → 出力値 0.0007337
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

# ── 定数 ──────────────────────────────────────────────────────────────────────

KP_NAMES = [
    "nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder",
    "L elbow", "R elbow", "L wrist", "R wrist", "L hip", "R hip",
    "L knee", "R knee", "L ankle", "R ankle", "backbone peak", "tail_1", "tail_2",
]

BLENDER_BBOX_SCALE = np.array([0.86658, 0.88791, 0.7679])
BLENDER_CENTER     = np.array([0.0, -0.11142, 1.11765])

KP3D_CENTER     = np.array([-1131.005, -1072.940,  286.610])
KP3D_NORM_SCALE = np.array([  449.675,   439.560,  378.880])

# 物理高さ (Blender bZ, m)
PHYSICAL_HEIGHTS_M = np.array([0.7337, 1.0179, 1.3035, 1.5016])

# KDE + affine パラメータ (validate_affine_correction.py と同一)
FLOOR_KP_KEYWORDS  = ("wrist", "ankle")
KDE_BW             = 0.03
MIN_PEAK_GAP_M     = 0.10
MAX_MATCH_DIST_M   = 0.10
MAX_AFFINE_SLOPE_DEV = 0.15

OUTPUT_SCALE = 1 / 1   # 出力 CSV の座標スケール（全軸に乗算）。1/1=Blenderメートルそのまま, 1/1000=mm相当

BASE_DIR  = Path("outputs/3d")
OUT_ROOT  = Path("outputs/dev/affine_pkl_csv")


# ── ユーティリティ (test_scale_kp3d_to_blender.py から流用) ──────────────────

def sanitize(name: str) -> str:
    key = name.lower().strip().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", key)


def unique_keys(names):
    seen = {}
    out = []
    for n in names:
        base = sanitize(n)
        k = base
        i = 2
        while k in seen:
            k = f"{base}_{i}"
            i += 1
        seen[k] = True
        out.append(k)
    return out


def rot_x(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

def rot_y(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], float)

def rot_z(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def rot_euler(rx=0.0, ry=0.0, rz=0.0, order="xyz"):
    mats = {"x": rot_x(rx), "y": rot_y(ry), "z": rot_z(rz)}
    R = np.eye(3)
    for ax in order:
        R = mats[ax] @ R
    return R

def _safe_norm(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)

def _rotmat_to_euler_xyz(R):
    r00=R[...,0,0]; r10=R[...,1,0]; r20=R[...,2,0]
    r11=R[...,1,1]; r12=R[...,1,2]
    r21=R[...,2,1]; r22=R[...,2,2]
    sy = np.sqrt(r00*r00 + r10*r10)
    singular = sy < 1e-8
    rx = np.where(~singular, np.arctan2(r21, r22), np.arctan2(-r12, r11))
    ry = np.where(~singular, np.arctan2(-r20, sy),  np.arctan2(-r20, sy))
    rz = np.where(~singular, np.arctan2(r10, r00),  np.zeros_like(r00))
    return np.stack([rx, ry, rz], axis=-1)

def add_cam_pose_with_euler(points, l_eye="pts_l_eye_", r_eye="pts_r_eye_",
                             l_ear="pts_l_ear_", r_ear="pts_r_ear_",
                             out_prefixes=("pts_l_cam_","pts_r_cam_"),
                             center_eye_prefix="pts_center_eye_", forward_sign=-1):
    need = ([f"{l_eye}{a}" for a in "xyz"] + [f"{r_eye}{a}" for a in "xyz"]
            + [f"{l_ear}{a}" for a in "xyz"] + [f"{r_ear}{a}" for a in "xyz"])
    for c in need:
        if c not in points.columns:
            raise ValueError(f"列がありません: {c}")
    LE  = points[[f"{l_eye}x",f"{l_eye}y",f"{l_eye}z"]].to_numpy(float)
    RE  = points[[f"{r_eye}x",f"{r_eye}y",f"{r_eye}z"]].to_numpy(float)
    LEr = points[[f"{l_ear}x",f"{l_ear}y",f"{l_ear}z"]].to_numpy(float)
    REr = points[[f"{r_ear}x",f"{r_ear}y",f"{r_ear}z"]].to_numpy(float)
    eye_c = (LE+RE)*0.5; ear_c = (LEr+REr)*0.5
    s_n = _safe_norm(RE-LE)
    f_n = _safe_norm((eye_c-ear_c)*float(forward_sign))
    u_raw = np.cross(f_n, s_n)
    mask_bad = ((np.linalg.norm(u_raw,axis=1)<1e-8)
                | np.isnan(u_raw).any(axis=1)
                | np.isnan(s_n).any(axis=1)
                | np.isnan(f_n).any(axis=1))
    if np.any(mask_bad):
        up = np.tile(np.array([0.,0.,1.])[None,:], (len(points),1))
        u_raw[mask_bad] = np.cross(f_n[mask_bad], up[mask_bad])
    u_n = _safe_norm(u_raw)
    s_ortho = _safe_norm(np.cross(u_n, f_n))
    R = np.stack([s_ortho, u_n, f_n], axis=-1)
    det = np.linalg.det(R)
    if np.any(det<0): R[det<0,:,0] *= -1.
    eul = _rotmat_to_euler_xyz(R)
    for pfx in out_prefixes:
        points[f"{pfx}rx"]=eul[:,0]; points[f"{pfx}ry"]=eul[:,1]; points[f"{pfx}rz"]=eul[:,2]
    if center_eye_prefix:
        points[f"{center_eye_prefix}rx"]=eul[:,0]
        points[f"{center_eye_prefix}ry"]=eul[:,1]
        points[f"{center_eye_prefix}rz"]=eul[:,2]
    return points


# ── KDE affine 補正 (validate_affine_correction.py と同一ロジック) ─────────────

def detect_peaks_kde(vals: np.ndarray) -> np.ndarray:
    """KDE のローカル最大値をピーク位置として返す。"""
    if len(vals) < 10:
        return np.array([float(np.median(vals))])
    kde = gaussian_kde(vals, bw_method=KDE_BW)
    xgrid = np.linspace(vals.min(), vals.max(), 2000)
    density = kde(xgrid)
    min_dist = max(1, int(MIN_PEAK_GAP_M / (xgrid[1] - xgrid[0])))
    peak_idx, _ = find_peaks(density, height=density.max()*0.03, distance=min_dist)
    if len(peak_idx) == 0:
        return np.array([float(xgrid[np.argmax(density)])])
    return np.sort(xgrid[peak_idx])


def fit_affine(peaks: np.ndarray) -> tuple[float, float, str]:
    """
    KDE ピークを物理高さに Hungarian マッチングして affine (a, b) を計算。
    |peak - physical| > MAX_MATCH_DIST_M のペアは除外。
    """
    n = len(peaks)
    if n == 0:
        return 1.0, 0.0, "no_peaks"
    if n == 1:
        nearest = PHYSICAL_HEIGHTS_M[np.argmin(np.abs(PHYSICAL_HEIGHTS_M - peaks[0]))]
        if abs(peaks[0] - nearest) > MAX_MATCH_DIST_M:
            return 1.0, 0.0, "no_peaks"
        return 1.0, nearest - peaks[0], "offset_only"

    use_peaks = peaks[:4]
    cost = np.abs(use_peaks[:, None] - PHYSICAL_HEIGHTS_M[None, :])
    ri, ci = linear_sum_assignment(cost)
    mp, mh = use_peaks[ri], PHYSICAL_HEIGHTS_M[ci]
    valid = np.abs(mp - mh) <= MAX_MATCH_DIST_M
    mp_v, mh_v = mp[valid], mh[valid]

    if len(mp_v) == 0:
        return 1.0, 0.0, "no_peaks"
    if len(mp_v) == 1:
        return 1.0, float(mh_v[0] - mp_v[0]), "offset_only"

    slope, intercept = np.polyfit(mp_v, mh_v, 1)
    status = "full_affine" if len(mp_v) >= 4 else f"partial_affine_{len(mp_v)}peaks"
    return float(slope), float(intercept), status


# ── pickle → CSV ──────────────────────────────────────────────────────────────

def load_pickle_to_df(pickle_path: Path, n_frames: int = -1) -> pd.DataFrame:
    """pickle を読み込み、回転済みキーポイント座標を DataFrame で返す (mm)。"""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    X = data["kp3d"]       # (n_animal, n_frame, n_kp, 3)
    S = data["kp3d_score"]

    n_animal, n_frame, n_kp, _ = X.shape
    if n_frames < 0 or n_frames > n_frame:
        n_frames = n_frame

    i_animal = 0
    Xa = X[i_animal, :n_frames].copy()
    Sa = S[i_animal, :n_frames].copy()

    kp_names = KP_NAMES[:]

    # 中点キーポイントを追加
    for (i1, i2, name) in [(5, 6, "shoulder_mid"), (11, 12, "base"), (1, 2, "center_eye")]:
        if max(i1, i2) < Xa.shape[1]:
            mid = (Xa[:, i1] + Xa[:, i2]) / 2
            Xa = np.concatenate([Xa, mid[:, None, :]], axis=1)
            Sa = np.concatenate([Sa, ((Sa[:, i1]+Sa[:, i2])/2)[:, None]], axis=1)
            kp_names.append(name)

    R = rot_euler(rx=-90, ry=0, rz=180)
    Xa = Xa @ R.T

    kp_keys = unique_keys(kp_names)
    n = Xa.shape[0]
    d = {"i_frame": np.arange(n, dtype=np.int64), "i_animal": np.zeros(n, dtype=np.int64)}
    for ki, key in enumerate(kp_keys):
        d[f"pts_{key}_x"] = Xa[:, ki, 0]
        d[f"pts_{key}_y"] = Xa[:, ki, 1]
        d[f"pts_{key}_z"] = Xa[:, ki, 2]

    return pd.DataFrame(d)


# ── グローバルスケール → bZ → affine 補正 → 出力 ──────────────────────────────

def scale_and_correct(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, float, float, str]:
    """
    回転済み CSV にグローバルスケールを適用後、
    KDE affine 補正を bZ 軸に適用して返す。

    Returns
    -------
    df_corrected : 補正済み DataFrame (bZ in m)
    slope        : affine slope a
    intercept    : affine intercept b
    status       : affine status
    """
    df = df_raw.copy()
    half_span = BLENDER_BBOX_SCALE / 2
    axis_map = {"_x": 0, "_y": 1, "_z": 2}

    # グローバルスケール適用
    for col in df.columns[2:]:
        for suffix, idx in axis_map.items():
            if col.endswith(suffix):
                df[col] = (
                    (df[col] - KP3D_CENTER[idx]) / KP3D_NORM_SCALE[idx]
                    * half_span[idx] + BLENDER_CENTER[idx]
                )
                break

    # floor KP の bZ 分布から KDE affine を計算
    z_cols = [c for c in df.columns
              if c.endswith("_z") and any(k in c for k in FLOOR_KP_KEYWORDS)]
    if z_cols:
        vals_all = df[z_cols].min(axis=1).values
        lo, hi   = np.percentile(vals_all, 1), np.percentile(vals_all, 99)
        vals     = vals_all[(vals_all >= lo) & (vals_all <= hi)]
        peaks    = detect_peaks_kde(vals)
        a, b, status = fit_affine(peaks)
    else:
        a, b, status = 1.0, 0.0, "no_floor_cols"

    # bZ に affine 適用
    for col in df.columns:
        if col.endswith("_z"):
            df[col] = a * df[col] + b

    # 全軸に OUTPUT_SCALE を適用
    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    df[coord_cols] *= OUTPUT_SCALE

    df = add_cam_pose_with_euler(df)

    return df, a, b, status


# ── 1セッション処理 ────────────────────────────────────────────────────────────

def process_session(
    session: str,
    base_dir: Path,
    csv_dir: Path,
    out_dir: Path,
    n_frames: int = -1,
) -> dict:
    pickle_path = base_dir / session / "kp3d_fxdJointLen.pickle"
    if not pickle_path.exists():
        pickle_path = base_dir / session / "kp3d.pickle"
    if not pickle_path.exists():
        raise FileNotFoundError(f"pickle not found: {base_dir / session}")

    df_raw = load_pickle_to_df(pickle_path, n_frames)

    # 中間 CSV (rotated mm) を保存
    csv_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = csv_dir / f"{session}_points.csv"
    df_raw.to_csv(raw_csv, index=False)

    # グローバルスケール + KDE affine 補正
    df_corr, a, b, status = scale_and_correct(df_raw)

    # 補正済み CSV を保存
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{session}_corrected.csv"
    df_corr.to_csv(out_csv, index=False)

    slope_warn = abs(a - 1.0) > MAX_AFFINE_SLOPE_DEV
    print(f"  [{status}] a={a:.4f} b={b:.4f}"
          + (" WARN: |a-1|>0.15" if slope_warn else ""))

    return {
        "session":    session,
        "animal":     session.split("_")[0],
        "n_frames":   len(df_raw),
        "slope":      round(a, 4),
        "intercept":  round(b, 4),
        "status":     status,
        "slope_warn": slope_warn,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="pickle → KDE affine 補正済み Blender CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  OPENBLAS_NUM_THREADS=1 python scripts/dev/affine_pkl_csv/pkl_to_corrected_csv.py \\\n"
            "      --session cj425_20210427_120000\n"
            "  OPENBLAS_NUM_THREADS=1 python scripts/dev/affine_pkl_csv/pkl_to_corrected_csv.py --all"
        ),
    )
    parser.add_argument("--session",   default=None, help="セッション名（単一処理）")
    parser.add_argument("--all",       action="store_true", help="--base-dir 以下の全セッションを処理")
    parser.add_argument("--n-frames",  type=int, default=-1, help="処理フレーム数 (-1=全フレーム)")
    parser.add_argument("--base-dir",  type=Path, default=BASE_DIR)
    parser.add_argument("--out-root",  type=Path, default=OUT_ROOT,
                        help="出力ルート (default: outputs/dev/affine_pkl_csv)")
    args = parser.parse_args()

    if args.all and args.session:
        parser.error("--session と --all は同時に指定できません")
    if not args.all and not args.session:
        parser.error("--session か --all のどちらかを指定してください")

    csv_dir = args.out_root / "csv"
    out_dir = args.out_root / "corrected"
    log_path = args.out_root / "affine_log.csv"

    sessions = (
        sorted(d.name for d in args.base_dir.iterdir() if d.is_dir())
        if args.all else [args.session]
    )
    print(f"{len(sessions)} セッションを処理します")

    records = []
    skipped = []
    for session in sessions:
        print(f"\n=== {session} ===")
        try:
            rec = process_session(session, args.base_dir, csv_dir, out_dir, args.n_frames)
            records.append(rec)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}", file=sys.stderr)
            skipped.append(session)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            skipped.append(session)

    if records:
        df_log = pd.DataFrame(records)
        df_log.to_csv(log_path, index=False)
        print(f"\naffine log: {log_path}")

        df = df_log[df_log["status"].notna()]
        print(f"\n{'Status':25} {'Count':>6}")
        for s, grp in df.groupby("status"):
            print(f"  {s:23} {len(grp):>6}")
        n_warn = int(df["slope_warn"].sum())
        print(f"\nSlope warnings: {n_warn} ({100*n_warn/len(df):.0f}%)")
        print(f"Slope  mean={df['slope'].mean():.4f}  std={df['slope'].std():.4f}  "
              f"min={df['slope'].min():.4f}  max={df['slope'].max():.4f}")
        print(f"Intercept mean={df['intercept'].mean():.4f}  std={df['intercept'].std():.4f}")

    if skipped:
        print(f"\n{len(skipped)} セッションをスキップ: {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
