"""
floorZ 共有ライブラリ（アルゴリズム本体）。

run/ と plot/ の両方から import される。CLI は持たない。

提供するもの:
  - 定数 (KP 名 / Blender 変換 / 物理高さ / KDE・affine パラメータ / 個体色)
  - 座標ユーティリティ (sanitize / 回転 / カメラ姿勢)
  - KDE ピーク検出 detect_peaks_kde / affine 回帰 fit_affine
  - pickle 入出力 load_pickle_to_df / floor_bz_series
  - グローバルスケール + affine 補正 scale_and_correct

垂直 affine 補正の考え方:
  各フレームで床接地 KP (wrist/ankle) の min(bZ) をとると、床・足場・棒・天井の
  滞在高さがモードとして分布に現れる。これを KDE で検出し、物理高さ
  PHYSICAL_HEIGHTS_M に Hungarian マッチング → 線形回帰し、bZ_corrected = a·bZ + b
  で各セッションを補正する。
"""
from __future__ import annotations

import pickle
import re
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

# 床接地キーポイント（インデックスと名前キーワード）
FLOOR_KP_IDX      = [9, 10, 15, 16]          # L/R wrist, L/R ankle
FLOOR_KP_KEYWORDS = ("wrist", "ankle")

# kp3d (mm) → Blender (m) のグローバル変換
BLENDER_BBOX_SCALE = np.array([0.86658, 0.88791, 0.7679])      # フルスパン
BLENDER_CENTER     = np.array([0.0, -0.11142, 1.11765])
KP3D_CENTER        = np.array([-1131.005, -1072.940, 286.610])
KP3D_NORM_SCALE    = np.array([449.675, 439.560, 378.880])

# 物理高さ (Blender bZ, m): 床 / 足場 / 棒 / 天井
PHYSICAL_HEIGHTS_M = np.array([0.7337, 1.0179, 1.3035, 1.5016])
PHYS_LABELS        = ["floor", "low", "bar", "ceil"]

# KDE + affine パラメータ
KDE_BW               = 0.03    # KDE bandwidth (m)
KDE_GRID_N           = 2000    # KDE 評価点数
MIN_PEAK_GAP_M       = 0.10    # 隣接ピーク間の最小距離 (m)
MAX_MATCH_DIST_M     = 0.10    # peak-physical のマッチ許容距離 (m)
MAX_AFFINE_SLOPE_DEV = 0.15    # |slope - 1| がこれを超えたら警告

ANIMAL_COLORS = {"cj425": "steelblue", "cj611": "darkorange", "cj357": "seagreen"}


# ── 座標ユーティリティ ─────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    key = name.lower().strip().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", key)


def unique_keys(names):
    seen, out = {}, []
    for n in names:
        base = sanitize(n)
        k, i = base, 2
        while k in seen:
            k = f"{base}_{i}"
            i += 1
        seen[k] = True
        out.append(k)
    return out


def rot_x(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_euler(rx=0.0, ry=0.0, rz=0.0, order="xyz"):
    mats = {"x": rot_x(rx), "y": rot_y(ry), "z": rot_z(rz)}
    R = np.eye(3)
    for ax in order:
        R = mats[ax] @ R
    return R


# kp3d → Blender 整列で使う固定回転 (x,y,z)->(-x,-z,-y) 相当
KP3D_ROT = rot_euler(rx=-90, ry=0, rz=180)


def _safe_norm(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _rotmat_to_euler_xyz(R):
    r00 = R[..., 0, 0]; r10 = R[..., 1, 0]; r20 = R[..., 2, 0]
    r11 = R[..., 1, 1]; r12 = R[..., 1, 2]
    r21 = R[..., 2, 1]; r22 = R[..., 2, 2]
    sy = np.sqrt(r00 * r00 + r10 * r10)
    singular = sy < 1e-8
    rx = np.where(~singular, np.arctan2(r21, r22), np.arctan2(-r12, r11))
    ry = np.where(~singular, np.arctan2(-r20, sy), np.arctan2(-r20, sy))
    rz = np.where(~singular, np.arctan2(r10, r00), np.zeros_like(r00))
    return np.stack([rx, ry, rz], axis=-1)


def add_cam_pose_with_euler(points, l_eye="pts_l_eye_", r_eye="pts_r_eye_",
                            l_ear="pts_l_ear_", r_ear="pts_r_ear_",
                            out_prefixes=("pts_l_cam_", "pts_r_cam_"),
                            center_eye_prefix="pts_center_eye_", forward_sign=-1):
    need = ([f"{l_eye}{a}" for a in "xyz"] + [f"{r_eye}{a}" for a in "xyz"]
            + [f"{l_ear}{a}" for a in "xyz"] + [f"{r_ear}{a}" for a in "xyz"])
    for c in need:
        if c not in points.columns:
            raise ValueError(f"列がありません: {c}")
    LE = points[[f"{l_eye}x", f"{l_eye}y", f"{l_eye}z"]].to_numpy(float)
    RE = points[[f"{r_eye}x", f"{r_eye}y", f"{r_eye}z"]].to_numpy(float)
    LEr = points[[f"{l_ear}x", f"{l_ear}y", f"{l_ear}z"]].to_numpy(float)
    REr = points[[f"{r_ear}x", f"{r_ear}y", f"{r_ear}z"]].to_numpy(float)
    eye_c = (LE + RE) * 0.5; ear_c = (LEr + REr) * 0.5
    s_n = _safe_norm(RE - LE)
    f_n = _safe_norm((eye_c - ear_c) * float(forward_sign))
    u_raw = np.cross(f_n, s_n)
    mask_bad = ((np.linalg.norm(u_raw, axis=1) < 1e-8)
                | np.isnan(u_raw).any(axis=1)
                | np.isnan(s_n).any(axis=1)
                | np.isnan(f_n).any(axis=1))
    if np.any(mask_bad):
        up = np.tile(np.array([0., 0., 1.])[None, :], (len(points), 1))
        u_raw[mask_bad] = np.cross(f_n[mask_bad], up[mask_bad])
    u_n = _safe_norm(u_raw)
    s_ortho = _safe_norm(np.cross(u_n, f_n))
    R = np.stack([s_ortho, u_n, f_n], axis=-1)
    det = np.linalg.det(R)
    if np.any(det < 0):
        R[det < 0, :, 0] *= -1.
    eul = _rotmat_to_euler_xyz(R)
    for pfx in out_prefixes:
        points[f"{pfx}rx"] = eul[:, 0]; points[f"{pfx}ry"] = eul[:, 1]; points[f"{pfx}rz"] = eul[:, 2]
    if center_eye_prefix:
        points[f"{center_eye_prefix}rx"] = eul[:, 0]
        points[f"{center_eye_prefix}ry"] = eul[:, 1]
        points[f"{center_eye_prefix}rz"] = eul[:, 2]
    return points


# ── KDE ピーク検出 / affine 回帰 ───────────────────────────────────────────────

def detect_peaks_kde(vals: np.ndarray) -> np.ndarray:
    """床 KP bZ 値の KDE ローカル最大値をピーク位置 (m, 昇順) として返す。"""
    vals = np.asarray(vals, float)
    if len(vals) < 10:
        return np.array([float(np.median(vals))])
    kde = gaussian_kde(vals, bw_method=KDE_BW)
    xgrid = np.linspace(vals.min(), vals.max(), KDE_GRID_N)
    density = kde(xgrid)
    min_dist = max(1, int(MIN_PEAK_GAP_M / (xgrid[1] - xgrid[0])))
    peak_idx, _ = find_peaks(density, height=density.max() * 0.03, distance=min_dist)
    if len(peak_idx) == 0:
        return np.array([float(xgrid[np.argmax(density)])])
    return np.sort(xgrid[peak_idx])


def kde_curve(vals: np.ndarray, n: int = KDE_GRID_N):
    """(xgrid, density) を返す。プロット用。"""
    vals = np.asarray(vals, float)
    kde = gaussian_kde(vals, bw_method=KDE_BW)
    xg = np.linspace(vals.min(), vals.max(), n)
    return xg, kde(xg)


def fit_affine(peaks: np.ndarray, return_pairs: bool = False):
    """
    検出ピークを物理高さに Hungarian マッチングして affine (slope, intercept) を計算。

    MAX_MATCH_DIST_M を超えるペアは除外する。

    return_pairs=False -> (slope, intercept, status)
    return_pairs=True  -> (slope, intercept, status, matched_peaks, matched_phys)
    """
    peaks = np.asarray(peaks, float)
    n = len(peaks)
    empty = (np.array([]), np.array([]))

    def _ret(a, b, status, mp=empty[0], mh=empty[1]):
        return (a, b, status, mp, mh) if return_pairs else (a, b, status)

    if n == 0:
        return _ret(1.0, 0.0, "no_peaks")
    if n == 1:
        nearest = PHYSICAL_HEIGHTS_M[np.argmin(np.abs(PHYSICAL_HEIGHTS_M - peaks[0]))]
        if abs(peaks[0] - nearest) > MAX_MATCH_DIST_M:
            return _ret(1.0, 0.0, "no_peaks")
        return _ret(1.0, float(nearest - peaks[0]), "offset_only",
                    peaks[:1], np.array([nearest]))

    use_peaks = peaks[:4]
    cost = np.abs(use_peaks[:, None] - PHYSICAL_HEIGHTS_M[None, :])
    ri, ci = linear_sum_assignment(cost)
    mp, mh = use_peaks[ri], PHYSICAL_HEIGHTS_M[ci]
    valid = np.abs(mp - mh) <= MAX_MATCH_DIST_M
    mp_v, mh_v = mp[valid], mh[valid]

    if len(mp_v) == 0:
        return _ret(1.0, 0.0, "no_peaks")
    if len(mp_v) == 1:
        return _ret(1.0, float(mh_v[0] - mp_v[0]), "offset_only", mp_v, mh_v)

    slope, intercept = np.polyfit(mp_v, mh_v, 1)
    status = "full_affine" if len(mp_v) >= 4 else f"partial_affine_{len(mp_v)}peaks"
    return _ret(float(slope), float(intercept), status, mp_v, mh_v)


def slope_warn(slope: float) -> bool:
    return abs(slope - 1.0) > MAX_AFFINE_SLOPE_DEV


# ── 床 KP の bZ 抽出 ───────────────────────────────────────────────────────────

def trim_1_99(vals: np.ndarray) -> np.ndarray:
    """1〜99 パーセンタイルでトリムする。"""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return vals
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    return vals[(vals >= lo) & (vals <= hi)]


def floor_z_cols(columns) -> list[str]:
    """CSV 列名から床 KP の _z 列を選ぶ。"""
    return [c for c in columns
            if c.endswith("_z") and any(k in c for k in FLOOR_KP_KEYWORDS)]


def floor_bz_series(pickle_path: Path, n_frames: int = -1) -> np.ndarray:
    """
    pickle → 回転 → グローバルスケール → 各フレーム min(床KP bZ) の系列 (m, NaN 込み)。

    補正前の bZ。トリムは呼び出し側で行う。
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    X = data["kp3d"][0]                       # (T, n_kp, 3)
    if n_frames and n_frames > 0:
        X = X[:n_frames]
    X = X @ KP3D_ROT.T
    half_z = BLENDER_BBOX_SCALE[2] / 2
    bz = (X[:, :, 2] - KP3D_CENTER[2]) / KP3D_NORM_SCALE[2] * half_z + BLENDER_CENTER[2]
    return np.nanmin(bz[:, FLOOR_KP_IDX], axis=1)


def resolve_pickle(session_dir: Path) -> Path | None:
    """kp3d_fxdJointLen.pickle を優先し、無ければ kp3d.pickle を返す。"""
    for name in ("kp3d_fxdJointLen.pickle", "kp3d.pickle"):
        p = session_dir / name
        if p.exists():
            return p
    return None


# ── pickle → DataFrame → 補正 ──────────────────────────────────────────────────

def load_pickle_to_df(pickle_path: Path, n_frames: int = -1) -> pd.DataFrame:
    """pickle を読み込み、回転済みキーポイント座標 (mm) を DataFrame で返す。"""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    X = data["kp3d"]       # (n_animal, n_frame, n_kp, 3)
    S = data["kp3d_score"]
    _, n_frame, _, _ = X.shape
    if n_frames < 0 or n_frames > n_frame:
        n_frames = n_frame

    Xa = X[0, :n_frames].copy()
    Sa = S[0, :n_frames].copy()
    kp_names = KP_NAMES[:]

    for (i1, i2, name) in [(5, 6, "shoulder_mid"), (11, 12, "base"), (1, 2, "center_eye")]:
        if max(i1, i2) < Xa.shape[1]:
            mid = (Xa[:, i1] + Xa[:, i2]) / 2
            Xa = np.concatenate([Xa, mid[:, None, :]], axis=1)
            Sa = np.concatenate([Sa, ((Sa[:, i1] + Sa[:, i2]) / 2)[:, None]], axis=1)
            kp_names.append(name)

    Xa = Xa @ KP3D_ROT.T
    kp_keys = unique_keys(kp_names)
    n = Xa.shape[0]
    d = {"i_frame": np.arange(n, dtype=np.int64), "i_animal": np.zeros(n, dtype=np.int64)}
    for ki, key in enumerate(kp_keys):
        d[f"pts_{key}_x"] = Xa[:, ki, 0]
        d[f"pts_{key}_y"] = Xa[:, ki, 1]
        d[f"pts_{key}_z"] = Xa[:, ki, 2]
    return pd.DataFrame(d)


def apply_global_scale(df: pd.DataFrame) -> pd.DataFrame:
    """回転済み mm DataFrame にグローバルスケールを適用し Blender m 空間にする。"""
    df = df.copy()
    half_span = BLENDER_BBOX_SCALE / 2
    axis_map = {"_x": 0, "_y": 1, "_z": 2}
    for col in df.columns[2:]:
        for suffix, idx in axis_map.items():
            if col.endswith(suffix):
                df[col] = ((df[col] - KP3D_CENTER[idx]) / KP3D_NORM_SCALE[idx]
                           * half_span[idx] + BLENDER_CENTER[idx])
                break
    return df


def scale_and_correct(df_raw: pd.DataFrame, output_scale: float = 1.0):
    """
    回転済み CSV にグローバルスケール → KDE affine 補正 (bZ) → output_scale を適用。

    Returns (df_corrected, slope, intercept, status)
    """
    df = apply_global_scale(df_raw)

    z_cols = floor_z_cols(df.columns)
    if z_cols:
        vals = trim_1_99(df[z_cols].min(axis=1).values)
        peaks = detect_peaks_kde(vals)
        a, b, status = fit_affine(peaks)
    else:
        a, b, status = 1.0, 0.0, "no_floor_cols"

    for col in df.columns:
        if col.endswith("_z"):
            df[col] = a * df[col] + b

    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    df[coord_cols] *= output_scale
    df = add_cam_pose_with_euler(df)
    return df, a, b, status
