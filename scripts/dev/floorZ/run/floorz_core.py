"""
floorZ 共有ライブラリ（アルゴリズム本体）。

run/ と plot/ の両方から import される。CLI は持たない。

提供するもの:
  - 定数 (KP 名 / Blender 変換 / 物理高さ / KDE・affine パラメータ / 個体色)
  - 座標ユーティリティ (sanitize / 回転 / カメラ姿勢)
  - KDE ピーク検出 detect_peaks_kde / affine 回帰 fit_affine
  - kp3d I/O: load_kp3d (pickle/npz) / load_pickle_to_df / floor_bz_series
  - グローバルスケール + affine 補正 scale_and_correct

垂直 affine 補正の考え方:
  各フレームで床接地 KP (wrist/ankle) の min(bZ) をとると、床・足場・棒・天井の
  滞在高さがモードとして分布に現れる。これを KDE で検出し、物理高さ
  PHYSICAL_HEIGHTS_M に Hungarian マッチング → 線形回帰し、bZ_corrected = a·bZ + b
  で各セッションを補正する。
"""
from __future__ import annotations

import itertools
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/dev
import _config as cfg  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from marmo3d.kp3d_io import load_kp3d, resolve_kp3d  # noqa: E402,F401

# ── 定数 ──────────────────────────────────────────────────────────────────────
# 物理キャリブレーション・骨格・座標変換は docs/method.yaml (= cfg) を単一ソースとする。

KP_NAMES = cfg.KP_NAMES_CANONICAL    # "L eye" 形 (sanitize でキー化する設計)

# 床接地キーポイント（インデックスと名前キーワード）
FLOOR_KP_IDX      = cfg.FLOOR_KP_IDX          # L/R wrist, L/R ankle
FLOOR_KP_KEYWORDS = cfg.FLOOR_KP_KEYWORDS

# kp3d (mm) → Blender (m) のグローバル変換
BLENDER_BBOX_SCALE = cfg.BLENDER_BBOX_SCALE   # フルスパン
BLENDER_CENTER     = cfg.BLENDER_CENTER
KP3D_CENTER        = cfg.KP3D_CENTER
KP3D_NORM_SCALE    = cfg.KP3D_NORM_SCALE

# 物理高さ (Blender bZ, m): 床 / 足場 / 棒 / 天井
PHYSICAL_HEIGHTS_M = cfg.PHYSICAL_HEIGHTS_M
PHYS_LABELS        = cfg.PHYS_LABELS

# KDE + affine パラメータ (アルゴリズムのチューニング値。config には置かない)
KDE_BW               = 0.03    # KDE bandwidth (m)
KDE_GRID_N           = 2000    # KDE 評価点数
MIN_PEAK_GAP_M       = 0.10    # 隣接ピーク間の最小距離 (m)
MAX_MATCH_DIST_M     = 0.10    # peak-physical のマッチ許容距離 (m)
MAX_AFFINE_SLOPE_DEV = 0.15    # |slope - 1| がこれを超えたら警告

ANIMAL_COLORS = cfg.ANIMAL_COLORS


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
KP3D_ROT = rot_euler(rx=cfg.ROTATION_DEG["rx"], ry=cfg.ROTATION_DEG["ry"],
                     rz=cfg.ROTATION_DEG["rz"])


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

def detect_peaks_kde(vals: np.ndarray,
                     return_density: bool = False):
    """床 KP bZ 値の KDE ローカル最大値をピーク位置 (m, 昇順) として返す。

    return_density=True のとき (peaks, densities) を返す。
    densities は各ピーク位置での KDE 密度値 (peaks と同順)。
    """
    vals = np.asarray(vals, float)
    if len(vals) < 10:
        peaks = np.array([float(np.median(vals))])
        return (peaks, np.array([1.0])) if return_density else peaks
    kde = gaussian_kde(vals, bw_method=KDE_BW)
    xgrid = np.linspace(vals.min(), vals.max(), KDE_GRID_N)
    density = kde(xgrid)
    min_dist = max(1, int(MIN_PEAK_GAP_M / (xgrid[1] - xgrid[0])))
    peak_idx, _ = find_peaks(density, height=density.max() * 0.03, distance=min_dist)
    if len(peak_idx) == 0:
        peak_idx = np.array([np.argmax(density)])
    sort_order = np.argsort(xgrid[peak_idx])
    peaks = xgrid[peak_idx[sort_order]]
    if return_density:
        return peaks, density[peak_idx[sort_order]]
    return peaks


def kde_curve(vals: np.ndarray, n: int = KDE_GRID_N):
    """(xgrid, density) を返す。プロット用。"""
    vals = np.asarray(vals, float)
    kde = gaussian_kde(vals, bw_method=KDE_BW)
    xg = np.linspace(vals.min(), vals.max(), n)
    return xg, kde(xg)


def fit_affine(peaks: np.ndarray, return_pairs: bool = False,
               peak_densities: np.ndarray | None = None,
               legacy: bool = False):
    """
    検出ピークを物理高さにマッチングして affine (slope, intercept) を計算。

    MAX_MATCH_DIST_M を超えるペアは除外する。

    peak_densities: ピークごとの KDE 密度値 (detect_peaks_kde の return_density=True で取得)。
                    渡すと密度優先グリーディーマッチングを使用し、dominant ピークが
                    minor ピークに押し出される問題を回避する。
    legacy: True で Hungarian マッチング（旧動作）を強制。比較・検証用。
            peak_densities=None の場合も旧動作にフォールバックする。

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
    use_legacy = legacy or (peak_densities is None)

    if use_legacy:
        # Hungarian マッチング（旧動作）
        cost = np.abs(use_peaks[:, None] - PHYSICAL_HEIGHTS_M[None, :])
        ri, ci = linear_sum_assignment(cost)
        mp, mh = use_peaks[ri], PHYSICAL_HEIGHTS_M[ci]
    else:
        # 密度優先グリーディーマッチング:
        # 密度の高いピークから順に最近傍の未使用物理高さへ割り当てる。
        # dominant ピークが minor ピークに物理高さを「消費」されるのを防ぐ。
        densities = np.asarray(peak_densities, float)[:len(use_peaks)]
        order = np.argsort(densities)[::-1]
        used_phys: set[int] = set()
        matched_p, matched_h = [], []
        for i in order:
            pk = use_peaks[i]
            candidates = [(j, h) for j, h in enumerate(PHYSICAL_HEIGHTS_M)
                          if j not in used_phys]
            if not candidates:
                break
            best_j, best_h = min(candidates, key=lambda x: abs(pk - x[1]))
            matched_p.append(pk)
            matched_h.append(best_h)
            used_phys.add(best_j)
        mp = np.array(matched_p)
        mh = np.array(matched_h)

    valid = np.abs(mp - mh) <= MAX_MATCH_DIST_M
    mp_v, mh_v = mp[valid], mh[valid]

    # dominant ピークが未マッチの場合に警告
    if not use_legacy and peak_densities is not None and len(peaks) > 1:
        dens_arr = np.asarray(peak_densities, float)
        max_i = int(np.argmax(dens_arr))
        max_peak = peaks[max_i]
        if not any(abs(max_peak - m) < 0.001 for m in mp_v):
            second = np.partition(dens_arr, -2)[-2]
            ratio = dens_arr[max_i] / second if second > 0 else 999.0
            if ratio > 5.0:
                warnings.warn(
                    f"dominant KDE peak {max_peak:.4f}m (density_ratio={ratio:.1f}x) "
                    "unmatched to any physical height; affine may be inaccurate",
                    UserWarning, stacklevel=2,
                )

    if len(mp_v) == 0:
        return _ret(1.0, 0.0, "no_peaks")
    if len(mp_v) == 1:
        return _ret(1.0, float(mh_v[0] - mp_v[0]), "offset_only", mp_v, mh_v)

    slope, intercept = np.polyfit(mp_v, mh_v, 1)
    status = "full_affine" if len(mp_v) >= 4 else f"partial_affine_{len(mp_v)}peaks"
    return _ret(float(slope), float(intercept), status, mp_v, mh_v)


def slope_warn(slope: float) -> bool:
    return abs(slope - 1.0) > MAX_AFFINE_SLOPE_DEV


# ── max% アンカー（上方向の到達点）と KDE ピークの統合 ───────────────────────────
# 信号 = 各フレームの「全KP の最大 bZ」(最高到達点)。その分布の上位 pct% 平均を
# 1 つの max% ピーク P とし、KDE ピーク群へ統合する（仕様: 表参照）。

def top_pct_peak(vals: np.ndarray, pct: float) -> float:
    """上位 pct% の平均を返す。pct<=0 のときは最大値（max のみ）。

    上裾こそ見たい信号なのでトリムはしない（NaN/inf のみ除去）。
    """
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    if pct <= 0:
        return float(vals.max())
    thr = np.percentile(vals, 100.0 - pct)
    top = vals[vals >= thr]
    return float(top.mean()) if len(top) else float(vals.max())


def merge_max_peak(kde_peaks: np.ndarray, P: float,
                   thr: float = MAX_MATCH_DIST_M):
    """max% ピーク P を KDE ピーク群に統合して最終アンカー集合を返す。

    仕様（棒 bar=1.3035 / 天井 ceil=1.5016 の最近傍分類）:
      - P が棒/天井いずれからも > thr → P 破棄 + warning (discard_far)
      - P と最近傍 KDE ピーク Q が同一プラットフォーム (|Q-P|<=thr):
          P=棒   → KDE を使い P を消す (drop_P)
          P=天井 → Q を消し P を使う   (replace_kde)
      - 別プラットフォーム (|Q-P|>thr):
          P=棒 かつ 既に天井 KDE ピークあり → 矛盾 → KDE のみ + warning (contradict_warn)
          それ以外 → P を新規アンカーとして追加
            (P=天井 かつ 棒 KDE あり=both / それ以外=far_new)

    注: 棒↔天井は 0.198m 離れるため「棒(Q)×天井(P)」は別プラットフォーム扱い(both)。
    Returns: (final_peaks 昇順, label, warn_msg)
    """
    bar  = float(PHYSICAL_HEIGHTS_M[PHYS_LABELS.index("bar")])
    ceil = float(PHYSICAL_HEIGHTS_M[PHYS_LABELS.index("ceil")])
    kde_peaks = np.sort(np.asarray(kde_peaks, float))

    if not np.isfinite(P):
        return kde_peaks, "discard_far", "max-peak が NaN"
    d_bar, d_ceil = abs(P - bar), abs(P - ceil)
    if min(d_bar, d_ceil) > thr:
        return kde_peaks, "discard_far", f"max-peak {P:.4f}m が棒/天井から>{thr}m"
    cls_P = "bar" if d_bar <= d_ceil else "ceil"

    if len(kde_peaks) == 0:
        return np.array([P]), "far_new", ""

    qi = int(np.argmin(np.abs(kde_peaks - P)))
    Q = float(kde_peaks[qi])

    if abs(Q - P) <= thr:           # Q と P は同一プラットフォーム
        if cls_P == "bar":
            return kde_peaks, "drop_P", ""
        merged = np.delete(kde_peaks, qi)
        return np.sort(np.append(merged, P)), "replace_kde", ""

    # 別プラットフォーム
    has_ceil_kde = bool(np.any(np.abs(kde_peaks - ceil) <= thr))
    if cls_P == "bar" and has_ceil_kde:
        return kde_peaks, "contradict_warn", \
            f"KDE に天井ピークあり だが max-peak {P:.4f}m=棒: 矛盾のため KDE のみ採用"
    has_bar_kde = bool(np.any(np.abs(kde_peaks - bar) <= thr))
    label = "both" if (cls_P == "ceil" and has_bar_kde) else "far_new"
    return np.sort(np.append(kde_peaks, P)), label, ""


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


def real_kp_z_cols(columns) -> list[str]:
    """実 20 KP の _z 列を選ぶ（派生点 shoulder_mid/base/center_eye は除外）。"""
    return [f"pts_{k}_z" for k in cfg.KP_KEYS if f"pts_{k}_z" in columns]


def floor_bz_series(pickle_path: Path, n_frames: int = -1) -> np.ndarray:
    """
    pickle → 回転 → グローバルスケール → 各フレーム min(床KP bZ) の系列 (m, NaN 込み)。

    補正前の bZ。トリムは呼び出し側で行う。
    """
    data = load_kp3d(pickle_path)
    X = data["kp3d"][0]                       # (T, n_kp, 3)
    if n_frames and n_frames > 0:
        X = X[:n_frames]
    X = X @ KP3D_ROT.T
    half_z = BLENDER_BBOX_SCALE[2] / 2
    bz = (X[:, :, 2] - KP3D_CENTER[2]) / KP3D_NORM_SCALE[2] * half_z + BLENDER_CENTER[2]
    return np.nanmin(bz[:, FLOOR_KP_IDX], axis=1)


def resolve_pickle(session_dir: Path) -> Path | None:
    """Return kp3d_fxdJointLen if present, else kp3d (npz before pickle). Name kept for compatibility."""
    return resolve_kp3d(session_dir)


# ── pickle → DataFrame → 補正 ──────────────────────────────────────────────────

def load_pickle_to_df(pickle_path: Path, n_frames: int = -1) -> pd.DataFrame:
    """pickle を読み込み、回転済みキーポイント座標 (mm) を DataFrame で返す。"""
    data = load_kp3d(pickle_path)

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


def compute_animal_norm_stats(pickle_paths, n_frames: int = -1,
                              pct: tuple[float, float] = (2.0, 98.0)):
    """
    複数 pickle から回転済み XYZ の下限/上限パーセンタイルで center/scale を計算する。

    個体ごとの水平・垂直方向規格化に使う。
    pct: (lo, hi) 規格化に使うパーセンタイル下限/上限。既定 (2, 98)。
    戻り値: center (3,), scale (3,)  単位 mm, 回転後空間。
    """
    pct_lo, pct_hi = pct
    bufs = [[], [], []]
    for p in pickle_paths:
        data = load_kp3d(p)
        X = data["kp3d"][0]   # (T, n_kp, 3)
        if n_frames and n_frames > 0:
            X = X[:n_frames]
        X = X @ KP3D_ROT.T
        for ax in range(3):
            bufs[ax].append(X[:, :, ax].ravel())
    centers, scales = [], []
    for ax in range(3):
        v = np.concatenate(bufs[ax])
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, pct_lo), np.percentile(v, pct_hi)
        centers.append((lo + hi) / 2)
        scales.append((hi - lo) / 2)
    return np.array(centers), np.array(scales)


# ── global norm (percentile キャッシュ付き) ──────────────────────────────────────
# method.yaml の coordinate_transform.kp3d_global_norm を percentile キャッシュとして
# 使う。指定 percentile が既にあれば読むだけ、無ければ全 pickle から計測空間 bbox を
# 計算して追記する。キャッシュ値は計測空間 mm の inlier bbox lo/hi (回転前)。

def compute_global_inlier_bbox_mm(pickle_paths, n_frames: int = -1,
                                  pct: tuple[float, float] = (2.0, 98.0)) -> dict:
    """全 pickle をプールし、計測空間 mm の各軸 percentile lo/hi を返す（回転前）。

    kp3d_inlier_bbox_mm と同じ表現。compute_animal_norm_stats が回転後空間で
    center/scale を返すのに対し、こちらは回転前の生 bbox を返す（B 案のキャッシュ用）。
    返り値: {"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}
    """
    pct_lo, pct_hi = pct
    bufs = [[], [], []]
    for p in pickle_paths:
        data = load_kp3d(p)
        X = data["kp3d"][0]   # (T, n_kp, 3) 計測空間 mm（回転前）
        if n_frames and n_frames > 0:
            X = X[:n_frames]
        for ax in range(3):
            bufs[ax].append(X[:, :, ax].ravel())
    bbox = {}
    for ax, name in zip(range(3), ("x", "y", "z")):
        v = np.concatenate(bufs[ax])
        v = v[np.isfinite(v)]
        bbox[name] = [float(np.percentile(v, pct_lo)), float(np.percentile(v, pct_hi))]
    return bbox


def bbox_mm_to_center_scale(bbox_mm: dict):
    """計測空間 mm の inlier bbox から回転後空間の center/scale (mm) を導出する。

    KP3D_ROT を bbox の 8 頂点に適用して回転後の AABB を取るため、回転定義に依存しない。
    apply_global_scale が使う center/scale と同一空間（compute_animal_norm_stats と一致）。
    """
    los = np.array([bbox_mm["x"][0], bbox_mm["y"][0], bbox_mm["z"][0]])
    his = np.array([bbox_mm["x"][1], bbox_mm["y"][1], bbox_mm["z"][1]])
    corners = np.array(list(itertools.product(*zip(los, his))))   # (8, 3)
    rot = corners @ KP3D_ROT.T
    lo_r, hi_r = rot.min(axis=0), rot.max(axis=0)
    center = (lo_r + hi_r) / 2.0
    scale = (hi_r - lo_r) / 2.0
    return center, scale


def _all_pickle_paths(input_dir: Path) -> list[Path]:
    """input_dir 直下の全 session ディレクトリから pickle を解決して集める。"""
    input_dir = Path(input_dir)
    paths = []
    for d in sorted(input_dir.iterdir()):
        if d.is_dir() and (p := resolve_pickle(d)) is not None:
            paths.append(p)
    return paths


def _load_global_norm_cache() -> dict:
    """method.yaml の coordinate_transform.kp3d_global_norm を {key: bbox} で返す。"""
    with open(cfg.METHOD_YAML_PATH) as f:
        d = yaml.safe_load(f)
    return dict(d["coordinate_transform"].get("kp3d_global_norm") or {})


def _fmt_mm(x: float) -> str:
    """mm 値を読みやすく整形（0.0001mm 精度・余分な 0 を除去）。"""
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _render_global_norm_block(cache: dict) -> list[str]:
    """kp3d_global_norm ブロックを method.yaml のテキスト行 (2-space インデント) に整形。"""
    lines = ["  kp3d_global_norm:"]
    for key, bbox in cache.items():
        lines.append(f'    "{key}":')
        for ax in ("x", "y", "z"):
            lo, hi = bbox[ax]
            lines.append(f"      {ax}: [{_fmt_mm(lo)}, {_fmt_mm(hi)}]")
    return lines


def _write_global_norm_cache(key: str, bbox: dict) -> None:
    """key→bbox をキャッシュに加え、method.yaml の該当ブロックだけを書き換える。

    既存のコメント・他キーは無傷のまま、kp3d_global_norm ブロックのみテキスト置換する。
    """
    cache = _load_global_norm_cache()
    cache[key] = {ax: [float(bbox[ax][0]), float(bbox[ax][1])] for ax in ("x", "y", "z")}

    text = cfg.METHOD_YAML_PATH.read_text().splitlines()
    start = next((i for i, ln in enumerate(text)
                  if ln.startswith("  kp3d_global_norm:")), None)
    if start is None:
        raise RuntimeError("method.yaml に kp3d_global_norm セクションが見つかりません")
    end = start + 1
    while end < len(text) and text[end].startswith("   "):   # 3+ space = ブロック内
        end += 1
    new_text = text[:start] + _render_global_norm_block(cache) + text[end:]
    cfg.METHOD_YAML_PATH.write_text("\n".join(new_text) + "\n")


def get_or_compute_global_norm(pct: tuple[float, float], input_dir,
                               n_frames: int = -1, write: bool = True,
                               recompute: bool = False):
    """percentile をキーに global norm の (center, scale) を取得する。

    method.yaml の kp3d_global_norm に該当 percentile があれば読むだけ、無ければ
    全 pickle から計測空間 bbox を計算して method.yaml に追記する（write=True 時）。
    返り値: (center, scale) いずれも回転後空間 mm, shape (3,)。
    """
    key = f"{pct[0]:g}-{pct[1]:g}"
    cache = _load_global_norm_cache()
    if key in cache and not recompute:
        bbox = cache[key]
        print(f"[global norm] pct {key}: method.yaml キャッシュを使用")
    else:
        why = "強制再計算" if (recompute and key in cache) else "未計算"
        paths = _all_pickle_paths(input_dir)
        print(f"[global norm] pct {key}: {why} → {len(paths)} pickle から計算します …")
        bbox = compute_global_inlier_bbox_mm(paths, n_frames, pct)
        if write:
            _write_global_norm_cache(key, bbox)
            print(f"[global norm] pct {key}: method.yaml に追記しました")
    center, scale = bbox_mm_to_center_scale(bbox)
    print(f"[global norm] pct {key}: "
          f"center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) "
          f"scale=({scale[0]:.1f}, {scale[1]:.1f}, {scale[2]:.1f}) mm")
    return center, scale


def apply_global_scale(df: pd.DataFrame,
                       center=None, scale=None) -> pd.DataFrame:
    """
    回転済み mm DataFrame にスケールを適用し Blender m 空間にする。

    center / scale (各 shape (3,), mm): 全軸の規格化パラメータを上書きする。
    None のときはグローバル定数 KP3D_CENTER / KP3D_NORM_SCALE を使う。
    """
    df = df.copy()
    half_span = BLENDER_BBOX_SCALE / 2
    axis_map = {"_x": 0, "_y": 1, "_z": 2}
    for col in df.columns[2:]:
        for suffix, idx in axis_map.items():
            if col.endswith(suffix):
                c = center[idx] if center is not None else KP3D_CENTER[idx]
                s = scale[idx]  if scale  is not None else KP3D_NORM_SCALE[idx]
                df[col] = ((df[col] - c) / s
                           * half_span[idx] + BLENDER_CENTER[idx])
                break
    return df


def scale_and_correct(df_raw: pd.DataFrame, output_scale: float = 1.0,
                      return_peaks: bool = False,
                      center=None, scale=None,
                      legacy: bool = False,
                      fit_mode: str = "kde", max_pct: float = 10.0):
    """
    回転済み CSV にスケール → KDE affine 補正 (bZ) → output_scale を適用。

    center / scale (shape (3,), mm): apply_global_scale に渡す規格化パラメータ。
      None のときはグローバル定数を使う（legacy モード）。
    legacy: True で fit_affine に Hungarian マッチング（旧動作）を強制する。
    fit_mode:
      "kde"     床KP min(bZ) の KDE ピークのみで回帰（従来）
      "kde_max" 全KP max(bZ) の上位 max_pct% 平均を max% ピークとし、KDE ピークへ
                merge_max_peak で統合してから Hungarian 回帰（新）
    max_pct: kde_max 時の上位パーセンタイル（10/5/1、<=0 で max のみ）

    return_peaks=False -> (df_corrected, slope, intercept, status)
    return_peaks=True  -> (df_corrected, slope, intercept, status, peaks, info)
      info = {"fit_mode", "max_pct", "max_peak", "merge_label"}

    peaks は補正前（global scale 後・affine 適用前）の分布から得たアンカー (Blender m, 昇順)。
    affine フィットに使った値そのもので、これを保存すれば plot 側で pickle 再計算が不要。
    """
    df = apply_global_scale(df_raw, center=center, scale=scale)

    info = {"fit_mode": fit_mode, "max_pct": max_pct,
            "max_peak": float("nan"), "merge_label": ""}

    z_cols = floor_z_cols(df.columns)
    if z_cols:
        vals = trim_1_99(df[z_cols].min(axis=1).values)
        peaks, peak_densities = detect_peaks_kde(vals, return_density=True)
        if fit_mode == "kde_max":
            mcols = real_kp_z_cols(df.columns)
            max_peak = top_pct_peak(df[mcols].max(axis=1).values, max_pct)
            peaks, label, warn_msg = merge_max_peak(peaks, max_peak)
            info["max_peak"], info["merge_label"] = max_peak, label
            if warn_msg:
                warnings.warn(warn_msg, UserWarning, stacklevel=2)
            # 統合後の少数アンカー（密度なしの P 混在）は Hungarian でマッチ
            a, b, status = fit_affine(peaks, legacy=True)
        else:
            a, b, status = fit_affine(peaks, peak_densities=peak_densities, legacy=legacy)
    else:
        peaks = np.array([])
        a, b, status = 1.0, 0.0, "no_floor_cols"

    for col in df.columns:
        if col.endswith("_z"):
            df[col] = a * df[col] + b

    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    df[coord_cols] *= output_scale
    df = add_cam_pose_with_euler(df)
    if return_peaks:
        return df, a, b, status, peaks, info
    return df, a, b, status
