"""
kinematics 解析の共有部品（run/plot 共通）。

定数・ボーン定義・特徴量抽出・ログ binned ヒストグラムからのパーセンタイル計算を
提供する。CLI は持たない。
"""
from __future__ import annotations

import numpy as np

FPS = 24.0
DT = 1.0 / FPS
NYQUIST = FPS / 2.0
HF_CUTOFF_HZ = 6.0          # "high frequency" band for jitter energy fraction

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
N_KP = 20

NEIGHBORS = {
    0: [1, 2, 3, 4], 1: [0, 2], 2: [0, 1], 3: [0, 5], 4: [0, 6],
    5: [3, 6, 7, 11, 17], 6: [4, 5, 8, 12, 17],
    7: [5, 9], 8: [6, 10], 9: [7], 10: [8],
    11: [5, 12, 13, 17], 12: [6, 11, 14, 17],
    13: [11, 15], 14: [12, 16], 15: [13], 16: [14],
    17: [5, 6, 11, 12, 18], 18: [17, 19], 19: [18],
}
BONES = sorted({tuple(sorted((a, b))) for a, nbrs in NEIGHBORS.items() for b in nbrs})
BONE_NAMES = [f"{KP_NAMES[a]}-{KP_NAMES[b]}" for a, b in BONES]
N_BONE = len(BONES)

# log-spaced histogram bins (mm/s, mm/s^2, mm/s^3)
SPEED_BINS = np.logspace(-1.0, 4.5, 121)
ACCEL_BINS = np.logspace(0.0, 6.0, 121)
JERK_BINS = np.logspace(0.0, 8.0, 121)

NPERSEG = 1024


def session_features(kp: np.ndarray):
    """kp: (T, 20, 3) mm -> (speed, accel, jerk, pos, vel_vec, acc_vec, bone_len)。"""
    vel_vec = np.diff(kp, axis=0) / DT
    acc_vec = np.diff(vel_vec, axis=0) / DT
    jerk_vec = np.diff(acc_vec, axis=0) / DT
    speed = np.linalg.norm(vel_vec, axis=-1)
    accel = np.linalg.norm(acc_vec, axis=-1)
    jerk = np.linalg.norm(jerk_vec, axis=-1)
    bone_len = np.stack(
        [np.linalg.norm(kp[:, a, :] - kp[:, b, :], axis=-1) for a, b in BONES], axis=-1)
    return speed, accel, jerk, kp, vel_vec, acc_vec, bone_len


def hist_percentile(counts: np.ndarray, bins: np.ndarray, q: float) -> float:
    """ログ binned ヒストグラム(counts)から幾何中心線形補間でパーセンタイルを推定。"""
    tot = counts.sum()
    if tot == 0:
        return float("nan")
    cdf = np.cumsum(counts) / tot
    centers = np.sqrt(bins[:-1] * bins[1:])
    return float(np.interp(q / 100.0, cdf, centers))
