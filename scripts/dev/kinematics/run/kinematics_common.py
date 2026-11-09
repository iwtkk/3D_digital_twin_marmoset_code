"""
kinematics 解析の共有部品（run/plot 共通）。

定数・ボーン定義・特徴量抽出・ログ binned ヒストグラムからのパーセンタイル計算を
提供する。CLI は持たない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/dev
import _config as cfg  # noqa: E402

FPS = cfg.FPS_3D
DT = 1.0 / FPS
NYQUIST = FPS / 2.0
HF_CUTOFF_HZ = 6.0          # "high frequency" band for jitter energy fraction (config 外)

KP_NAMES = cfg.KP_NAMES
N_KP = cfg.N_KP

NEIGHBORS = cfg.NEIGHBORS
BONES = cfg.BONES
BONE_NAMES = cfg.BONE_NAMES
N_BONE = cfg.N_BONE

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
