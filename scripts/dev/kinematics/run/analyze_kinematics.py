"""
anipose 3D キーポイントの運動学・信号品質解析 (run)。

動きを「信号」として特徴づける: 速度/加速度/ジャークの分布、PSD（周波数別エネルギー）、
（固定のはずの）ボーン長の安定性。全セッションを streaming で集計し、累積配列を NPZ に、
要約を JSON に保存する。可視化は plot/plot_kinematics.py。

注意: 3D (kp3d_fxdJointLen.pickle) は anipose で時間平滑化済み。PSD 高周波端は
平滑化フィルタのロールオフであり、生の検出ノイズではない。

入力 (--input-dir, 既定 outputs/3d):  {input-dir}/{session}/kp3d_fxdJointLen.pickle
出力 (--output-dir, 既定 scripts/dev/kinematics_files):
  kinematics_accum.npz       累積ヒストグラム / PSD / ボーン統計 / 代表セッション時系列
  kinematics_summary.json    パーセンタイル等の要約
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import signal

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import kinematics_common as kc  # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/kinematics_files")


class HistAccum:
    def __init__(self, bins):
        self.bins = bins
        self.counts = np.zeros((kc.N_KP, len(bins) - 1), dtype=np.int64)
        self.vmax = np.zeros(kc.N_KP)

    def add(self, vals):  # (T, N_KP)
        lo, hi = self.bins[0], self.bins[-1]
        for ki in range(kc.N_KP):
            col = vals[:, ki]
            self.vmax[ki] = max(self.vmax[ki], float(np.nanmax(col)) if col.size else 0.0)
            c, _ = np.histogram(np.clip(col, lo, hi), bins=self.bins)
            self.counts[ki] += c


class PSDAccum:
    def __init__(self, freqs):
        self.freqs = freqs
        self.sum = np.zeros((kc.N_KP, len(freqs)))
        self.n = 0

    def add(self, sig_txa):  # (T, N_KP, 3)
        if sig_txa.shape[0] < kc.NPERSEG:
            return
        _, pxx = signal.welch(sig_txa, fs=kc.FPS, nperseg=kc.NPERSEG, axis=0)
        self.sum += pxx.mean(axis=-1).T
        self.n += 1

    def mean(self):
        return self.sum / max(self.n, 1)


class BoneAccum:
    def __init__(self):
        self.n = np.zeros(kc.N_BONE)
        self.s = np.zeros(kc.N_BONE)
        self.s2 = np.zeros(kc.N_BONE)

    def add(self, blen):  # (T, N_BONE)
        self.n += blen.shape[0]
        self.s += blen.sum(axis=0)
        self.s2 += (blen ** 2).sum(axis=0)

    def stats(self):
        mean = self.s / np.maximum(self.n, 1)
        var = np.maximum(self.s2 / np.maximum(self.n, 1) - mean ** 2, 0.0)
        std = np.sqrt(var)
        cv = np.divide(std, mean, out=np.zeros_like(std), where=mean > 1e-9)
        return mean, std, cv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", type=str, default=None,
                    help="時系列プロット用の代表セッション (省略時は先頭)")
    ap.add_argument("--max-sessions", type=int, default=None, help="集計セッション数の上限 (debug)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p.name for p in args.input_dir.iterdir() if p.is_dir())
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
    ts_session = args.session or (sessions[0] if sessions else None)

    spd = HistAccum(kc.SPEED_BINS)
    acc = HistAccum(kc.ACCEL_BINS)
    jrk = HistAccum(kc.JERK_BINS)
    freqs = np.fft.rfftfreq(kc.NPERSEG, d=kc.DT)
    psd_pos, psd_vel, psd_acc = PSDAccum(freqs), PSDAccum(freqs), PSDAccum(freqs)
    bone = BoneAccum()
    joint_len_ref, ts_cache, n_used = None, None, 0

    print(f"Processing {len(sessions)} sessions … (time-series session: {ts_session})")
    for i, s in enumerate(sessions):
        pkl = args.input_dir / s / "kp3d_fxdJointLen.pickle"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        kp = d["kp3d"][0]
        if joint_len_ref is None and "joint_len" in d:
            joint_len_ref = np.asarray(d["joint_len"][0])
        if kp.shape[0] < 4:
            continue
        speed, accel, jerk, pos, vel_vec, acc_vec, bone_len = kc.session_features(kp)
        spd.add(speed); acc.add(accel); jrk.add(jerk)
        psd_pos.add(pos); psd_vel.add(vel_vec); psd_acc.add(acc_vec)
        bone.add(bone_len)
        n_used += 1
        if s == ts_session:
            ts_cache = (speed, accel, jerk, bone_len)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sessions)}] {s}  (used {n_used})")

    print(f"Sessions used: {n_used}; PSD segments per signal: {psd_vel.n}")
    b_mean, b_std, b_cv = bone.stats()
    mvel = psd_vel.mean()
    hf = freqs >= kc.HF_CUTOFF_HZ
    hf_frac = mvel[:, hf].sum(axis=1) / np.maximum(mvel.sum(axis=1), 1e-12)

    npz = {
        "speed_counts": spd.counts, "accel_counts": acc.counts, "jerk_counts": jrk.counts,
        "speed_bins": kc.SPEED_BINS, "accel_bins": kc.ACCEL_BINS, "jerk_bins": kc.JERK_BINS,
        "speed_vmax": spd.vmax, "accel_vmax": acc.vmax, "jerk_vmax": jrk.vmax,
        "psd_freqs": freqs, "psd_pos_mean": psd_pos.mean(),
        "psd_vel_mean": mvel, "psd_acc_mean": psd_acc.mean(),
        "bone_mean": b_mean, "bone_std": b_std, "bone_cv": b_cv,
        "ts_session": ts_session or "",
    }
    if ts_cache is not None:
        npz.update({"ts_speed": ts_cache[0], "ts_accel": ts_cache[1],
                    "ts_jerk": ts_cache[2], "ts_bone_len": ts_cache[3]})
    np.savez_compressed(args.output_dir / "kinematics_accum.npz", **npz)
    print(f"  saved {args.output_dir / 'kinematics_accum.npz'}")

    summary = {
        "n_sessions_used": n_used, "fps": kc.FPS, "nyquist_hz": kc.NYQUIST,
        "note": "3D keypoints are anipose-smoothed; 2D keypoints are the true raw signal.",
        "speed_percentiles_mm_s": {
            kc.KP_NAMES[ki]: {q: round(kc.hist_percentile(spd.counts[ki], kc.SPEED_BINS, q), 2)
                              for q in (50, 95, 99)} for ki in range(kc.N_KP)},
        "speed_max_mm_s": {kc.KP_NAMES[ki]: round(spd.vmax[ki], 2) for ki in range(kc.N_KP)},
        "accel_p99_mm_s2": {kc.KP_NAMES[ki]: round(kc.hist_percentile(acc.counts[ki], kc.ACCEL_BINS, 99), 2)
                            for ki in range(kc.N_KP)},
        "jerk_p99_mm_s3": {kc.KP_NAMES[ki]: round(kc.hist_percentile(jrk.counts[ki], kc.JERK_BINS, 99), 2)
                           for ki in range(kc.N_KP)},
        "bone_length": {kc.BONE_NAMES[bi]: {
            "mean_mm": round(float(b_mean[bi]), 3), "std_mm": round(float(b_std[bi]), 3),
            "cv_pct": round(float(b_cv[bi] * 100), 3)} for bi in range(kc.N_BONE)},
        "velocity_hf_energy_fraction": {kc.KP_NAMES[ki]: round(float(hf_frac[ki]), 4)
                                        for ki in range(kc.N_KP)},
        "joint_len_ref": None if joint_len_ref is None else [round(float(v), 2) for v in joint_len_ref],
    }
    with open(args.output_dir / "kinematics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved {args.output_dir / 'kinematics_summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
