#!/usr/bin/env python3
"""
セッション・フレーム単位で再構成誤差が大きい箇所を調べるスクリプト。

出力:
  - セッションごとの統計 (mean / p95 / p99 / max)
  - 全体の上位 TOP_N フレーム (session名 / フレームindex / 誤差)
  - 各セッション内の上位 TOP_N_PER_SESSION フレーム
"""

import argparse
import glob
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_BASE = str(_PROJECT_ROOT / "3D_digital_twin_marmoset_data" / "marmoset_keypoint_data" / "3D_temporal_keypoint_data")


def main():
    parser = argparse.ArgumentParser(description='再構成誤差が大きいフレームを調べる')
    parser.add_argument('--base', default=_DEFAULT_BASE, help='3Dキーポイントデータのルートディレクトリ')
    parser.add_argument('--top-n', type=int, default=30, help='全体ワースト上位件数')
    parser.add_argument('--top-n-per-session', type=int, default=10, help='セッションごとのワースト上位件数')
    parser.add_argument('--threshold', type=float, default=None, help='誤差閾値(px)。指定すると超えたフレーム数を表示')
    parser.add_argument('--sessions', nargs='*', default=None, help='処理するセッション名。未指定なら全セッション')
    args = parser.parse_args()

    result_path = args.base
    TOP_N = args.top_n
    TOP_N_PER_SESSION = args.top_n_per_session
    ERR_THRESHOLD = args.threshold

    sessions = args.sessions or sorted(glob.glob('*', root_dir=result_path))

    # (session, frame_idx, mean_err_of_frame) のリスト
    all_records = []

    print("=" * 70)
    print(f"{'Session':<35} {'nFrame':>8} {'mean':>7} {'p95':>7} {'p99':>7} {'max':>7}")
    print("=" * 70)

    for session in sessions:
        picklefile = os.path.join(result_path, session, 'kp3d_fxdJointLen.pickle')
        if not os.path.isfile(picklefile):
            continue

        with open(picklefile, 'rb') as f:
            data = pickle.load(f)

        kp3d_err = np.array(data['kp3d_err'], dtype=float)

        if kp3d_err.ndim != 3 or kp3d_err.shape[0] != 1:
            print(f'[WARN] unexpected shape {kp3d_err.shape} in {session}, skip')
            continue

        arr = kp3d_err[0]  # (nFrame, 20)

        # フレームごとの平均誤差 (NaN は除外して平均)
        frame_mean_err = np.nanmean(arr, axis=1)  # (nFrame,)

        valid = frame_mean_err[np.isfinite(frame_mean_err)]
        if valid.size == 0:
            continue

        n = len(frame_mean_err)
        mean_v = float(np.nanmean(frame_mean_err))
        p95_v  = float(np.nanpercentile(frame_mean_err, 95))
        p99_v  = float(np.nanpercentile(frame_mean_err, 99))
        max_v  = float(np.nanmax(frame_mean_err))

        print(f"{session:<35} {n:>8,} {mean_v:>7.2f} {p95_v:>7.2f} {p99_v:>7.2f} {max_v:>7.2f}")

        for fi, err in enumerate(frame_mean_err):
            if np.isfinite(err):
                all_records.append((session, fi, float(err)))

    print("=" * 70)

    # ---- 全体 Top-N -------------------------------------------------- #
    all_records_arr = sorted(all_records, key=lambda x: x[2], reverse=True)

    print(f"\n[全体 ワースト {TOP_N} フレーム]")
    print(f"{'Rank':>5}  {'Session':<35} {'Frame':>8}  {'mean_err':>9}")
    print("-" * 65)
    for rank, (session, fi, err) in enumerate(all_records_arr[:TOP_N], 1):
        print(f"{rank:>5}  {session:<35} {fi:>8,}  {err:>9.3f}")

    # ---- セッションごとの Top-N -------------------------------------- #
    session_records = defaultdict(list)
    for session, fi, err in all_records:
        session_records[session].append((fi, err))

    print(f"\n[セッション別 ワースト {TOP_N_PER_SESSION} フレーム]")
    for session in sorted(session_records.keys()):
        recs = sorted(session_records[session], key=lambda x: x[1], reverse=True)
        print(f"\n  {session}")
        print(f"  {'Frame':>8}  {'mean_err':>9}")
        for fi, err in recs[:TOP_N_PER_SESSION]:
            print(f"  {fi:>8,}  {err:>9.3f}")

    # ---- 閾値超えフレーム数 ------------------------------------------ #
    if ERR_THRESHOLD is not None:
        print(f"\n[閾値 {ERR_THRESHOLD} px 超のフレーム数]")
        over_thresh = defaultdict(int)
        for session, fi, err in all_records:
            if err > ERR_THRESHOLD:
                over_thresh[session] += 1
        for session in sorted(over_thresh.keys()):
            n_session = len(session_records[session])
            cnt = over_thresh[session]
            print(f"  {session:<35} {cnt:>6,} / {n_session:>8,} ({100*cnt/n_session:.1f}%)")


if __name__ == "__main__":
    main()
