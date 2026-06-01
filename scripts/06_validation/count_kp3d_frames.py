#!/usr/bin/env python3
"""kp3dファイルのフレーム数を集計する。

3D_temporal_keypoint_data/ 以下の全セッションを走査し、
個体ごと・全体のフレーム数を表示する。

Usage:
    python scripts/06_validation/count_kp3d_frames.py [DATA_DIR]

    DATA_DIR: 3D_temporal_keypoint_data/ のパス（省略時はデフォルト使用）
"""

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).parents[2] / "3D_digital_twin_marmoset_data" / \
    "marmoset_keypoint_data" / "3D_temporal_keypoint_data"
KP3D_FILENAME = "kp3d_fxdJointLen.pickle"
# セッション名末尾の _YYYYMMDD_HHMMSS を除いた部分を個体IDとして扱う
SESSION_DATE_PATTERN = re.compile(r"_\d{8}_\d{6}$")


def load_frame_count(pickle_path: Path) -> int:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    # shape: (1, frames, keypoints, 3)
    return data["kp3d"].shape[1]


def extract_animal_id(session_name: str) -> str:
    return SESSION_DATE_PATTERN.sub("", session_name)


def main():
    parser = argparse.ArgumentParser(description="kp3dフレーム数集計")
    parser.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR,
                        help="3D_temporal_keypoint_data/ のパス")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        print(f"エラー: ディレクトリが見つかりません: {data_dir}", file=sys.stderr)
        sys.exit(1)

    sessions = sorted(p for p in data_dir.iterdir()
                      if p.is_dir() and (p / KP3D_FILENAME).exists())

    if not sessions:
        print(f"エラー: {KP3D_FILENAME} を含むセッションが見つかりません", file=sys.stderr)
        sys.exit(1)

    # 個体ID → {セッション名: フレーム数}
    animal_sessions: dict[str, dict[str, int]] = defaultdict(dict)
    missing = []

    print(f"セッション数: {len(sessions)}")
    print("集計中...\n")

    for session_dir in sessions:
        pkl = session_dir / KP3D_FILENAME
        try:
            n_frames = load_frame_count(pkl)
        except Exception as e:
            missing.append((session_dir.name, str(e)))
            continue
        animal_id = extract_animal_id(session_dir.name)
        animal_sessions[animal_id][session_dir.name] = n_frames

    # --- 個体別サマリー ---
    print("=" * 60)
    print("個体別フレーム数サマリー")
    print("=" * 60)

    total_all = 0
    for animal_id in sorted(animal_sessions):
        sessions_map = animal_sessions[animal_id]
        total = sum(sessions_map.values())
        total_all += total
        print(f"\n【{animal_id}】")
        print(f"  セッション数 : {len(sessions_map)}")
        print(f"  合計フレーム数: {total:,}")
        print(f"  平均フレーム数: {total // len(sessions_map):,} / セッション")

    print("\n" + "=" * 60)
    print(f"全個体合計フレーム数: {total_all:,}")
    print(f"全セッション数      : {sum(len(v) for v in animal_sessions.values())}")
    print("=" * 60)

    # --- 個体別詳細 ---
    print("\n詳細（個体別・セッション別）")
    print("-" * 60)
    for animal_id in sorted(animal_sessions):
        print(f"\n{animal_id}")
        for session_name, n in sorted(animal_sessions[animal_id].items()):
            print(f"  {session_name}: {n:,} フレーム")

    if missing:
        print(f"\n警告: 読み込み失敗セッション ({len(missing)}件)")
        for name, err in missing:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
