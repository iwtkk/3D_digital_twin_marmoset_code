#!/usr/bin/env python3
"""dailylife_ プレフィックスをディレクトリ名・ファイル名から除去する。

対象ディレクトリ（直下のセッションディレクトリ）:
  - marmoset_keypoint_data/2D_temporal_keypoint_data/
  - marmoset_keypoint_data/3D_temporal_keypoint_data/
  - marmoset_keypoint_data/video_data/

対象ファイル（各セッションディレクトリ以下を再帰検索）:
  - dailylife_*.json
  - dailylife_*.blend

Usage:
    python scripts/rename_remove_dailylife_prefix.py            # dry-run
    python scripts/rename_remove_dailylife_prefix.py --execute  # 実際にリネーム
"""

import argparse
from pathlib import Path

BASE = Path(__file__).parents[1] / "3D_digital_twin_marmoset_data"
DATA_ROOT = BASE / "marmoset_keypoint_data"
TWIN_ROOT = BASE / "3D_digital_twin_data"
TARGET_DIRS = [
    DATA_ROOT / "2D_temporal_keypoint_data",
    DATA_ROOT / "3D_temporal_keypoint_data",
    DATA_ROOT / "video_data",
    TWIN_ROOT / "rendered_enviromental_images",
]
PREFIX = "dailylife_"
FILE_SUFFIXES = {".json", ".blend"}


def collect_dir_renames(target_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    renames, conflicts = [], []
    for src in sorted(target_dir.iterdir()):
        if not src.name.startswith(PREFIX):
            continue
        dst = target_dir / src.name[len(PREFIX):]
        if dst.exists():
            conflicts.append(f"衝突: {dst}")
        else:
            renames.append((src, dst))
    return renames, conflicts


def collect_file_renames(target_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    renames, conflicts = [], []
    for src in sorted(target_dir.rglob(f"{PREFIX}*")):
        if not src.is_file() or src.suffix not in FILE_SUFFIXES:
            continue
        dst = src.parent / src.name[len(PREFIX):]
        if dst.exists():
            conflicts.append(f"衝突: {dst}")
        else:
            renames.append((src, dst))
    return renames, conflicts


def main():
    parser = argparse.ArgumentParser(description="dailylife_ プレフィックスを除去")
    parser.add_argument("--execute", action="store_true",
                        help="実際にリネームする（省略時は dry-run）")
    args = parser.parse_args()

    mode = "実行" if args.execute else "dry-run"
    print(f"モード: {mode}\n")

    dir_renames, dir_conflicts = [], []
    file_renames, file_conflicts = [], []

    for target_dir in TARGET_DIRS:
        if not target_dir.exists():
            print(f"[スキップ] 存在しない: {target_dir}")
            continue
        dr, dc = collect_dir_renames(target_dir)
        fr, fc = collect_file_renames(target_dir)
        dir_renames.extend(dr)
        dir_conflicts.extend(dc)
        file_renames.extend(fr)
        file_conflicts.extend(fc)
        print(f"{target_dir.name}: ディレクトリ {len(dr)}件、ファイル {len(fr)}件"
              + (f"、衝突 {len(dc)+len(fc)}件" if dc or fc else ""))

    all_conflicts = dir_conflicts + file_conflicts
    if all_conflicts:
        print("\n警告（衝突のため対象外）:")
        for c in all_conflicts:
            print(f"  {c}")

    print(f"\n[ディレクトリ] {len(dir_renames)}件:")
    for src, dst in dir_renames:
        print(f"  {src.name}  →  {dst.name}  [{src.parent.name}]")

    print(f"\n[ファイル] {len(file_renames)}件:")
    for src, dst in file_renames:
        print(f"  {src.name}  →  {dst.name}  [{src.parent}]")

    if not args.execute:
        print("\n実際にリネームするには --execute を付けて再実行してください。")
        return

    print("\nリネーム実行中...")
    # ファイルを先に（親ディレクトリがリネームされると旧パスが無効になるため）
    for src, dst in file_renames:
        src.rename(dst)
    for src, dst in dir_renames:
        src.rename(dst)
    print(f"完了: ディレクトリ {len(dir_renames)}件、ファイル {len(file_renames)}件")


if __name__ == "__main__":
    main()
