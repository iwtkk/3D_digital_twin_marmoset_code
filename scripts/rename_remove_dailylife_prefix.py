#!/usr/bin/env python3
"""Remove the dailylife_ prefix from directory and file names.

Target directories (session directories directly under):
  - marmoset_keypoint_data/2D_temporal_keypoint_data/
  - marmoset_keypoint_data/3D_temporal_keypoint_data/
  - marmoset_keypoint_data/video_data/

Target files (searched recursively under each session directory):
  - dailylife_*.json
  - dailylife_*.blend

Usage:
    python scripts/rename_remove_dailylife_prefix.py            # dry-run
    python scripts/rename_remove_dailylife_prefix.py --execute  # actually rename
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
            conflicts.append(f"Conflict: {dst}")
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
            conflicts.append(f"Conflict: {dst}")
        else:
            renames.append((src, dst))
    return renames, conflicts


def main():
    parser = argparse.ArgumentParser(description="Remove the dailylife_ prefix")
    parser.add_argument("--execute", action="store_true",
                        help="Actually rename (dry-run if omitted)")
    args = parser.parse_args()

    mode = "execute" if args.execute else "dry-run"
    print(f"Mode: {mode}\n")

    dir_renames, dir_conflicts = [], []
    file_renames, file_conflicts = [], []

    for target_dir in TARGET_DIRS:
        if not target_dir.exists():
            print(f"[Skip] Does not exist: {target_dir}")
            continue
        dr, dc = collect_dir_renames(target_dir)
        fr, fc = collect_file_renames(target_dir)
        dir_renames.extend(dr)
        dir_conflicts.extend(dc)
        file_renames.extend(fr)
        file_conflicts.extend(fc)
        print(f"{target_dir.name}: {len(dr)} directories, {len(fr)} files"
              + (f", {len(dc)+len(fc)} conflicts" if dc or fc else ""))

    all_conflicts = dir_conflicts + file_conflicts
    if all_conflicts:
        print("\nWarnings (skipped due to conflicts):")
        for c in all_conflicts:
            print(f"  {c}")

    print(f"\n[Directories] {len(dir_renames)}:")
    for src, dst in dir_renames:
        print(f"  {src.name}  →  {dst.name}  [{src.parent.name}]")

    print(f"\n[Files] {len(file_renames)}:")
    for src, dst in file_renames:
        print(f"  {src.name}  →  {dst.name}  [{src.parent}]")

    if not args.execute:
        print("\nRe-run with --execute to actually rename.")
        return

    print("\nRenaming...")
    # Files first (old paths become invalid once parent directories are renamed)
    for src, dst in file_renames:
        src.rename(dst)
    for src, dst in dir_renames:
        src.rename(dst)
    print(f"Done: {len(dir_renames)} directories, {len(file_renames)} files")


if __name__ == "__main__":
    main()
