#!/usr/bin/env python3
"""Count the number of frames in kp3d files.

Scans all sessions under 3D_temporal_keypoint_data/ and
prints frame counts per animal and in total.

Usage:
    python scripts/06_validation/count_kp3d_frames.py [DATA_DIR]

    DATA_DIR: path to 3D_temporal_keypoint_data/ (uses the default if omitted)
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
# Treat the session name without the trailing _YYYYMMDD_HHMMSS as the animal ID
SESSION_DATE_PATTERN = re.compile(r"_\d{8}_\d{6}$")


def load_frame_count(pickle_path: Path) -> int:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    # shape: (1, frames, keypoints, 3)
    return data["kp3d"].shape[1]


def extract_animal_id(session_name: str) -> str:
    return SESSION_DATE_PATTERN.sub("", session_name)


def main():
    parser = argparse.ArgumentParser(description="Count kp3d frames")
    parser.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to 3D_temporal_keypoint_data/")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        print(f"Error: directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    sessions = sorted(p for p in data_dir.iterdir()
                      if p.is_dir() and (p / KP3D_FILENAME).exists())

    if not sessions:
        print(f"Error: no sessions containing {KP3D_FILENAME} found", file=sys.stderr)
        sys.exit(1)

    # animal ID -> {session name: frame count}
    animal_sessions: dict[str, dict[str, int]] = defaultdict(dict)
    missing = []

    print(f"Sessions: {len(sessions)}")
    print("Counting...\n")

    for session_dir in sessions:
        pkl = session_dir / KP3D_FILENAME
        try:
            n_frames = load_frame_count(pkl)
        except Exception as e:
            missing.append((session_dir.name, str(e)))
            continue
        animal_id = extract_animal_id(session_dir.name)
        animal_sessions[animal_id][session_dir.name] = n_frames

    # --- Per-animal summary ---
    print("=" * 60)
    print("Frame count summary per animal")
    print("=" * 60)

    total_all = 0
    for animal_id in sorted(animal_sessions):
        sessions_map = animal_sessions[animal_id]
        total = sum(sessions_map.values())
        total_all += total
        print(f"\n【{animal_id}】")
        print(f"  Sessions     : {len(sessions_map)}")
        print(f"  Total frames : {total:,}")
        print(f"  Mean frames  : {total // len(sessions_map):,} / session")

    print("\n" + "=" * 60)
    print(f"Total frames (all animals): {total_all:,}")
    print(f"Total sessions            : {sum(len(v) for v in animal_sessions.values())}")
    print("=" * 60)

    # --- Per-animal details ---
    print("\nDetails (per animal, per session)")
    print("-" * 60)
    for animal_id in sorted(animal_sessions):
        print(f"\n{animal_id}")
        for session_name, n in sorted(animal_sessions[animal_id].items()):
            print(f"  {session_name}: {n:,} frames")

    if missing:
        print(f"\nWarning: sessions that failed to load ({len(missing)})")
        for name, err in missing:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
