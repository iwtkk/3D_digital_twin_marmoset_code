"""
Extract image frames from viddata using selected_frames.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Path setup (consistency/ → 06_validation/ → scripts/ → project root)
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import marmo3d.marmo_dataset as marmo_dataset
from marmo3d.validation.utils import (
    load_valid_config,
    expand_path_template,
    discover_camera_ids,
    get_project_root,
)
from utils.frame_to_imgs import resolve_video_frame

# Project root
project_root = get_project_root()


def load_selection(selection_path: Path) -> Dict:
    """Load the selection result JSON"""
    with open(selection_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_session(selections: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group frames by session ID

    Returns:
        {session_id: [{selection...}, ...], ...}
    """
    grouped = defaultdict(list)
    for sel in selections:
        session_id = sel["session_id"]
        grouped[session_id].append(sel)

    return {session_id: sorted(items, key=lambda x: int(x["frame"])) for session_id, items in grouped.items()}


def extract_frame(
    mp4_path: Path,
    frame_in_chunk: int,
    output_path: Path,
) -> bool:
    """
    Extract the specified frame from an mp4 and save it as PNG

    Args:
        mp4_path: Input mp4 file
        frame_in_chunk: Frame index within the mp4 chunk (0-based)
        output_path: Output PNG file path

    Returns:
        True on success
    """
    if not mp4_path.exists():
        print(f"  WARN: mp4 not found: {mp4_path}")
        return False

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract frame with ffmpeg
    filter_expr = f"select=eq(n\\,{frame_in_chunk})"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(mp4_path),
                "-vf",
                filter_expr,
                "-vsync",
                "vfr",
                "-frames:v",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: ffmpeg failed: {e.stderr.decode()}")
        return False


def extract_session_frames(
    session_id: str,
    selections: List[Dict],
    video_root: Path,
    output_root: Path,
    camera_ids: List[str],
) -> None:
    """
    Extract the specified frames from all cameras of one session

    Args:
        session_id: Session ID
        selections: List of selection rows from selected_frames.json
        video_root: viddata directory
        output_root: Output root directory
        camera_ids: List of camera IDs to extract
    """
    print(f"\n{session_id}:")
    print(f"  Frames: {len(selections)}")
    print(f"  Cameras: {len(camera_ids)}")

    total_extracted = 0
    total_skipped = 0
    total_failed = 0

    for camera_id in camera_ids:
        for selection in selections:
            frame_number = int(selection["frame"])
            result_dir = Path(selection["kp3d_path"]).parent if selection.get("kp3d_path") else None
            stored_frame = frame_number
            if result_dir is not None:
                frame_num_path = result_dir / str(camera_id) / "frame_num.npy"
                if frame_num_path.exists():
                    frame_nums = np.load(frame_num_path)
                    if frame_number >= len(frame_nums):
                        print(f"  WARN: frame out of range: {frame_num_path} frame={frame_number}")
                        total_failed += 1
                        continue
                    stored_frame = int(frame_nums[frame_number])

            # Output path
            output_path = (
                output_root
                / session_id
                / f"Camera.{camera_id}"
                / f"{frame_number:07d}.png"
            )

            # Skip if it already exists
            if output_path.exists():
                total_skipped += 1
                continue

            # Get mp4 path and frame within the chunk
            video_frame = resolve_video_frame(video_root, session_id, camera_id, stored_frame)
            if video_frame is None:
                print(f"  WARN: mp4 not found: session={session_id} camera={camera_id} stored_frame={stored_frame}")
                total_failed += 1
                continue
            mp4_path, frame_in_chunk = video_frame

            # Extract frame
            success = extract_frame(mp4_path, frame_in_chunk, output_path)

            if success:
                total_extracted += 1
            else:
                total_failed += 1

    print(f"  Extracted: {total_extracted}")
    if total_skipped > 0:
        print(f"  Skipped (existing): {total_skipped}")
    if total_failed > 0:
        print(f"  Failed: {total_failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract image frames from viddata using selected_frames.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Selection result JSON file path (selected_frames.json)",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="Path to consistency.yaml (default: config/consistency.yaml)",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help="viddata directory path (overrides consistency.yaml)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory (overrides consistency.yaml)",
    )
    parser.add_argument(
        "--camera-ids",
        nargs="+",
        default=None,
        help="Camera IDs to extract (read from dataset_spec if omitted)",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        help="Maximum number of sessions to process (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args()

    # Load consistency.yaml
    if args.valid_config:
        valid_config_path = Path(args.valid_config)
    else:
        valid_config_path = project_root / "config" / "consistency.yaml"

    if not valid_config_path.exists():
        print(f"Error: consistency.yaml not found: {valid_config_path}")
        sys.exit(1)

    valid_config = load_valid_config(valid_config_path)
    validation_config = valid_config.get("validation", {})

    # Get the dataset_spec.yaml path
    dataset_spec_template = validation_config.get("dataset_spec", "{PROJECT_ROOT}/config/dataset_spec.yaml")
    spec_path = expand_path_template(dataset_spec_template, project_root)

    if not spec_path.exists():
        print(f"Error: dataset_spec.yaml not found: {spec_path}")
        sys.exit(1)

    dataset_info = marmo_dataset.MarmoDataset.from_yaml(
        spec_path, project_root=project_root
    )

    # Get camera IDs
    if args.camera_ids:
        camera_ids = args.camera_ids
    else:
        camera_ids = discover_camera_ids(dataset_info)

    # Load selection results
    selection_path = Path(args.input)
    if not selection_path.exists():
        print(f"Error: input file not found: {selection_path}")
        sys.exit(1)

    selection_data = load_selection(selection_path)
    print(f"Loaded selection results: {selection_path}")
    print(f"  Seed: {selection_data['seed']}")
    print(f"  Total frames: {selection_data['total_frames']}")

    # Group by session
    grouped = group_by_session(selection_data["selections"])
    print(f"  Sessions: {len(grouped)}")

    # Path setup (command-line args > consistency.yaml > dataset_info)
    if args.video_root:
        video_root = Path(args.video_root)
    else:
        # Get video_root from dataset_info
        video_root = dataset_info.video_root

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        # Get from consistency.yaml
        output_root_template = validation_config.get("frame_extraction", {}).get(
            "output_root", "{PROJECT_ROOT}/outputs/validation/frames"
        )
        output_root = expand_path_template(output_root_template, project_root)

    if not video_root.exists():
        print(f"Error: viddata directory not found: {video_root}")
        sys.exit(1)

    print(f"\nStarting extraction")
    print(f"  Input: {video_root}")
    print(f"  Output: {output_root}")
    print(f"  Cameras: {len(camera_ids)}")
    print(f"  Camera IDs: {', '.join(camera_ids)}")

    # Extract per session
    sessions_to_process = list(sorted(grouped.items()))
    if args.limit_sessions:
        sessions_to_process = sessions_to_process[:args.limit_sessions]
        print(f"  Limit: processing only the first {args.limit_sessions} sessions")

    for i, (session_id, selections) in enumerate(sessions_to_process, 1):
        if args.verbose:
            print(f"\n[{i}/{len(sessions_to_process)}] Processing: {session_id}")
        extract_session_frames(
            session_id=session_id,
            selections=selections,
            video_root=video_root,
            output_root=output_root,
            camera_ids=camera_ids,
        )

    print("\nDone")


if __name__ == "__main__":
    main()
