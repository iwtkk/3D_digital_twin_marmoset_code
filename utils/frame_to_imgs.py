from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_ROOT = (
    PROJECT_ROOT
    / "3D_digital_twin_marmoset_data"
    / "marmoset_keypoint_data"
    / "video_data"
)
DEFAULT_CAMERA_IDS = (
    "23506226",
    "23511607",
    "23506214",
    "23506236",
    "23511614",
    "23506237",
    "23511613",
    "23506239",
)
CHUNK_FRAMES = 10000


def _resolve_stored_frame(frame: int, result_dir: Path | None, camera_id: str) -> int:
    if result_dir is None:
        return frame

    frame_num_path = result_dir / str(camera_id) / "frame_num.npy"
    if not frame_num_path.exists():
        return frame

    frame_nums = np.load(frame_num_path)
    if frame < 0 or frame >= len(frame_nums):
        raise IndexError(f"frame {frame} is out of range for {frame_num_path} ({len(frame_nums)} frames)")
    return int(frame_nums[frame])


def _find_video_frame(video_root: Path, session: str, camera_id: str, stored_frame: int, ext: str) -> tuple[Path, int] | None:
    video_folder = video_root / f"{session}.{camera_id}"
    if video_folder.exists():
        for npz_path in sorted(video_folder.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            if "frame_number" not in data:
                continue
            frame_numbers = data["frame_number"]
            if frame_numbers.size == 0:
                continue
            if int(frame_numbers[0]) <= stored_frame <= int(frame_numbers[-1]):
                matches = np.flatnonzero(frame_numbers == stored_frame)
                if matches.size == 0:
                    continue
                video_path = npz_path.with_suffix(f".{ext}")
                if video_path.exists():
                    return video_path, int(matches[0])

    chunk_index = stored_frame // CHUNK_FRAMES
    frame_in_chunk = stored_frame % CHUNK_FRAMES

    chunked = video_folder / f"{chunk_index:06d}.{ext}"
    if chunked.exists():
        return chunked, frame_in_chunk

    flat = video_root / f"{session}_{camera_id}_{chunk_index:06d}.{ext}"
    if flat.exists():
        return flat, frame_in_chunk

    return None


def resolve_video_frame(
    video_root: str | Path,
    session: str,
    camera_id: str,
    stored_frame: int,
    ext: str = "mp4",
) -> tuple[Path, int] | None:
    return _find_video_frame(Path(video_root), session, str(camera_id), int(stored_frame), ext)


def _extract_frame(video_path: Path, frame_in_chunk: int, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_expr = f"select=eq(n\\,{frame_in_chunk})"

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_expr,
            "-vsync",
            "vfr",
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARN: ffmpeg failed for {video_path}: {result.stderr.strip()}")
        return False
    return output_path.exists()


def base_frame_to_imgs(
    frame: int,
    video_root: str | Path = DEFAULT_VIDEO_ROOT,
    output_dir: str | Path = "tmp",
    *,
    session_id: str | None = None,
    camera_ids: Iterable[str] = DEFAULT_CAMERA_IDS,
    result_dir: str | Path | None = None,
    ext: str = "mp4",
) -> list[str]:
    """Extract one frame for each camera and return generated image paths.

    `frame` is the 3D-result timeline index. When `result_dir/<cam>/frame_num.npy`
    exists, it is used to map that index back to the stored video frame.
    """
    frame = int(frame)
    video_root = Path(video_root)
    output_dir = Path(output_dir)
    result_path = Path(result_dir) if result_dir is not None else None

    if session_id is None:
        if result_path is None:
            raise ValueError("session_id is required when result_dir is not provided")
        session_id = result_path.name

    image_paths: list[str] = []
    for camera_id in [str(c) for c in camera_ids]:
        stored_frame = _resolve_stored_frame(frame, result_path, camera_id)
        video_frame = _find_video_frame(video_root, session_id, camera_id, stored_frame, ext)
        if video_frame is None:
            print(f"WARN: video not found for session={session_id}, camera={camera_id}, frame={stored_frame}")
            continue
        video_path, frame_in_chunk = video_frame

        output_path = output_dir / f"{session_id}_{camera_id}_f{frame:07d}.png"
        if _extract_frame(video_path, frame_in_chunk, output_path):
            image_paths.append(str(output_path))

    return image_paths


def frame_to_imgs(
    frame: int,
    session: str = "dailylife_cj611_20230226_090000",
    input_dir: str | Path = DEFAULT_VIDEO_ROOT,
    output_dir: str | Path = "tmp",
    ext: str = "mp4",
) -> list[str]:
    return base_frame_to_imgs(
        frame,
        video_root=input_dir,
        output_dir=output_dir,
        session_id=session,
        ext=ext,
    )
