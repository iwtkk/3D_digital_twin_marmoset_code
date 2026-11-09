"""
Script that randomly selects frames per animal and saves them to JSON/CSV
"""
from __future__ import annotations

import argparse
import json
import csv
import pickle
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path setup
current_script = Path(__file__).resolve()
marmo3d_dir = current_script.parent.parent  # src/marmo3d/
src_dir = marmo3d_dir.parent                # src/
project_root = src_dir.parent               # project root

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import marmo3d.marmo_dataset as marmo_dataset


@dataclass
class FrameSelection:
    """Information about a selected frame"""
    animal: str
    session_id: str
    frame: int
    kp3d_path: str


@dataclass
class SelectionResult:
    """Overall selection result"""
    seed: int
    frames_per_animal: int
    animals: List[str]
    total_frames: int
    selections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FramesCache:
    """Cache of all frame information"""
    animals: List[str]
    frames_by_animal: Dict[str, List[Dict[str, Any]]]


def load_kp3d_frame_count(kp3d_path: Path) -> int:
    """Get the frame count from kp3d.pickle (minimum over all animal_ids)"""
    with open(kp3d_path, "rb") as f:
        data = pickle.load(f)

    if "kp3d" not in data:
        raise KeyError(f"'kp3d' key not found: {kp3d_path}")

    return min(len(x) for x in data["kp3d"])


def collect_frames(session: marmo_dataset.Session) -> List[Dict[str, Any]]:
    """
    Collect frame information from a single session

    Args:
        session: target session

    Returns:
        [{"session_id": ..., "frame": ..., "kp3d_path": ...}, ...]
    """
    kp3d_path = session.kp3d_path()
    if kp3d_path is None or not kp3d_path.exists():
        return []

    try:
        frame_count = load_kp3d_frame_count(kp3d_path)
    except Exception as e:
        print(f"Warning: failed to load {kp3d_path}: {e}")
        return []

    frames = []
    for frame_idx in range(frame_count):
        frames.append({
            "session_id": session.session_id,
            "frame": frame_idx,
            "kp3d_path": str(kp3d_path),
        })

    return frames


def collect_animal_frames(
    ds: marmo_dataset.MarmoDataset,
    animal: str,
) -> List[Dict[str, Any]]:
    """
    Collect frame information from all sessions of the given animal

    Args:
        ds: dataset
        animal: animal ID

    Returns:
        List of frame information across all sessions
    """
    sessions = list(ds.iter_sessions(valid_only=True, log_skips=False))
    animal_sessions = ds.filter(sessions, animal=animal)

    all_frames = []
    for session in animal_sessions:
        frames = collect_frames(session)
        all_frames.extend(frames)

    return all_frames


def save_frames_cache(cache: FramesCache, cache_path: Path) -> None:
    """Save the frame cache"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cache), f, ensure_ascii=False, indent=2)
    print(f"Saved frame cache: {cache_path}")


def load_frames_cache(cache_path: Path) -> Optional[FramesCache]:
    """Load the frame cache"""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FramesCache(
            animals=data["animals"],
            frames_by_animal=data["frames_by_animal"],
        )
    except Exception as e:
        print(f"Warning: failed to load cache: {e}")
        return None


def collect_all_frames(
    ds: marmo_dataset.MarmoDataset,
    animals: List[str],
    cache_path: Optional[Path] = None,
) -> FramesCache:
    """
    Collect frame information for all animals (with caching)

    Args:
        ds: dataset
        animals: list of target animals
        cache_path: cache file path (saved/loaded if given)

    Returns:
        FramesCache
    """
    # Load from cache if available
    if cache_path:
        cache = load_frames_cache(cache_path)
        if cache and set(cache.animals) == set(animals):
            print(f"Loaded from cache: {cache_path}")
            return cache

    # Collect
    frames_by_animal = {}
    for animal in animals:
        print(f"Collecting frames for animal {animal}...")
        frames = collect_animal_frames(ds, animal)
        frames_by_animal[animal] = frames
        print(f"  Available frames: {len(frames)}")

    cache = FramesCache(animals=animals, frames_by_animal=frames_by_animal)

    # Save cache
    if cache_path:
        save_frames_cache(cache, cache_path)

    return cache


def select_random_frames(
    cache: FramesCache,
    frames_per_animal: int,
    seed: int,
) -> SelectionResult:
    """
    Randomly select frames from the cache
    """
    random.seed(seed)

    result = SelectionResult(
        seed=seed,
        frames_per_animal=frames_per_animal,
        animals=cache.animals,
        total_frames=0,
    )

    for animal in cache.animals:
        all_frames = cache.frames_by_animal.get(animal, [])

        if len(all_frames) == 0:
            print(f"Warning: no frames available for {animal}")
            continue

        # Select all frames if fewer than requested are available
        n_select = min(frames_per_animal, len(all_frames))
        selected = random.sample(all_frames, n_select)

        for frame_info in selected:
            selection = FrameSelection(
                animal=animal,
                session_id=frame_info["session_id"],
                frame=frame_info["frame"],
                kp3d_path=frame_info["kp3d_path"],
            )
            result.selections.append(asdict(selection))

        print(f"{animal}: selected {n_select} frames")

    result.total_frames = len(result.selections)
    return result


def save_json(result: SelectionResult, output_path: Path) -> None:
    """Save the result to JSON"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {output_path}")


def save_csv(result: SelectionResult, output_path: Path) -> None:
    """Save the result to CSV"""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["animal", "session_id", "frame", "kp3d_path"])
        writer.writeheader()
        writer.writerows(result.selections)
    print(f"Saved CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly select frames per animal"
    )
    parser.add_argument(
        "--animals",
        nargs="+",
        default=["cj611", "cj357", "cj425"],
        help="Target animal IDs (default: cj611 cj357 cj425)",
    )
    parser.add_argument(
        "--frames-per-animal",
        type=int,
        default=100,
        help="Number of frames to select per animal (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="selected_frames.json",
        help="Output file path (default: selected_frames.json)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Frame cache file path (saved/loaded if given)",
    )
    parser.add_argument(
        "--dataset-spec",
        type=str,
        default=None,
        help="Path to dataset_spec.yaml (default: config/dataset_spec.yaml)",
    )

    args = parser.parse_args()

    # Load dataset
    if args.dataset_spec:
        spec_path = Path(args.dataset_spec)
    else:
        spec_path = project_root / "config" / "dataset_spec.yaml"

    if not spec_path.exists():
        print(f"Error: dataset_spec.yaml not found: {spec_path}")
        sys.exit(1)

    print(f"Dataset spec: {spec_path}")
    ds = marmo_dataset.MarmoDataset.from_yaml(spec_path, project_root=project_root)

    # Cache path
    cache_path = Path(args.cache) if args.cache else None

    # Collect frames (with caching)
    cache = collect_all_frames(ds, args.animals, cache_path)

    # Random selection
    result = select_random_frames(
        cache=cache,
        frames_per_animal=args.frames_per_animal,
        seed=args.seed,
    )

    print(f"\nTotal selected frames: {result.total_frames}")

    # Save
    output_path = Path(args.output)

    if args.format in ("json", "both"):
        json_path = output_path.with_suffix(".json") if args.format == "both" else output_path
        save_json(result, json_path)

    if args.format in ("csv", "both"):
        csv_path = output_path.with_suffix(".csv") if args.format == "both" else output_path
        save_csv(result, csv_path)


if __name__ == "__main__":
    main()
