#!/usr/bin/env python3

"""
Search kp3d_err pickles and report the entries with the largest reprojection error.

The script mirrors the globbing style used in plot_repr_err_distribution.py but instead of
plotting it surfaces the exact file, animal index, frame index, and joint index that produced
the top-k finite kp3d_err values. Default behaviour inspects results/3d*_det/daily*/kp3d*.pickle.
Use --skip-session (session directory substring or glob) to exclude specific recordings.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import math
import pickle
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import argcomplete
import numpy as np

DEFAULT_SKIP_SESSIONS: tuple[str, ...] = ()


@dataclass(order=True)
class ErrorEntry:
    """Container used in the heap. Entries are ordered by `value` for heapq."""

    value: float
    path: str
    animal_idx: int
    frame_idx: int
    joint_idx: int

    def as_dict(self) -> dict[str, object]:
        """Convenience for possible downstream usage."""
        return {
            "value": self.value,
            "path": self.path,
            "animal_idx": self.animal_idx,
            "frame_idx": self.frame_idx,
            "joint_idx": self.joint_idx,
        }


def iter_paths(patterns: Sequence[str]) -> Iterator[str]:
    """Yield sorted, de-duplicated paths that match any of the provided patterns."""

    seen: set[str] = set()
    for pattern in patterns:
        for match in sorted(glob.iglob(pattern)):
            if match not in seen:
                seen.add(match)
                yield match


def top_indices(arr: np.ndarray, k: int) -> Iterable[tuple[float, tuple[int, ...]]]:
    """Return up to k largest finite values in arr along with their multi-indices."""

    if k <= 0:
        return []

    flat = arr.reshape(-1)
    finite_mask = np.isfinite(flat)
    if not np.any(finite_mask):
        return []

    finite_indices = np.nonzero(finite_mask)[0]
    finite_values = flat[finite_mask]
    count = finite_values.size
    if count <= k:
        order = np.argsort(finite_values)[::-1]
    else:
        selection = np.argpartition(finite_values, -k)[-k:]
        order = selection[np.argsort(finite_values[selection])[::-1]]

    for pos in order:
        flat_index = finite_indices[pos]
        value = float(finite_values[pos])
        multi_index = np.unravel_index(flat_index, arr.shape)
        yield value, tuple(int(i) for i in multi_index)


def find_top_errors(
    paths: Sequence[str],
    key: str,
    top_k: int,
    skip_sessions: Sequence[str] | None = None,
) -> list[ErrorEntry]:
    """Scan all pickles and keep the largest finite kp3d_err entries."""

    heap: list[ErrorEntry] = []
    skip_sessions = tuple(skip_sessions or ())

    for raw_path in paths:
        path = Path(raw_path)
        if should_skip_session(path, skip_sessions):
            continue
        with path.open("rb") as f:
            data = pickle.load(f)

        if key not in data:
            raise KeyError(f"{key} not found in {path}")

        arr = np.asarray(data[key], dtype=float)
        if arr.size == 0:
            continue

        for value, indices in top_indices(arr, top_k):
            animal_idx, frame_idx, joint_idx = _normalize_indices(indices)
            entry = ErrorEntry(value=value, path=str(path), animal_idx=animal_idx, frame_idx=frame_idx, joint_idx=joint_idx)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            else:
                if value > heap[0].value:
                    heapq.heapreplace(heap, entry)

    return sorted(heap, key=lambda item: item.value, reverse=True)


def _normalize_indices(indices: tuple[int, ...]) -> tuple[int, int, int]:
    """
    kp3d_err arrays are typically shaped (n_animals, n_frames, n_joints).
    Accept anything with at least three dimensions and fall back to sensible defaults.
    """

    if len(indices) == 3:
        return indices  # type: ignore[return-value]

    # When the array has more than three dims we assume the last three map to (animal, frame, joint)
    if len(indices) > 3:
        return tuple(int(i) for i in indices[-3:])  # type: ignore[return-value]

    # If the array is smaller than 3 dims pad with zeros.
    padded = [0, 0, 0]
    for i, value in enumerate(indices):
        padded[i] = int(value)
    return tuple(padded)  # type: ignore[return-value]


def should_skip_session(path: Path, sessions: Sequence[str]) -> bool:
    """Return True if the path belongs to one of the sessions we need to skip."""

    path_str = str(path)
    for session in sessions:
        if not session:
            continue
        if any(ch in session for ch in "*?[]"):
            if fnmatch.fnmatch(path_str, session):
                return True
        elif session in path_str:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate the kp3d pickle containing the largest reprojection errors.")
    argcomplete.autocomplete(parser)
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["results/3d*_det/daily*/kp3d*.pickle"],
        help="Glob pattern(s) pointing at kp3d pickle files.",
    )
    parser.add_argument("--key", default="kp3d_err", help="Key to inspect inside each pickle (default: %(default)s).")
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Report this many highest entries. Always includes the absolute maximum.",
    )
    parser.add_argument(
        "--show-each-max",
        action="store_true",
        help="Also print the per-file maximum while scanning (useful for debugging).",
    )
    parser.add_argument(
        "--skip-session",
        action="append",
        default=[],
        metavar="SESSION",
        help="Skip pickles whose path contains SESSION (or matches the glob). "
        "May be specified multiple times.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()


    patterns = args.pattern
    top_k = max(1, args.top)

    skip_sessions = tuple(DEFAULT_SKIP_SESSIONS) + tuple(args.skip_session)

    paths = list(iter_paths(patterns))
    if not paths:
        raise FileNotFoundError(f"No files matched any of: {patterns}")

    filtered_paths: list[str] = []
    skipped_count = 0
    for raw_path in paths:
        path_obj = Path(raw_path)
        if should_skip_session(path_obj, skip_sessions):
            skipped_count += 1
            continue
        filtered_paths.append(raw_path)

    if skipped_count:
        print(f"Skipped {skipped_count} file(s) due to skip rules.")

    if not filtered_paths:
        raise FileNotFoundError(
            f"After applying skip rules no files remained for key '{args.key}'."
        )

    if args.show_each_max:
        for raw_path in filtered_paths:
            path = Path(raw_path)
            with path.open("rb") as f:
                data = pickle.load(f)

            if args.key not in data:
                raise KeyError(f"{args.key} not found in {path}")

            arr = np.asarray(data[args.key], dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                print(f"{path}: no finite values")
                continue
            max_value = float(np.max(finite))
            print(f"{path}: max={max_value:.6f}")

    results = find_top_errors(filtered_paths, args.key, top_k, skip_sessions)
    if not results:
        raise RuntimeError(f"No finite values found for key '{args.key}'.")

    print(f"Scanned {len(filtered_paths)} pickle(s). Top {len(results)} entries for key '{args.key}':\n")
    for rank, entry in enumerate(results, start=1):
        print(
            f"[{rank}] value={entry.value:.6f} | file={entry.path}\n"
            f"    animal_idx={entry.animal_idx} (0-based, {entry.animal_idx + 1}-based={entry.animal_idx + 1})\n"
            f"    frame_idx={entry.frame_idx} (0-based, {entry.frame_idx + 1}-based={entry.frame_idx + 1})\n"
            f"    joint_idx={entry.joint_idx} (0-based, {entry.joint_idx + 1}-based={entry.joint_idx + 1})\n"
        )

    top_entry = results[0]
    if not math.isfinite(top_entry.value):
        print("Warning: the reported maximum is not finite.")


if __name__ == "__main__":
    main()
