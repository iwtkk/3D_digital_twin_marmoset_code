#!/usr/bin/env python3
"""Remove the hash directory level under cp_render blend/csv.

Before: cp_render/{session_id}/{blend,csv}/{hash}/file
After:  cp_render/{session_id}/{blend,csv}/file

Usage:
    python scripts/flatten_cp_render_hash.py            # dry-run
    python scripts/flatten_cp_render_hash.py --execute  # actually move files
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_ROOT = Path(__file__).parents[1] / "outputs" / "validation" / "consistency" / "cp_render"


def main():
    parser = argparse.ArgumentParser(description="Remove the hash level from cp_render")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"cp_render directory (default: {DEFAULT_ROOT})")
    parser.add_argument("--execute", action="store_true",
                        help="Actually move files (dry-run if omitted)")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: directory not found: {args.root}")
        return

    mode = "execute" if args.execute else "dry-run"
    print(f"Mode: {mode}")
    print(f"Target: {args.root}")
    print()

    moved = 0
    errors = []

    for session in sorted(args.root.iterdir()):
        if not session.is_dir():
            continue
        for subdir_name in ["blend", "csv"]:
            subdir = session / subdir_name
            if not subdir.exists():
                continue
            hash_dirs = [d for d in subdir.iterdir() if d.is_dir()]
            for hash_dir in hash_dirs:
                for src in sorted(hash_dir.iterdir()):
                    dst = subdir / src.name
                    if dst.exists():
                        errors.append(f"Conflict: {dst} already exists")
                        continue
                    print(f"Move: {src.relative_to(args.root)} → {dst.relative_to(args.root)}")
                    if args.execute:
                        shutil.move(str(src), str(dst))
                    moved += 1
                if args.execute and hash_dir.exists() and not any(hash_dir.iterdir()):
                    hash_dir.rmdir()
                    print(f"Remove: {hash_dir.relative_to(args.root)}/")

    print()
    print(f"{'Moved' if args.execute else 'dry-run'}: {moved} files")
    if errors:
        print(f"\nWarnings ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    if not args.execute:
        print("\nRe-run with --execute to actually move files.")


if __name__ == "__main__":
    main()
