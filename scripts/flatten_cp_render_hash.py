#!/usr/bin/env python3
"""cp_render の blend/csv 配下のハッシュ層を除去する。

Before: cp_render/{session_id}/{blend,csv}/{hash}/file
After:  cp_render/{session_id}/{blend,csv}/file

Usage:
    python scripts/flatten_cp_render_hash.py            # dry-run
    python scripts/flatten_cp_render_hash.py --execute  # 実際に移動
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_ROOT = Path(__file__).parents[1] / "outputs" / "validation" / "consistency" / "cp_render"


def main():
    parser = argparse.ArgumentParser(description="cp_render のハッシュ層を除去")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"cp_render ディレクトリ（デフォルト: {DEFAULT_ROOT}）")
    parser.add_argument("--execute", action="store_true",
                        help="実際に移動する（省略時は dry-run）")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"エラー: ディレクトリが見つかりません: {args.root}")
        return

    mode = "実行" if args.execute else "dry-run"
    print(f"モード: {mode}")
    print(f"対象: {args.root}")
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
                        errors.append(f"衝突: {dst} が既に存在します")
                        continue
                    print(f"移動: {src.relative_to(args.root)} → {dst.relative_to(args.root)}")
                    if args.execute:
                        shutil.move(str(src), str(dst))
                    moved += 1
                if args.execute and hash_dir.exists() and not any(hash_dir.iterdir()):
                    hash_dir.rmdir()
                    print(f"削除: {hash_dir.relative_to(args.root)}/")

    print()
    print(f"{'移動' if args.execute else 'dry-run'}: {moved}ファイル")
    if errors:
        print(f"\n警告 ({len(errors)}件):")
        for e in errors:
            print(f"  {e}")
    if not args.execute:
        print("\n実際に移動するには --execute を付けて再実行してください。")


if __name__ == "__main__":
    main()
