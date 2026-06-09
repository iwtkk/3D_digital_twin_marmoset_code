#!/usr/bin/env python3
"""指定ディレクトリ内のファイル名から特定の文字列を取り除く（リネーム）。

取り除く文字列:
  - work_
  - dailylife_
  - _points.csv_transformedtest

これらは work_dailylife_*_points.csv_transformedtest.csv のように
組み合わさって出現することが多い。すべて除去して元の名前に戻す。
  例: work_dailylife_sessionA_points.csv_transformedtest.csv
        →  sessionA_points.csv.csv ではなく sessionA.csv

Usage:
    python scripts/dev/strip_marked_filenames.py DIR            # dry-run
    python scripts/dev/strip_marked_filenames.py DIR --execute  # 実際にリネーム
    python scripts/dev/strip_marked_filenames.py DIR --recursive # サブディレクトリも対象
"""

import argparse
from pathlib import Path

MARKERS = ["work_", "dailylife_", "_points.csv_transformedtest"]


def new_name(name: str) -> str:
    for m in MARKERS:
        name = name.replace(m, "")
    return name


def collect_renames(target_dir: Path, recursive: bool) -> tuple[list[tuple[Path, Path]], list[str]]:
    paths = target_dir.rglob("*") if recursive else target_dir.iterdir()
    renames, conflicts = [], []
    for src in sorted(p for p in paths if p.is_file()):
        dst_name = new_name(src.name)
        if dst_name == src.name:
            continue
        dst = src.parent / dst_name
        if dst.exists():
            conflicts.append(f"衝突: {dst}")
        else:
            renames.append((src, dst))
    return renames, conflicts


def main():
    parser = argparse.ArgumentParser(
        description="ファイル名から特定文字列を除去してリネーム")
    parser.add_argument("directory", type=Path, help="対象ディレクトリ")
    parser.add_argument("--execute", action="store_true",
                        help="実際にリネームする（省略時は dry-run）")
    parser.add_argument("--recursive", action="store_true",
                        help="サブディレクトリも再帰的に対象にする")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"ディレクトリが存在しません: {args.directory}")

    mode = "実行" if args.execute else "dry-run"
    print(f"モード: {mode}")
    print(f"対象ディレクトリ: {args.directory}")
    print(f"除去する文字列: {', '.join(MARKERS)}\n")

    renames, conflicts = collect_renames(args.directory, args.recursive)

    if conflicts:
        print("警告（同名ファイルが既に存在するため対象外）:")
        for c in conflicts:
            print(f"  {c}")
        print()

    print(f"[リネーム対象] {len(renames)}件:")
    for src, dst in renames:
        print(f"  {src.name}  →  {dst.name}")

    if not args.execute:
        print("\n実際にリネームするには --execute を付けて再実行してください。")
        return

    print("\nリネーム実行中...")
    for src, dst in renames:
        src.rename(dst)
    print(f"完了: {len(renames)}件リネームしました。")


if __name__ == "__main__":
    main()
