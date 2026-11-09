#!/usr/bin/env python3
"""指定ディレクトリ内のファイル名から特定の文字列を取り除く（リネーム）。

取り除く文字列:
  - work_
  - dailylife_
  - _points.csv_transformedtest

例: work_dailylife_sessionA_points.csv_transformedtest.csv → sessionA.csv

入力 (--input-dir, 必須): リネーム対象ディレクトリ
  既定は dry-run。実際にリネームするには --execute を付ける。

注意: ファイル名を入力ディレクトリ内でそのまま書き換える in-place ユーティリティのため、
--output-dir / --session は受け付けるが使用しない（インターフェース統一のため）。

使い方:
  python3 strip_marked_filenames.py --input-dir DIR              # dry-run
  python3 strip_marked_filenames.py --input-dir DIR --execute    # 実行
  python3 strip_marked_filenames.py --input-dir DIR --recursive  # 再帰
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKERS = ["work_", "dailylife_", "_points.csv_transformedtest"]


def new_name(name: str) -> str:
    for m in MARKERS:
        name = name.replace(m, "")
    return name


def collect_renames(target_dir: Path, recursive: bool):
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True, help="対象ディレクトリ")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="（未使用: in-place リネーム）")
    ap.add_argument("--session", default=None, help="（未使用）")
    ap.add_argument("--execute", action="store_true",
                    help="実際にリネームする（省略時は dry-run）")
    ap.add_argument("--recursive", action="store_true",
                    help="サブディレクトリも再帰的に対象にする")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        ap.error(f"ディレクトリが存在しません: {args.input_dir}")

    print(f"モード: {'実行' if args.execute else 'dry-run'}")
    print(f"対象ディレクトリ: {args.input_dir}")
    print(f"除去する文字列: {', '.join(MARKERS)}\n")

    renames, conflicts = collect_renames(args.input_dir, args.recursive)
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
