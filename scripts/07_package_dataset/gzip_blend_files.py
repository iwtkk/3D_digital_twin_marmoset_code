"""
.blend を gzip 圧縮して別ディレクトリへ書き出す（公開アップロード用）。

Blender は gzip 圧縮された .blend をそのまま開ける（拡張子は .blend のまま）。
Blender で開き直して保存しないので、ファイル形式のバージョンは元のまま変わらない。

使い方:
  .venv/bin/python scripts/07_package_dataset/gzip_blend_files.py \
      --src-dir <dataset_work>/3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_simplified_marmoset \
      --dst-dir <dataset_upload>/3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_simplified_marmoset \
      --workers 32 --verify

入出力台帳: scripts/07_package_dataset/io.md#gzip_blend_files
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CHUNK = 16 << 20


def _md5_plain(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _md5_gunzip(path: Path) -> str:
    h = hashlib.md5()
    with gzip.open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def gzip_one(src: Path, dst: Path, level: int, verify: bool) -> tuple[str, int, int, str]:
    """src を gzip して dst へ。既に dst があればスキップ（検証のみ）。"""
    if not dst.exists():
        tmp = dst.with_name(dst.name + ".tmp")
        with open(src, "rb") as fi, gzip.GzipFile(tmp, "wb", compresslevel=level, mtime=0) as fo:
            shutil.copyfileobj(fi, fo, CHUNK)
        os.replace(tmp, dst)
    status = "ok"
    if verify and _md5_plain(src) != _md5_gunzip(dst):
        status = "MISMATCH"
    return src.name, src.stat().st_size, dst.stat().st_size, status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True, type=Path)
    ap.add_argument("--dst-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--level", type=int, default=6, help="gzip 圧縮レベル (1-9)")
    ap.add_argument("--verify", action="store_true", help="展開後の md5 が元と一致するか確認")
    args = ap.parse_args()

    srcs = sorted(args.src_dir.glob("*.blend"))
    if not srcs:
        print(f"[ERR] .blend がありません: {args.src_dir}", file=sys.stderr)
        return 1
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    n_bad = 0
    tot_in = tot_out = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(gzip_one, s, args.dst_dir / s.name, args.level, args.verify) for s in srcs]
        for i, fu in enumerate(as_completed(futs), 1):
            name, n_in, n_out, status = fu.result()
            tot_in += n_in
            tot_out += n_out
            if status != "ok":
                n_bad += 1
                print(f"[{status}] {name}", file=sys.stderr)
            if i % 20 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {tot_in/1e9:.2f} GB -> {tot_out/1e9:.2f} GB", flush=True)
    print(f"done: {len(srcs)} files, {tot_in/1e9:.2f} GB -> {tot_out/1e9:.2f} GB, mismatch={n_bad}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
