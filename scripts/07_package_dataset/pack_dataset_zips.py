"""
データセットのフォルダを、1ファイルの上限サイズ未満の zip 群に分割して書き出す（公開アップロード用）。

サブコマンド:
  pack   1つのフォルダ（component）を {prefix}_part{NNN}.zip 群にする。
         zip 内のパスはデータセットルートからの相対パスなので、全 zip を同じ場所で展開すると
         元のフォルダ構成に戻る。メンバーへの割り振りは元ファイルサイズ（=圧縮後サイズの上限）で
         事前に決めるので、圧縮率に関係なく上限を超えない。割り振り表を {prefix}_manifest.csv に出す。
  check  アップロード用ディレクトリ全体を検査する（ファイル数・各ファイルサイズ・manifest と元データの一致、
         必要なら zip の CRC）。
  wrap   Wrap every zip of an upload directory in an outer zip of the same name (stored, one member)
         for Harvard Dataverse, which unzips uploaded zips once. Other files are hard-linked.
         Outer zips are verified (single member, same name, same size and CRC as the inner zip).

使い方:
  .venv/bin/python scripts/07_package_dataset/pack_dataset_zips.py pack \
      --dataset-root <dataset_work> --component marmoset_keypoint_data/video_data \
      --prefix video_data --out-dir <dataset_upload>/marmoset_keypoint_data --workers 16
  .venv/bin/python scripts/07_package_dataset/pack_dataset_zips.py check \
      --upload-dir <dataset_upload> --dataset-root <dataset_work> --max-files 1000
  .venv/bin/python scripts/07_package_dataset/pack_dataset_zips.py wrap \
      --upload-dir <dataset_upload_npz> --out-dir <dataset_upload_dataverse> --workers 8

入出力台帳: scripts/07_package_dataset/io.md#pack_dataset_zips
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 既に圧縮済みの形式は無圧縮で格納する（再圧縮しても縮まず遅いだけ）
STORED_EXT = {".mp4", ".h5", ".png", ".gz", ".pth", ".npz", ".zip"}
# zip のローカルヘッダ・セントラルディレクトリのエントリあたりの上限見積もり（+パス長×2）
ENTRY_OVERHEAD = 128
EOCD = 22
MANIFEST_FIELDS = ["archive", "component", "member", "size_bytes", "session"]


def _session_of(rel_in_component: str) -> str:
    """component 内の相対パスの先頭要素から session 名を取り出す（例: cj357_..._090000.23506214 -> cj357_..._090000）。"""
    head = rel_in_component.split("/", 1)[0]
    for suf in (".csv", ".blend", ".json"):
        if head.endswith(suf):
            head = head[: -len(suf)]
    return head.split(".", 1)[0]


def _under(rel: str, prefixes: list[str]) -> bool:
    """True if rel lies under one of the folder prefixes (each ending with '/')."""
    return any(rel.startswith(p) for p in prefixes)


def plan_bins(files: list[tuple[str, int]], max_bytes: int) -> list[list[tuple[str, int]]]:
    """(arcname, size) を順番どおりに詰め、各 bin の見積もり zip サイズが max_bytes 以下になるよう分ける。"""
    bins, cur, est = [], [], EOCD
    for arc, size in files:
        add = size + ENTRY_OVERHEAD + 2 * len(arc.encode())
        if add + EOCD > max_bytes:
            raise ValueError(f"単体で上限を超えるファイル: {arc} ({size} bytes)")
        if cur and est + add > max_bytes:
            bins.append(cur)
            cur, est = [], EOCD
        cur.append((arc, size))
        est += add
    if cur:
        bins.append(cur)
    return bins


def write_zip(root: str, zip_path: str, members: list[tuple[str, int]], level: int, limit: int,
              store: bool = False) -> tuple[str, int]:
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for arc, _ in members:
            ct = (zipfile.ZIP_STORED if store or Path(arc).suffix.lower() in STORED_EXT
                  else zipfile.ZIP_DEFLATED)
            zf.write(os.path.join(root, arc), arc, compress_type=ct,
                     compresslevel=level if ct == zipfile.ZIP_DEFLATED else None)
    size = os.path.getsize(tmp)
    if size >= limit:
        raise RuntimeError(f"上限超過: {zip_path} {size} >= {limit}")
    os.replace(tmp, zip_path)
    return zip_path, size


def cmd_pack(args) -> int:
    root = args.dataset_root.resolve()
    if args.files:
        # individual files: the manifest's component column holds the file's own path (so check can compare it)
        missing = [f for f in args.files if not (root / f).is_file()]
        if missing:
            print(f"[ERR] files not found: {missing}", file=sys.stderr)
            return 1
        files = sorted((f, (root / f).stat().st_size) for f in args.files)
        comp_of = {f: f for f, _ in files}
        label = f"{len(files)} files"
    else:
        comp = root / args.component
        excl = [e.rstrip("/") + "/" for e in (args.exclude or [])]
        files = sorted(
            (str(p.relative_to(root)), p.stat().st_size) for p in comp.rglob("*")
            if p.is_file() and not _under(str(p.relative_to(root)), excl)
        )
        if not files:
            print(f"[ERR] ファイルがありません: {comp}", file=sys.stderr)
            return 1
        comp_of = {f: args.component for f, _ in files}
        label = args.component
    bins = plan_bins(files, args.max_bytes - args.margin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(bins))))
    names = [f"{args.prefix}_part{i:0{width}d}.zip" for i in range(1, len(bins) + 1)]

    # manifest は割り振りから決まるので先に書く
    man = args.out_dir / f"{args.prefix}_manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_FIELDS)
        for name, b in zip(names, bins):
            for arc, size in b:
                c = comp_of[arc]
                sess = "" if c == arc else _session_of(str(Path(arc).relative_to(c)))
                w.writerow([name, c, arc, size, sess])
    print(f"{label}: {len(files)} files -> {len(bins)} zip  (manifest {man.name})", flush=True)

    todo = [(n, b) for n, b in zip(names, bins) if not (args.out_dir / n).exists()]
    total = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(write_zip, str(root), str(args.out_dir / n), b, args.level, args.max_bytes, args.store)
                for n, b in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            _, size = fu.result()
            total += size
            if i % 10 == 0 or i == len(futs):
                print(f"  [{i}/{len(futs)}] written {total/1e9:.2f} GB", flush=True)
    return 0


def cmd_check(args) -> int:
    up = args.upload_dir
    files = [p for p in up.rglob("*") if p.is_file()]
    if not files:
        print(f"[NG] ファイルがありません: {up}")
        return 1
    bad = 0
    leftovers = [p for p in files if p.name.endswith(".tmp")]
    for p in leftovers:
        print(f"[NG] 書きかけのファイル: {p}")
        bad += 1
    over = [p for p in files if p.stat().st_size >= args.max_bytes]
    for p in over:
        print(f"[NG] サイズ上限以上: {p} {p.stat().st_size}")
        bad += 1
    total = sum(p.stat().st_size for p in files)
    print(f"files={len(files)} (max {args.max_files})  total={total/1e9:.2f} GB  "
          f"largest={max(p.stat().st_size for p in files)/1e9:.3f} GB (limit {args.max_bytes/1e9:.3f})")
    if len(files) > args.max_files:
        print(f"[NG] ファイル数が上限超過: {len(files)} > {args.max_files}")
        bad += 1

    # manifest と元データの一致（全メンバーが1回ずつ、サイズ一致）と、zip 中身との一致
    roots = [r.resolve() for r in args.dataset_root] if args.dataset_root else []
    # components of every manifest: a component nested in another one (packed with pack --exclude) is left out
    # when the outer component is compared with the source
    all_comps = set()
    for man in up.rglob("*_manifest.csv"):
        all_comps |= {r["component"] for r in csv.DictReader(open(man))}
    for man in sorted(up.rglob("*_manifest.csv")):
        rows = list(csv.DictReader(open(man)))
        by_zip: dict[str, list[dict]] = {}
        for r in rows:
            by_zip.setdefault(r["archive"], []).append(r)
        for zname, rs in by_zip.items():
            zp = man.parent / zname
            if not zp.exists():
                print(f"[NG] zip がありません: {zp}")
                bad += 1
                continue
            with zipfile.ZipFile(zp) as zf:
                info = {i.filename: i.file_size for i in zf.infolist()}
                if info != {r["member"]: int(r["size_bytes"]) for r in rs}:
                    print(f"[NG] manifest と zip の中身が不一致: {zp}")
                    bad += 1
                if args.crc and zf.testzip() is not None:
                    print(f"[NG] CRC エラー: {zp}")
                    bad += 1
        if roots:
            # the source must match one of the roots (gzipped .blend files live in the staging directory)
            mani = {r["member"]: int(r["size_bytes"]) for r in rows}
            comps = sorted({r["component"] for r in rows})
            hit = None
            for root in roots:
                src = {}
                for c in comps:
                    cp = root / c
                    if cp.is_file():
                        src[c] = cp.stat().st_size
                    else:
                        nested = [o + "/" for o in all_comps if o != c and o.startswith(c + "/")]
                        src.update({str(p.relative_to(root)): p.stat().st_size for p in cp.rglob("*")
                                    if p.is_file() and not _under(str(p.relative_to(root)), nested)})
                if src == mani:
                    hit = root
                    break
            if hit is None:
                print(f"[NG] manifest differs from the source: {man.name} (manifest={len(mani)}, roots={[str(r) for r in roots]})")
                bad += 1
            else:
                print(f"[OK] {man.name}: {len(rows)} members, {len(by_zip)} zip, matches the source ({hit})")
    print("RESULT:", "OK" if bad == 0 else f"NG ({bad})")
    return 1 if bad else 0


def wrap_one(src: str, dst: str, limit: int) -> tuple[str, str]:
    """Write dst = zip holding src as its only member (stored). Returns (name, status)."""
    name = os.path.basename(src)
    size = os.path.getsize(src)
    if os.path.exists(dst) and _wrapped_ok(src, dst):
        return name, "skipped"
    tmp = dst + ".tmp"
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        zf.write(src, name, compress_type=zipfile.ZIP_STORED)
    if os.path.getsize(tmp) >= limit:
        os.remove(tmp)
        raise RuntimeError(f"outer zip over the limit: {dst}")
    with zipfile.ZipFile(tmp) as zf:
        info = zf.infolist()
    if len(info) != 1 or info[0].filename != name or info[0].file_size != size:
        raise RuntimeError(f"outer zip does not hold {name} correctly")
    os.replace(tmp, dst)
    return name, "written"


def _wrapped_ok(src: str, dst: str, crc: bool = False) -> bool:
    """dst holds exactly one member named like src with the same size (and CRC if crc)."""
    try:
        with zipfile.ZipFile(dst) as zf:
            info = zf.infolist()
            if len(info) != 1 or info[0].filename != os.path.basename(src) \
                    or info[0].file_size != os.path.getsize(src):
                return False
            if crc:
                with open(src, "rb") as f:
                    c = 0
                    for chunk in iter(lambda: f.read(1 << 24), b""):
                        c = zlib.crc32(chunk, c)
                return c == info[0].CRC and zf.testzip() is None
            return True
    except (OSError, zipfile.BadZipFile):
        return False


def cmd_wrap(args) -> int:
    up, out = args.upload_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(p.is_dir() for p in up.iterdir()):
        print(f"[ERR] {up} has sub-folders; wrap expects a flat upload directory", file=sys.stderr)
        return 1
    zips = sorted(p for p in up.iterdir() if p.suffix == ".zip")
    others = sorted(p for p in up.iterdir() if p.is_file() and p.suffix != ".zip")
    for p in others:                     # manifests, extract_dataset.py: same content (hard link, or copy)
        d = out / p.name
        if not d.exists():
            try:
                os.link(p.resolve(), d)
            except OSError:              # another file system
                shutil.copy2(p, d)
    n_written = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(wrap_one, str(z), str(out / z.name), args.max_bytes) for z in zips]
        for i, fu in enumerate(as_completed(futs), 1):
            _, status = fu.result()
            n_written += status == "written"
            if i % 20 == 0 or i == len(futs):
                print(f"  [{i}/{len(futs)}] wrapped ({n_written} written)", flush=True)
    bad = 0
    if args.verify_crc:
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_wrapped_ok, [str(z) for z in zips], [str(out / z.name) for z in zips],
                              [True] * len(zips)))
        for z, ok in zip(zips, res):
            if not ok:
                print(f"[NG] CRC / contents differ: {z.name}")
                bad += 1
    files = [p for p in out.iterdir() if p.is_file()]
    over = [p for p in files if p.stat().st_size >= args.max_bytes]
    for p in over:
        print(f"[NG] over the size limit: {p.name} {p.stat().st_size}")
    bad += len(over)
    if len(files) != len(zips) + len(others):
        print(f"[NG] file count {len(files)} != {len(zips) + len(others)}")
        bad += 1
    print(f"files={len(files)} (zips {len(zips)}, others {len(others)})  "
          f"largest={max(p.stat().st_size for p in files)/1e9:.3f} GB")
    print("RESULT:", "OK" if bad == 0 else f"NG ({bad})")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--dataset-root", required=True, type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--component", help="folder, relative to dataset-root")
    g.add_argument("--files", nargs="+", help="individual files, relative to dataset-root")
    p.add_argument("--exclude", action="append",
                   help="sub-folder of --component (relative to dataset-root) left out because it is packed as its own "
                        "component (e.g. marmoset_keypoint_data/training_data/3dkp_gt); repeatable")
    p.add_argument("--store", action="store_true", help="store every member uncompressed (e.g. gzipped .blend)")
    p.add_argument("--prefix", required=True, help="zip 名の接頭辞")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-bytes", type=int, default=2_500_000_000, help="1ファイルの上限（この値未満にする）")
    p.add_argument("--margin", type=int, default=10_000_000, help="上限からさらに引く余裕")
    p.add_argument("--level", type=int, default=6, help="deflate 圧縮レベル")
    p.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("check")
    c.add_argument("--upload-dir", required=True, type=Path)
    c.add_argument("--dataset-root", type=Path, nargs="+",
                   help="also compare manifests with the source data (several allowed; OK if one matches)")
    c.add_argument("--max-files", type=int, default=1000)
    c.add_argument("--max-bytes", type=int, default=2_500_000_000)
    c.add_argument("--crc", action="store_true", help="全 zip の CRC を検査（全データを読むので遅い）")
    w = sub.add_parser("wrap", help="wrap each zip in an outer zip of the same name (Harvard Dataverse)")
    w.add_argument("--upload-dir", required=True, type=Path, help="flat upload directory (zips + manifests)")
    w.add_argument("--out-dir", required=True, type=Path)
    w.add_argument("--max-bytes", type=int, default=2_500_000_000)
    w.add_argument("--workers", type=int, default=8)
    w.add_argument("--verify-crc", action="store_true", help="re-read every file and compare CRCs (slow)")
    args = ap.parse_args()
    return {"pack": cmd_pack, "check": cmd_check, "wrap": cmd_wrap}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
