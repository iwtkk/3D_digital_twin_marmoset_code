#!/usr/bin/env python3
"""Upload a flat upload directory to a Harvard Dataverse dataset through the API (dvuploader).

Subcommands
  info    check the token, the dataset and whether direct (S3) upload is available
  upload  upload every file of --upload-dir (resumable: files already in the dataset are skipped by name)
  verify  compare the dataset's file list (draft version) with --upload-dir (names and sizes; --md5 also checksums)

Notes
  - Upload the plain zips (path_registry.outputs.dataset_upload_npz). Harvard Dataverse offers direct (S3) upload,
    which does not unzip, so zips stay as they are (tested 2026-11-08: a wrapped zip stayed wrapped).
    With --force-native Dataverse would unzip each zip; use the wrapped dataset_upload_dataverse in that case.
    verify accepts a wrapper's own size or its inner size.
  - Every file is sent on its own with tabIngest=false (dvuploader's own bundling of small files is disabled,
    because a bundle would be uploaded with ingest on).
  - Token: $DATAVERSE_API_TOKEN, or --token-file (default ~/.config/dataverse/token, keep it chmod 600).

Example
  .venv/bin/python scripts/07_package_dataset/upload_dataverse.py upload \
      --pid doi:10.7910/DVN/XXXXXX --upload-dir <dataset_upload_dataverse> --batch-size 10 --parallel 4

I/O ledger: scripts/07_package_dataset/io.md
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# one package per file -> each upload carries its own tabIngest=false (must be set before importing dvuploader)
os.environ.setdefault("DVUPLOADER_MAX_PKG_SIZE", "1")

import httpx  # noqa: E402

DEFAULT_SERVER = "https://dataverse.harvard.edu"
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "dataverse" / "token"


def read_token(token_file: Path) -> str:
    tok = os.environ.get("DATAVERSE_API_TOKEN", "").strip()
    if tok:
        return tok
    if not token_file.exists():
        sys.exit(f"[ERR] no token: set DATAVERSE_API_TOKEN or create {token_file}")
    return token_file.read_text().strip()


def api_get(server: str, token: str, path: str, **params) -> dict:
    r = httpx.get(f"{server}{path}", params=params, headers={"X-Dataverse-key": token}, timeout=120)
    try:
        d = r.json()
    except ValueError:
        sys.exit(f"[ERR] {path}: HTTP {r.status_code} {r.text[:300]}")
    if r.status_code != 200 or d.get("status") != "OK":
        sys.exit(f"[ERR] {path}: HTTP {r.status_code} {d.get('message', d)}")
    return d["data"]


def dataset_files(server: str, token: str, pid: str, with_md5: bool = False) -> dict:
    """filename -> size (or (size, md5) with with_md5) of the latest version (draft if one exists)."""
    data = api_get(server, token, "/api/datasets/:persistentId/versions/:latest/files", persistentId=pid)
    out = {}
    for f in data:
        df = f["dataFile"]
        name = os.path.join(f.get("directoryLabel", ""), df["filename"]) if f.get("directoryLabel") else df["filename"]
        size = int(df.get("filesize", -1))
        ck = df.get("checksum") or {}
        out[name] = (size, ck.get("value") if ck.get("type") == "MD5" else None) if with_md5 else size
    return out


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def local_files(upload_dir: Path) -> list[Path]:
    if not upload_dir.is_dir():
        sys.exit(f"[ERR] not a directory: {upload_dir}")
    if any(p.is_dir() for p in upload_dir.iterdir()):
        sys.exit(f"[ERR] {upload_dir} has sub-folders; expected a flat upload directory")
    return sorted(p for p in upload_dir.iterdir() if p.is_file() and not p.name.endswith((".tmp", ".part")))


def accepted_sizes(p: Path) -> set[int]:
    """The file's own size, plus the inner size if it is a wrapper zip (single member of the same name)."""
    sizes = {p.stat().st_size}
    if p.suffix == ".zip":
        try:
            with zipfile.ZipFile(p) as z:
                infos = z.infolist()
                if len(infos) == 1 and infos[0].filename == p.name:
                    sizes.add(infos[0].file_size)
        except zipfile.BadZipFile:
            pass
    return sizes


def cmd_info(a, token: str) -> int:
    me = api_get(a.server, token, "/api/users/:me")
    print(f"user: {me.get('displayName')} ({me.get('identifier')})")
    ds = api_get(a.server, token, "/api/datasets/:persistentId", persistentId=a.pid)
    lv = ds.get("latestVersion", {})
    title = next((f["value"] for f in lv.get("metadataBlocks", {}).get("citation", {}).get("fields", [])
                  if f.get("typeName") == "title"), "?")
    print(f"dataset: {a.pid}  title: {title}  version state: {lv.get('versionState')}  files: {len(lv.get('files', []))}")
    r = httpx.get(f"{a.server}/api/datasets/:persistentId/uploadurls", params={"persistentId": a.pid, "size": 1024},
                  headers={"X-Dataverse-key": token}, timeout=60)
    print(f"direct (S3) upload: {'yes' if r.status_code == 200 else 'no'} (HTTP {r.status_code})")
    return 0


def cmd_upload(a, token: str) -> int:
    from dvuploader import DVUploader, File

    files = local_files(a.upload_dir)
    if a.only:
        files = [p for p in files if any(p.match(g) for g in a.only)]
    existing = dataset_files(a.server, token, a.pid)
    todo = [p for p in files if p.name not in existing]
    total = sum(p.stat().st_size for p in todo)
    print(f"local: {len(files)}  already in dataset: {len(files) - len(todo)}  to upload: {len(todo)} ({total / 1e9:.2f} GB)")
    if a.dry_run:
        for p in todo:
            print(f"  {p.name}")
        return 0

    failed = []
    for i in range(0, len(todo), a.batch_size):
        batch = todo[i:i + a.batch_size]
        print(f"== batch {i // a.batch_size + 1}/{(len(todo) + a.batch_size - 1) // a.batch_size}: "
              f"{batch[0].name} .. {batch[-1].name}", flush=True)
        t0 = time.time()
        try:
            DVUploader(files=[File(filepath=str(p), tabIngest=False) for p in batch]).upload(
                persistent_id=a.pid, dataverse_url=a.server, api_token=token,
                n_parallel_uploads=a.parallel, force_native=a.force_native, replace_existing=False)
        except Exception as e:  # keep going; the next run retries whatever is missing
            print(f"[WARN] batch failed: {type(e).__name__}: {e}", flush=True)
        now = dataset_files(a.server, token, a.pid)
        missing = [p.name for p in batch if p.name not in now]
        failed += missing
        gb = sum(p.stat().st_size for p in batch) / 1e9
        print(f"   {len(batch) - len(missing)}/{len(batch)} in dataset, {gb:.1f} GB in {time.time() - t0:.0f} s"
              + (f"; missing: {missing}" if missing else ""), flush=True)
    print(f"RESULT: {'OK' if not failed else f'{len(failed)} files missing (rerun to retry)'}")
    return 0 if not failed else 1


def cmd_verify(a, token: str) -> int:
    files = local_files(a.upload_dir)
    remote_full = dataset_files(a.server, token, a.pid, with_md5=True)
    remote = {k: v[0] for k, v in remote_full.items()}
    missing, bad_size, bad_md5 = [], [], []
    for p in files:
        if p.name not in remote:
            missing.append(p.name)
        elif remote[p.name] not in accepted_sizes(p):
            bad_size.append((p.name, remote[p.name], sorted(accepted_sizes(p))))
    if a.md5:
        cand = [p for p in files if p.name in remote and remote[p.name] == p.stat().st_size]
        print(f"md5: hashing {len(cand)} local files ({sum(p.stat().st_size for p in cand) / 1e9:.1f} GB) ...", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            for p, h in zip(cand, ex.map(md5_file, [str(p) for p in cand])):
                if remote_full[p.name][1] != h:
                    bad_md5.append((p.name, remote_full[p.name][1], h))
        print(f"md5 mismatch: {len(bad_md5)} (wrapper zips with a different size are not hashed)")
        for n, r, l in bad_md5[:50]:
            print(f"  md5      {n}: dataset {r}, local {l}")
    extra = sorted(set(remote) - {p.name for p in files})
    print(f"local: {len(files)}  dataset: {len(remote)}  missing: {len(missing)}  size mismatch: {len(bad_size)}  "
          f"extra in dataset: {len(extra)}")
    for n in missing[:50]:
        print(f"  missing  {n}")
    for n, r, l in bad_size[:50]:
        print(f"  size     {n}: dataset {r}, local {l}")
    for n in extra[:50]:
        print(f"  extra    {n}")
    ok = not missing and not bad_size and not extra and not bad_md5
    print(f"RESULT: {'OK' if ok else 'NG'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=DEFAULT_SERVER, help=f"Dataverse URL (default: {DEFAULT_SERVER})")
    ap.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("info", "upload", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--pid", required=True, help="dataset persistent id, e.g. doi:10.7910/DVN/XXXXXX")
        if name != "info":
            p.add_argument("--upload-dir", type=Path, required=True)
        if name == "verify":
            p.add_argument("--md5", action="store_true", help="also compare MD5 with the dataset (reads every local file)")
            p.add_argument("--workers", type=int, default=8, help="parallel hash workers (default 8)")
        if name == "upload":
            p.add_argument("--batch-size", type=int, default=10, help="files per dvuploader call (default 10)")
            p.add_argument("--parallel", type=int, default=4, help="parallel uploads within a batch (default 4)")
            p.add_argument("--only", nargs="+", help="upload only names matching these globs (e.g. for a test)")
            p.add_argument("--force-native", action="store_true", help="never use direct (S3) upload")
            p.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    a = ap.parse_args()
    a.server = a.server.rstrip("/")
    token = read_token(a.token_file)
    return {"info": cmd_info, "upload": cmd_upload, "verify": cmd_verify}[a.cmd](a, token)


if __name__ == "__main__":
    sys.exit(main())
