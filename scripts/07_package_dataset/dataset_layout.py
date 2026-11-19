"""
Write the dataset layout to YAML (for editing) and Markdown (for reading), and rearrange the
real directory to match an edited YAML.

  snapshot: scan the directory and write the YAML (keeps path/remove/description of an existing YAML)
  md:       write the Markdown view (tree + table) from the YAML
  plan:     list the moves implied by current vs path / remove in the YAML (writes nothing)
  apply:    carry out the plan (dry-run unless --execute), then update the YAML
  undo:     replay an apply log backwards

Files are only moved with os.rename; their contents are never rewritten (hard-linked originals
stay as they are). Entries with remove: true are not deleted but moved to removed_backup under
the same relative path.
I/O ledger: scripts/07_package_dataset/io.md#dataset_layout
"""
from __future__ import annotations

import argparse
import collections
import datetime
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = ROOT / "docs/dataset_layout.yaml"
DEFAULT_MD = ROOT / "docs/dataset_layout.md"
CAMERA_CONFIG = ROOT / "config/marmo_cj425m/config.yaml"
LOG_DIR = ROOT / "outputs/dataset_layout"

SESSION_RE = r"cj\d+_\d{8}_\d{6}"
# placeholder -> regex. {cam} is built from camera_id in the config
PLACEHOLDERS = {"session": SESSION_RE, "chunk": r"\d{6}", "frame": r"\d{7}", "name": r"[^/]+?"}
NAME_COLLAPSE = 20      # collapse basenames to {name} when one directory has more distinct patterns than this

HEADER = """\
# Dataset layout (docs/dataset_layout.md is generated from this file)
#
# How to edit:
#   path:        change it to move or rename files (do not edit current)
#                use the same placeholders as current: {session} {cam} {chunk} {frame} {name}
#   remove:      true moves the files to removed_backup under the same relative path (never deletes)
#   description: free text
# Apply:
#   .venv/bin/python scripts/07_package_dataset/dataset_layout.py plan      # review the moves
#   .venv/bin/python scripts/07_package_dataset/dataset_layout.py apply --execute
"""


def camera_ids() -> list[str]:
    return [str(c) for c in yaml.safe_load(open(CAMERA_CONFIG))["camera_id"]]


def placeholder_res() -> dict[str, str]:
    return {**PLACEHOLDERS, "cam": "(?:" + "|".join(camera_ids()) + ")"}


# ── patterns ────────────────────────────────────────────────────────────
def generalize(rel: str, cams: list[str]) -> str:
    """Real path -> pattern with placeholders (collapsing to {name} is done separately)."""
    q = re.sub(SESSION_RE, "{session}", rel)
    q = re.sub(r"(?<=[./_])(" + "|".join(cams) + r")(?=[./_]|$)", "{cam}", q)
    q = re.sub(r"(?<![\d{])\d{7}(?![\d}])", "{frame}", q)
    q = re.sub(r"(?<![\d{])\d{6}(?![\d}])", "{chunk}", q)
    return q


def pattern_regex(pattern: str, res: dict[str, str]) -> re.Pattern:
    """Pattern -> regex with named groups; repeated placeholders become back-references."""
    out, seen = [], set()
    for lit, key in re.findall(r"([^{]*)(?:\{(\w+)\})?", pattern):
        out.append(re.escape(lit))
        if not key:
            continue
        if key not in res:
            raise ValueError(f"unknown placeholder {{{key}}}: {pattern}")
        out.append(f"(?P={key})" if key in seen else f"(?P<{key}>{res[key]})")
        seen.add(key)
    return re.compile("".join(out) + r"\Z")


def scan(root: Path) -> list[tuple[str, int]]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = Path(dp) / fn
            files.append((str(p.relative_to(root)), p.lstat().st_size))
    return sorted(files)


def group_files(files, cams) -> dict[str, list[tuple[str, int]]]:
    """Pattern -> [(rel, size)]. Directories with many distinct names are collapsed to {name}."""
    pats = {rel: generalize(rel, cams) for rel, _ in files}
    by_dir = collections.defaultdict(set)
    for q in pats.values():
        d, b = os.path.split(q)
        by_dir[(d, os.path.splitext(b)[1])].add(b)
    collapse = {k for k, v in by_dir.items() if len(v) > NAME_COLLAPSE}
    groups = collections.defaultdict(list)
    for rel, size in files:
        q = pats[rel]
        d, b = os.path.split(q)
        ext = os.path.splitext(b)[1]
        if (d, ext) in collapse:
            q = os.path.join(d, "{name}" + ext)
        groups[q].append((rel, size))
    return dict(sorted(groups.items()))


def empty_dirs(root: Path) -> list[str]:
    return sorted(str(Path(dp).relative_to(root)) for dp, dns, fns in os.walk(root)
                  if not dns and not fns and Path(dp) != root)


# ── YAML I/O ────────────────────────────────────────────────────────────
def load_layout(path: Path) -> dict:
    return yaml.safe_load(open(path)) if path.exists() else {}


def save_layout(layout: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(layout, sort_keys=False, allow_unicode=True, width=120)
    path.write_text(HEADER + "\n" + body)


def match_entries(layout: dict, files, res) -> tuple[list[tuple[dict, list]], list[str]]:
    """Assign real files to entries by current; files matching no entry are returned separately."""
    entries = layout.get("entries", [])
    regs = [pattern_regex(e["current"], res) for e in entries]
    hits = [[] for _ in entries]
    unmatched = []
    for rel, size in files:
        for i, rg in enumerate(regs):
            m = rg.match(rel)
            if m:
                hits[i].append((rel, size, m.groupdict()))
                break
        else:
            unmatched.append(rel)
    return list(zip(entries, hits)), unmatched


# ── subcommands ─────────────────────────────────────────────────────────
def cmd_snapshot(args):
    root = Path(args.root or load_layout(args.yaml).get("dataset_root"))
    old = {e["current"]: e for e in load_layout(args.yaml).get("entries", [])}
    groups = group_files(scan(root), camera_ids())
    entries = []
    for pat, fl in groups.items():
        prev = old.get(pat, {})
        entries.append({
            "current": pat,
            "path": prev.get("path", pat),
            "remove": bool(prev.get("remove", False)),
            "count": len(fl),
            "size_gb": round(sum(s for _, s in fl) / 1e9, 2),
            "example": fl[0][0],
            "description": prev.get("description", ""),
        })
    layout = {
        "dataset_root": str(root),
        "removed_backup": load_layout(args.yaml).get("removed_backup", str(root) + "_removed"),
        "snapshot_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "total": {"files": sum(e["count"] for e in entries),
                  "size_gb": round(sum(e["size_gb"] for e in entries), 1)},
        "entries": entries,
        "empty_dirs": empty_dirs(root),
    }
    save_layout(layout, args.yaml)
    print(f"{len(entries)} entries, {layout['total']['files']} files -> {args.yaml}")
    cmd_md(args)


def cmd_md(args):
    layout = load_layout(args.yaml)
    entries = layout["entries"]
    lines = ["# Dataset layout", "",
             f"`{layout['dataset_root']}` ({layout['total']['files']} files, "
             f"{layout['total']['size_gb']} GB, scanned {layout['snapshot_at']})", "",
             "Generated from `docs/dataset_layout.yaml`. Edit the YAML, not this file.", "",
             "## Tree", "", "```"]
    tree: dict = {}
    for e in entries:
        node = tree
        parts = e["current"].split("/")
        for p in parts[:-1]:
            node = node.setdefault(p + "/", {})
        node[parts[-1]] = e
    def walk(node, prefix):
        items = list(node.items())
        for i, (k, v) in enumerate(items):
            last = i == len(items) - 1
            br = "└── " if last else "├── "
            if isinstance(v, dict) and "current" not in v:
                n = sum(1 for _ in _leaves(v)); gb = sum(x["size_gb"] for x in _leaves(v))
                lines.append(f"{prefix}{br}{k}  ({n} patterns, {gb:.1f} GB)")
                walk(v, prefix + ("    " if last else "│   "))
            else:
                tag = "  [remove]" if v["remove"] else (f"  -> {v['path']}" if v["path"] != v["current"] else "")
                lines.append(f"{prefix}{br}{k}  ×{v['count']}, {v['size_gb']} GB{tag}")
    def _leaves(v):
        for x in v.values():
            if isinstance(x, dict) and "current" not in x:
                yield from _leaves(x)
            else:
                yield x
    walk(tree, "")
    lines += ["```", "", "## Entries", "", "| Path | Files | GB | Description |", "|---|---:|---:|---|"]
    for e in entries:
        lines.append(f"| `{e['current']}` | {e['count']} | {e['size_gb']} | {e['description']} |")
    if layout.get("empty_dirs"):
        lines += ["", f"## Empty directories ({len(layout['empty_dirs'])})", ""]
        lines += [f"- `{d}`" for d in layout["empty_dirs"]]
    args.md.write_text("\n".join(lines) + "\n")
    print(f"-> {args.md}")


def build_plan(layout: dict):
    """Returns (moves[(src, dst)], problems[str], unmatched[str]); src/dst are absolute paths."""
    root = Path(layout["dataset_root"]); backup = Path(layout["removed_backup"])
    res = placeholder_res()
    matched, unmatched = match_entries(layout, scan(root), res)
    moves, problems = [], []
    for e, hits in matched:
        if len(hits) != e.get("count", len(hits)):
            problems.append(f"file count differs from the snapshot ({len(hits)} != {e['count']}): {e['current']}")
        if e.get("remove"):
            moves += [(root / rel, backup / rel) for rel, _, _ in hits]
            continue
        if e["path"] == e["current"]:
            continue
        cur_keys = set(re.findall(r"\{(\w+)\}", e["current"]))
        new_keys = set(re.findall(r"\{(\w+)\}", e["path"]))
        if new_keys - cur_keys:
            problems.append(f"placeholders not in current {sorted(new_keys - cur_keys)}: {e['path']}")
            continue
        for rel, _, g in hits:
            moves.append((root / rel, root / e["path"].format(**g)))
    dsts = collections.Counter(d for _, d in moves)
    problems += [f"duplicate destination ({n} files): {d}" for d, n in dsts.items() if n > 1]
    srcs = {s for s, _ in moves}
    problems += [f"destination already exists: {d}" for _, d in moves if d.exists() and d not in srcs]
    return moves, problems, unmatched


def _show(p: Path, root: Path, backup: Path) -> str:
    for base, tag in ((root, ""), (backup, "[removed_backup]/")):
        if base in p.parents:
            return tag + str(p.relative_to(base))
    return str(p)


def print_plan(moves, problems, unmatched, root: Path, backup: Path):
    summary = collections.Counter()
    for s, d in moves:
        summary[(_show(s.parent, root, backup), _show(d.parent, root, backup))] += 1
    for (sd, dd), n in sorted(summary.items()):
        print(f"  {n:6d}  {sd}/  ->  {dd}/")
    print(f"{len(moves)} moves")
    if unmatched:
        print(f"{len(unmatched)} files match no YAML entry (left in place):")
        for u in unmatched[:10]:
            print("   ", u)
    for p in problems:
        print("problem:", p)


def cmd_plan(args):
    layout = load_layout(args.yaml)
    moves, problems, unmatched = build_plan(layout)
    print_plan(moves, problems, unmatched, Path(layout["dataset_root"]), Path(layout["removed_backup"]))
    return moves, problems


def prune_empty(start: Path, stops: list[Path]):
    """Remove empty directories upwards from start, only below one of stops (stops themselves are kept)."""
    d = start
    while any(st in d.parents for st in stops) and d.is_dir() and not any(d.iterdir()):
        d.rmdir(); d = d.parent


def cmd_apply(args):
    layout = load_layout(args.yaml)
    root = Path(layout["dataset_root"])
    moves, problems = cmd_plan(args)
    if problems:
        sys.exit("problems found; nothing was done")
    if not moves:
        print("no changes"); return
    if not args.execute:
        print("dry-run (add --execute to run)"); return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"apply_{datetime.datetime.now():%Y%m%d_%H%M%S}.tsv"
    with open(log_path, "w") as log:
        for s, d in moves:
            d.parent.mkdir(parents=True, exist_ok=True)
            os.rename(s, d)
            log.write(f"{s}\t{d}\n"); log.flush()
            prune_empty(s.parent, [root])
    print(f"moved {len(moves)} files. log: {log_path}")
    # make the YAML describe the new layout
    layout["entries"] = [dict(e, current=e["path"]) for e in layout["entries"] if not e.get("remove")]
    save_layout(layout, args.yaml)
    cmd_snapshot(argparse.Namespace(root=None, yaml=args.yaml, md=args.md))


def cmd_undo(args):
    layout = load_layout(args.yaml)
    pairs = [l.rstrip("\n").split("\t") for l in open(args.log)]
    for s, d in reversed(pairs):
        if not args.execute:
            continue
        Path(s).parent.mkdir(parents=True, exist_ok=True)
        os.rename(d, s)
        prune_empty(Path(d).parent, [Path(layout["dataset_root"]), Path(layout["removed_backup"])])
    print(f"{'restored' if args.execute else 'would restore (dry-run, add --execute)'} {len(pairs)} files")
    if args.execute:
        print("The YAML was not restored; run snapshot to rescan the directory")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("snapshot"); p.add_argument("--root", help="dataset root (required on the first run)")
    sub.add_parser("md")
    sub.add_parser("plan")
    p = sub.add_parser("apply"); p.add_argument("--execute", action="store_true")
    p = sub.add_parser("undo"); p.add_argument("log", type=Path); p.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    {"snapshot": cmd_snapshot, "md": cmd_md, "plan": cmd_plan, "apply": cmd_apply, "undo": cmd_undo}[args.cmd](args)


if __name__ == "__main__":
    main()
