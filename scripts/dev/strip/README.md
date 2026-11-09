# strip - Utility for removing filename markers

An in-place renaming utility that restores filenames containing `work_` / `dailylife_` / `_points.csv_transformedtest`
to their plain names. Dry-run by default; `--execute` to apply.

| Category | Script | Input | Action |
|---|---|---|---|
| run | `strip_marked_filenames.py` | `--input-dir DIR` | Rewrites filenames in DIR |

Accepts `--input-dir / --output-dir / --session / --help` (output-dir /
session are unused because it operates in place). `--execute` / `--recursive` are available.
