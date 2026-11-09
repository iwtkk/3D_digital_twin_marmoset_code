# bbox - Analysis of pre-correction Blender bbox overflow

Aggregates, per session, how far the pre-correction Blender coordinates (transformedcsv, with only the global scale applied) overflow
the Blender bbox. Vertical (z) and horizontal (x,y) are treated separately.

| Category | Script | Input (default) | Output |
|---|---|---|---|
| run | `analyze_bbox_overflow.py` | `../ez2use/.../transformedcsv` | `bbox_overflow.csv` → `bbox_files/` |
| plot | `plot_bbox_overflow.py` | `bbox_files/bbox_overflow.csv` | `bbox_overflow_top.png` → `bbox_plot/` |

All scripts support `--input-dir / --output-dir / --session / --help`.
