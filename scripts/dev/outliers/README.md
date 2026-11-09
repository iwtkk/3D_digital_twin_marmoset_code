# outliers - Analysis of keypoint position outliers and bbox overflow

Two strands:
1. **kp3d outliers** — Detect (frame,kp) where raw kp3d (mm) falls outside the global inlier bbox.
2. **Blender bbox overflow** — Rank, per session, how far the corrected CSV (Blender m) overflows the Blender bbox
   and its vertical offset.

All scripts accept `--input-dir / --output-dir / --session / --help`
(the old `--csv-dir/--out-dir` are kept as aliases).

## run/ (-> outliers_files/)
| Script | Input (default) | Output |
|---|---|---|
| `analyze_kp3d_outliers.py` | `outputs/3d` (pickle) | `{session}_outlier.json`, `outlier_summary.json` |

## plot/ (-> outliers_plot/)
| Script | Input (default) | Output |
|---|---|---|
| `plot_kp_outlier_dist.py` | `outliers_files` (JSON) | `deviation_dist.png` |
| `plot_bbox_deviation_dist.py` | `floorZ_files/corrected` (CSV) | `deviation_dist*.png`, `position_dist.png`, `position_hist.npz` |
| `plot_kp_dist_before_after.py` | `floorZ_files/corrected` + transformedcsv | `position_pre_post.png` |
| `plot_vertical_offset_rank.py` | `floorZ_files/corrected` (CSV) | `vertical_offset_by_session.{csv,png}`, `vertical_dist_by_session.png` |
| `plot_kp_position_dist.py` | `floorZ_files/corrected` (CSV) | `position_dist_{session}.png`, `position_dist_{animal}.png` |

`plot_bbox_deviation_dist.py` provides shared helpers (bbox constants, position histogram aggregation),
which `plot_kp_dist_before_after.py` and `plot_vertical_offset_rank.py` import.
For how the bbox is evaluated, see this section of `README.md` and each script's docstring.
