# floorZ - Vertical affine correction

A pipeline that applies KDE to the per-frame `min(bZ)` distribution of floor-contact KPs (wrist/ankle) to detect peaks,
and fits them to the physical heights `[0.7337, 1.0179, 1.3035, 1.5016] m` (floor/platform/bar/ceiling) with an affine
`bZ_corrected = a·bZ + b`.

All scripts accept `--input-dir / --output-dir / --session / --help`
(METHOD_SPEC compliant). The default output location is `scripts/dev/floorZ_files/` for run and
`scripts/dev/floorZ_plot/` for plot.

## run/ (algorithms and data generation -> floorZ_files/)

| Script | Input (default) | Output |
|---|---|---|
| `floorz_core.py` | — | Shared library (no CLI) |
| `pkl_to_corrected_csv.py` | `outputs/3d` (pickle) | `csv/`, `corrected/`, `affine_log.csv` |
| `validate_affine.py` | `outputs/animate_temp/test_scale` | `affine_summary.csv` |
| `estimate_floor_y.py` | `outputs/3d` (pickle) | `y_corrections.csv` (old kp3d Y-axis method) |

## plot/ (visualization -> floorZ_plot/)

| Script | Input (default) | Output |
|---|---|---|
| `plot_bz_dist_after.py` | `floorZ_files/corrected` + `affine_log.csv` | `grid_after_correction.png` (+`--before-after`) |
| `plot_bz_dist_before.py` | `../ez2use/.../transformedcsv` | `height_dist_grid_before.png` |
| `plot_bz_before_after_session.py` | `../ez2use/.../transformedcsv` | `before_after_{session}.png` |
| `plot_scaletest_validation.py` | `outputs/animate_temp/test_scale` | `height_dist_grid.png`, `affine_summary_plot.png` |
| `plot_floor_y_estimate.py` | `floorZ_files/y_corrections.csv` + pickle | `floor_estimate_summary.png` etc. |
| `plot_floor_z_per_session.py` | `floorZ_files/corrected` | `floor_estimate_corrected.png` |
| `plot_affine_regression.py` | `outputs/3d` | `affine_regression_explained.png` (explanatory figure) |
| `plot_kde_peak_check.py` | `outputs/3d` | `what_is_happening_kde_fixed.png` (explanatory figure) |
| `plot_correction_pipeline.py` | `outputs/3d` | `what_is_happening_kde.png` (explanatory figure) |

All parameters such as the KDE bandwidth / match distance, and the coordinate transform constants, are consolidated in
`run/floorz_core.py`. Edit this file when changing the algorithm.
See `docs/` for background discussion.

Example:
```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/floorZ/run/pkl_to_corrected_csv.py
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/floorZ/plot/plot_corrected.py --before-after
```
