# quality_score - Distributions of kp3d_score / kp3d_err

Without applying thresholds, describes the distributions of the detection confidence score (higher is better, clamped at 0.5) and the reprojection error err (px,
lower is better). score and err are complementary (low score = uncertain detection, high err = geometric
inconsistency), so the joint distribution is also examined.

| Category | Script | Input (default) | Output |
|---|---|---|---|
| run | `analyze_quality_score.py` | `outputs/3d` | `quality_dist_summary.json`, `quality_subsample.npz` → `quality_score_files/` |
| plot | `plot_quality_score.py` | `quality_score_files` | dist/joint/perkp/heatmap/clamp_nan figures → `quality_score_plot/` |

All scripts support `--input-dir / --output-dir / --session / --help`.
