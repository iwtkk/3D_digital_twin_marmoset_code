# scripts/dev - Analysis scripts

Each target follows the conventions in `docs/METHOD_SPEC.md` and is split into `run/` (algorithms and data generation) and
`plot/` (visualization). All scripts accept `--input-dir / --output-dir / --session /
--help`. The default output location is `scripts/dev/{target}_files/` for run and
`scripts/dev/{target}_plot/` for plot.

| target | Description | Details |
|---|---|---|
| `floorZ` | Vertical correction of floor-contact KPs via KDE+affine | [README](floorZ/README.md) |
| `teleport` | Detection of physically implausible keypoint jumps (v1/v2) | [README](teleport/README.md) |
| `kinematics` | Velocity/acceleration/jerk distributions, PSD, bone-length stability | [README](kinematics/README.md) |
| `outliers` | kp3d / Blender bbox overflow and vertical-offset ranking | [README](outliers/README.md) |
| `bbox` | Aggregation of pre-correction Blender bbox overflow | [README](bbox/README.md) |
| `quality_score` | Distributions of kp3d_score / kp3d_err | [README](quality_score/README.md) |
| `reproj_tail` | Identifying the KPs / frames / sessions that form the reprojection error tail, and sensitivity analysis of excluding them | [README](reproj_tail/README.md) |
| `strip` | Utility for removing filename markers | [README](strip/README.md) |
| `images_to_video` | Image sequence -> MP4 | [README](images_to_video/README.md) |
| `pose2d` | Overlay raw 2D pose estimation results on real camera videos for visual inspection | [README](pose2d/README.md) |
| `render_compare` | Comparison videos with real camera footage and the Blender render side by side | [README](render_compare/README.md) |
| `fur_ablation` | SSIM against real images per fur variant (hair-curve / simplified / none) and fur thickness | [README](fur_ablation/README.md) |
| `consistency_roi` | Compute digital twin validation PSNR/SSIM in an ROI around the marmoset | [README](consistency_roi/README.md) |
| `consistency_camvar` | Isolating the causes of camera-to-camera variation in digital twin validation SSIM/PSNR | [README](consistency_camvar/README.md) |
| `pose2d_error` | 2D keypoint estimation error on the training data (per-keypoint violins) | [README](pose2d_error/README.md) |
| `benchmark` | Processing speed of each pipeline stage (2D inference / 3D reconstruction / Blender) and real-time feasibility | [README](benchmark/README.md) |

In the typical flow, run generates data (CSV/JSON/NPZ) in `{target}_files/`, and plot
reads it and writes figures to `{target}_plot/`. Shared algorithms are consolidated in each target's `run/*_core.py` /
`*_common.py`.
