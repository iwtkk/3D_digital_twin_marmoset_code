# Validation – Reprojection Error

Quantify 3D reconstruction accuracy via reprojection error.
Corresponds to paper section: **Technical Validation – 再投影誤差**

## Scripts

| Script | Purpose |
|---|---|
| `plot_ray_repr_error.py` | Visualize 3D rays and reprojection for one keypoint |
| `plot_repr_err_distribution.py` | Histogram of errors across all keypoints |
| `kp3d_err_violinplot.py` | Violin plot of per-keypoint error distribution |
| `analyze_large_repr_err.py` | Find frames/keypoints with large errors |
| `compute_violin_stats.py` | Compute statistics for violin plot |
| `plot_violin_from_stats.py` | Plot violin from precomputed stats |
| `kp3d_err_export.py` | Export errors to MATLAB or pickle format (`--format mat\|pickle\|both`) |
| `plot_frustum_kps_epi.py` | Frustum and epipolar line visualization |

## Usage

```bash
# Visualize 3D rays for keypoint 9 (L wrist) at frame 70
python plot_ray_repr_error.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/ \
  --config_yaml ../../config/marmo_cj425m/config.yaml \
  --frame 70 --animal 0 --kpn 9

# Error distribution plot
python plot_repr_err_distribution.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/

# Violin plot
python kp3d_err_violinplot.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/

# Find outlier frames
python analyze_large_repr_err.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/ \
  --threshold 10.0
```
