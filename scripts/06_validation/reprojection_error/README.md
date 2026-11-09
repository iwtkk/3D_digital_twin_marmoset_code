# Validation – Reprojection Error

Quantify 3D reconstruction accuracy via reprojection error.
Corresponds to paper section: **Technical Validation – Reprojection error**

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
| `run_all.sh` | Regenerate all outputs above for one kp3d source (`BASE OUT [RAY_SESSION]`) |

## Usage

Regenerate everything (raw error, and the error of the reprojection-affine-aligned keypoints computed by
`scripts/dev/reproj_tail/run/compute_aligned_reproj_err.py`):

```bash
bash run_all.sh /data/iwata/3D_digital_twin_marmoset_data/marmoset_keypoint_data/3D_temporal_keypoint_data \
    outputs/validation/reprojection_error/raw cj611_20230226_090000
bash run_all.sh outputs/dev/reproj_tail_aligned/err_npz outputs/validation/reprojection_error/aligned
```

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

## Note on the violin KDE bandwidth

`compute_violin_stats.py` and `kp3d_err_violinplot.py` use Scott's rule with a robust scale
(`min(std, IQR/1.349)`, `compute_violin_stats.robust_bw_factor`). With the plain std, a single failed
triangulation (R elbow reaches 6.0e5 px in `cj611_20210831_110000`) widened the kernel and flattened that
keypoint's violin. Without such outliers the bandwidth is identical to scipy/matplotlib's default.
