# 04 – Validation

Evaluate reconstruction quality using reprojection error, SSIM, PSNR,
and visual inspection.

## Reprojection error analysis

### Visualize rays for a single keypoint

```bash
python plot_ray_repr_error.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/ \
  --config_yaml ../../config/marmo_cj425m/config.yaml \
  --frame 70 --animal 0 --kpn 9
```

### Error distribution across all keypoints

```bash
python plot_repr_err_distribution.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/
```

### Violin plot

```bash
python kp3d_err_violinplot.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/
```

### Find frames with large errors

```bash
python analyze_large_repr_err.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/
```

## SSIM / PSNR

Compare real camera frames with rendered pseudo-view frames:

```bash
python ssim.py \
  --real outputs/frames/{session}/{cam_id}/ \
  --rendered outputs/render/{session}/{cam_id}/ \
  --output outputs/validation/ssim_{session}.csv
```

Or use the class-based pipeline:

```bash
python run_pipeline.py \
  --session {session} \
  --config ../../config/consistency.yaml
```

## Scripts

| Script | Purpose |
|---|---|
| `plot_ray_repr_error.py` | 3D ray visualization for reprojection |
| `plot_repr_err_distribution.py` | Error histogram across keypoints |
| `kp3d_err_violinplot.py` | Violin plot of errors |
| `analyze_large_repr_err.py` | Find outlier frames/keypoints |
| `compute_violin_stats.py` | Compute statistics for violin plot |
| `plot_violin_from_stats.py` | Plot from precomputed stats |
| `kp3d_err_export.py` | Export errors to MATLAB or pickle format |
| `ssim.py` | SSIM/PSNR between real and rendered |
| `run_pipeline.py` | Full validation pipeline |
| `calculate_metrics.py` | Batch metric calculation |
| `visualize.py` | Visualization utilities |
| `outlier_detector.py` | Outlier frame detection |
| `plot_3dkp_on_2dimg.py` | Overlay 3D keypoints on image |
| `plot_frustum_kps_epi.py` | Frustum and epipolar visualization |
| `visualize_3D_frames_png.py` | 3D pose visualization as PNG |
| `visualize_touch_temporal.py` | Temporal touch event visualization |
