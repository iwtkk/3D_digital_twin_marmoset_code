# Camera Mapping

## Camera serial ID → camera name

8 synchronized cameras (FLIR Blackfly S) were used to record marmoset behavior.

| Camera name | Serial ID | Approximate position |
|---|---|---|
| Cam 1 | 23506226 | - |
| Cam 2 | 23511607 | - |
| Cam 3 | 23506214 | - |
| Cam 4 | 23506236 | - |
| Cam 5 | 23511614 | - |
| Cam 6 | 23506237 | - |
| Cam 7 | 23511613 | - |
| Cam 8 | 23506239 | - |

## Camera order in config

The order of cameras used in reconstruction is defined in `config/marmo_cj425m/config.yaml`:

```yaml
camera_id:
  - 23506239   # Cam 8
  - 23506237   # Cam 6
  - 23506236   # Cam 4
  - 23511607   # Cam 2
  - 23511613   # Cam 7
  - 23511614   # Cam 5
  - 23506214   # Cam 3
  - 23506226   # Cam 1
```

This order must match the order used in `calibration.toml` and in `kp2d_f.pickle` (the last axis).

## Calibration files

Calibration parameters are stored in `config/marmo_cj425m/`:

| File | Contents |
|---|---|
| `cam_intrinsic.h5` | Intrinsic parameters (focal length, principal point, distortion) |
| `cam_extrinsic.h5` | Extrinsic parameters (rotation, translation) before optimization |
| `cam_extrinsic_optim.h5` | Optimized extrinsic parameters |
| `config.yaml` | Camera serial IDs and calibration settings |

At runtime, the pipeline uses `calibration.toml` (anipose format) generated from the above HDF5 files.

## Camera intrinsics

Image resolution: **2048 × 1536 pixels**

Calibration was performed using a chessboard pattern with:
- Square size: **23 mm**

## Coordinate system

- World coordinates origin: defined by the calibration setup.
- Z-axis: pointing upward (away from the cage floor).
- Units: **millimeters**.

## Reprojection error analysis

Scripts for inspecting per-camera reprojection error:

```bash
# Plot 3D rays and reprojection for a single keypoint
python src/plot_ray_repr_error.py \
  --base /path/to/results/session/ \
  --config_yaml config/marmo_cj425m/config.yaml \
  --frame 70 --animal 0 --kpn 0

# Distribution of reprojection errors across all keypoints
python repr_error/plot_repr_err_distribution.py \
  --base /path/to/results/session/
```
