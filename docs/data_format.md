# Data Format Reference

## Input files

### Video files

- Format: `.mp4` or `.mkv`
- Resolution: 2048 × 1536 px
- Frame rate: synchronized across all 8 cameras

### Calibration file (`calibration.toml`)

anipose-format TOML file. One section per camera:

```toml
[cam_23506226]
name = "23506226"
size = [2048, 1536]
matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
distortions = [k1, k2, p1, p2, k3]
rotation = [rx, ry, rz]   # Rodrigues vector
translation = [tx, ty, tz] # in mm
```

### 2D keypoint file (`kp2d_f.pickle`)

```python
import pickle
with open("kp2d_f.pickle", "rb") as f:
    kp2d_f = pickle.load(f)

# Shape: (n_frame, n_keypoint, n_individual, 3, n_camera)
# dtype: float32
# kp2d_f[frame, kp, animal, :, cam] = [x, y, confidence]
```

---

## Intermediate files

### Per-camera 2D keypoint JSON (MMPose output)

```json
[
  {
    "image_id": 0,
    "keypoints": [x0, y0, score0, x1, y1, score1, ...]
  }
]
```

---

## Output files

### Released 3D keypoint file (`{session}.npz`)

One NPZ per session, produced by `scripts/05_create_digital_twin/alignment_3D_keypoint/run/export_aligned_npz.py`.
It contains only numeric and string arrays, so it loads without `allow_pickle`.

```python
import numpy as np
z = np.load("cj357_20221009_090000.npz")   # allow_pickle=False (default)
```

`T` = number of 3D frames (24 fps), `K` = 23 keypoints (20 measured + 3 derived), `C` = 8 cameras.
Terms (Aligned / Raw keypoints, 3D frame, Frame number, Video frame index, Blend frame) follow `GLOSSARY.md`;
see docs/adr/0001-release-only-aligned-keypoints-as-npz.md for why only this format is released.

| key | shape | dtype | description |
|---|---|---|---|
| `kp3d` | (T, K, 3) | float64 | **Aligned keypoints**: Blender world coordinates [m], after the fixed rotation and the per-session affine fitted by minimizing reprojection error |
| `kp_names` | (K,) | str | Keypoint names. The first 20 are measured; `shoulder_mid`, `base`, `center_eye` are midpoints |
| `head_euler_xyz` | (T, 3) | float64 | Head orientation, Euler XYZ [rad] |
| `kp3d_mm` | (T, 20, 3) | float64 | **Raw keypoints**: triangulated [mm] in the calibration frame (before rotation/alignment) |
| `kp3d_score` | (T, 20) | float64 | Minimum 2D confidence over the cameras used for triangulation |
| `kp3d_err` | (T, 20) | float64 | Mean reprojection error [px] |
| `joint_len` | (30,) | float64 | Fixed bone lengths [mm] |
| `rot` | (3, 3) | float64 | Fixed rotation applied to `kp3d_mm` |
| `affine_A`, `affine_t` | (3, 3), (3,) | float64 | Per-session affine fitted by minimizing reprojection error |
| `i_frame` | (T,) | int64 | 3D frame (0..T-1). The per-session Blender scenes use Blend frame = 3D frame + 1 |
| `camera_id` | (C,) | str | Camera IDs |
| `frame_num` | (C, T) | int64 | Frame number: imgstore `frame_number` of each camera matched to each 3D frame |
| `frame_min` | (C,) | int64 | First `frame_number` of each camera (Video frame index = `frame_num - frame_min`) |
| `frame_num_source` | () | str | `step1` (saved at triangulation) or `reconstructed` |
| `fps` | () | int64 | 24 |

NaN marks keypoints that were not triangulated (fewer than 2 cameras).

```python
# Relation between the Raw and Aligned keypoints (measured keypoints)
P = ((z["kp3d_mm"] @ z["rot"].T) / 1000) @ z["affine_A"].T + z["affine_t"]
assert np.allclose(P, z["kp3d"][:, :20], equal_nan=True)
```

### Intermediate 3D keypoint file (`kp3d.pickle` / `kp3d_fxdJointLen.pickle`, internal only)

Output of marmo3Dpose `reconst_single_step3.py`; `kp3d_fxdJointLen.pickle` is written when
`joint_len.npy` exists. It is the input to the NPZ export and is not distributed.

```python
with open("kp3d_fxdJointLen.pickle", "rb") as f:
    data = pickle.load(f)

# data['kp3d']       : (n_individual, n_frame, n_keypoint=20, 3) float64, mm
# data['kp3d_score'] : (n_individual, n_frame, n_keypoint) float64
# data['kp3d_err']   : (n_individual, n_frame, n_keypoint) float64, px
# data['joint_len']  : list of (30,) float64 arrays, one per individual
```

### Reprojection error (`pred_gt_comparison.json`)

```json
{
  "session_name": {
    "frame": 0,
    "keypoint": 0,
    "reprojection_error_px": 1.23,
    "cameras": ["23506226", ...]
  }
}
```

---

## File naming conventions

Results are stored per recording session:

```
results/
└── 3d_v0p8_dark_fix_20/
    └── dailylife_cj611_20230226_110000/
        ├── calibration.toml
        ├── kp2d_f.pickle
        ├── kp3d.pickle
        └── kp3d_fxdJointLen.pickle
```

Session name format: `{behavior}_{subject}_{date}_{time}`
- Example: `dailylife_cj611_20230226_110000`
  - behavior: `dailylife`
  - subject: `cj611`
  - date: `20230226` (2023-02-26)
  - time: `110000` (11:00:00)
