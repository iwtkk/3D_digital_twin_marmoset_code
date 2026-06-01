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

### 3D keypoint file (`kp3d.pickle`)

```python
import pickle
with open("kp3d.pickle", "rb") as f:
    kp3d = pickle.load(f)

# Shape: (n_frame, n_keypoint, 3)
# dtype: float64
# Units: millimeters
# kp3d[frame, kp] = [X, Y, Z]
```

### Fixed-joint-length 3D keypoints (`kp3d_fxdJointLen.pickle`)

```python
with open("kp3d_fxdJointLen.pickle", "rb") as f:
    data = pickle.load(f)

# data is a dict:
# data['kp3d'] : array of shape (n_individual, n_frame, n_keypoint, 3)
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
