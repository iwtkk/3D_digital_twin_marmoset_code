# Validation – Keypoint Quality

Visualize and inspect estimated 2D/3D keypoints.
Corresponds to paper section: **Technical Validation – 推定された2D・3Dキーポイント**

## Scripts

| Script | Purpose |
|---|---|
| `test_projection_error.py` | Triangulate one frame and compute reprojection error |
| `plot_3dkp_on_2dimg.py` | Overlay 3D keypoints projected onto camera images |
| `visualize_3D_frames_png.py` | Render 3D pose as PNG for each frame |
| `visualize_touch_temporal.py` | Temporal visualization of touch events |

## Usage

```bash
# Inspect one frame
python test_projection_error.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/ \
  --config_yaml ../../config/marmo_cj425m/config.yaml \
  --frame 70 --animal 0

# Overlay keypoints on image
python plot_3dkp_on_2dimg.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/ \
  --config_yaml ../../config/marmo_cj425m/config.yaml \
  --frame 70 --cam_id 23506226

# Export 3D pose frames as PNG
python visualize_3D_frames_png.py \
  --base data/marmoset_keypoint_data/3D_temporal_keypoint/{session}/ \
  --output outputs/validation/keypoints/{session}/
```
