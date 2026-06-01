# Blender Usage

## Requirements

- Blender 4.5.0
- Python 3.10 (Blender's built-in Python)
- NumPy (available in Blender's Python)

## Scripts overview

All Blender-side scripts are in `animate/BlenderSync/`:

| Script | Purpose |
|---|---|
| `blender_animation.py` | Apply 3D keypoints to IK rig bones (main animation script) |
| `allcamrender.py` | Render from all virtual camera positions |
| `csvpsuedoeye.py` | Pseudo-eye-view CSV generation |
| `process_dlc.py` | DLC-format keypoint processing |
| `transform_dlc.py` | Coordinate transform for DLC data |
| `psuedo.py` | Pseudo-view rendering utilities |
| `render_range.py` | Render a frame range |
| `run_blender_chunks_parallel.py` | Parallel rendering in chunks |

## Step 1 – Apply 3D keypoints to the rig

```bash
blender /path/to/marmoset_scene.blend \
  --background \
  --python animate/BlenderSync/blender_animation.py \
  -- \
  --keypoints3d /path/to/kp3d_fxdJointLen.pickle \
  --session dailylife_cj611_20230226_110000 \
  --start_frame 0 \
  --end_frame 500
```

This script:
1. Loads the `.pickle` file containing 3D keypoints
2. Iterates over frames, setting IK bone world positions from keypoint coordinates
3. Inserts keyframes for each bone at each frame
4. Saves the animated `.blend` file

## Step 2 – Render pseudo-first-person view

```bash
blender /path/to/animated_scene.blend \
  --background \
  --python animate/BlenderSync/allcamrender.py \
  -- \
  --output_dir outputs/render/dailylife_cj611_20230226_110000/ \
  --cam_id 23506226
```

## Step 3 – Parallel rendering (long sessions)

For sessions with many frames, use chunk-based parallel rendering:

```bash
python animate/run_blender_chunks_parallel.py \
  --blend /path/to/animated_scene.blend \
  --output_dir outputs/render/ \
  --total_frames 3000 \
  --chunk_size 100 \
  --n_workers 4
```

Or use the provided shell scripts:

```bash
bash animate/run_render.sh
bash animate/run_animate_skip_parallized.sh
```

## Coordinate system notes

- 3D keypoints from reconstruction are in **millimeters**, world coordinates.
- Blender uses **meters** by default. Scale factor: multiply mm × 0.001 when setting bone locations.
- Camera extrinsics in `calibration.toml` define the same world coordinate system.

## Virtual cameras in Blender

Camera positions in Blender correspond to the real cameras (serial IDs).
Camera placement uses `set_object_world_loc_euler()` from `blender_animation.py`:

```python
set_object_world_loc_euler(cam_obj, loc_xyz_meters, euler_xyz_radians)
```

Camera parameters (intrinsics) are loaded from `config/marmo_cj425m/cam_intrinsic.h5`.

## Troubleshooting

**Blender can't find the script**
- Run Blender from the repository root: `cd /path/to/repo && blender ...`

**Keypoints look flipped**
- Check the coordinate system. The reconstruction applies a −90° rotation around the X-axis for visualization in `plot_ray_repr_error.py`. The raw keypoints in `.pickle` are in the unrotated world frame.

**Bones don't move**
- Confirm that the IK bone names in your `.blend` file match those listed in [bone_mapping.md](bone_mapping.md).
- Check that keyframes are being inserted: open the Blender timeline and look for keyframe markers.
