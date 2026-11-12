# alignment_3D_keypoint

Per-session affine alignment of the 3D keypoints to the digital twin by minimising the reprojection error,
and its evaluation. Tool details (inputs / outputs) are in `.agent_docs/docs/tool_registry.md`.

## plot/plot_real_vs_blender_2d.py

Overlays on real camera frames the 2D projections of the same 3D keypoints:

| colour | projection |
|---|---|
| red | real cameras (omnidir, `cam_intrinsic.h5` + `cam_extrinsic_optim.h5`) from `kp3d_mm` |
| cyan | Blender cameras (equisolid) after the initial placement `A0 @ X_m + t0` (fixed rotation, per-axis scale, translation) |
| magenta | Blender cameras after the fitted per-session affine `A @ X_m + t` |

`X_m = (kp3d_mm @ rot.T) / 1000`. Reads `outputs/reproj_eval/aligned_npz/{session}.npz`, so the frame
mapping uses the step1 `frame_num` (not `frame_map.py`).

```bash
.venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/plot/plot_real_vs_blender_2d.py \
    --session cj357_20221009_090000 --n-frames 3
```

Outputs go to `outputs/reproj_eval/real_vs_blender/{session}/`: full-size overlays per camera, a 2x4 grid per
frame, and `{session}_px_diff.csv` (median / p90 distance from red to cyan and to magenta, per camera).
The grid layout is set by `--grid-order` (default right-most column: 23511613 top, 23506239 bottom).
`--clean-cams CAM` also saves that camera's grid cell without overlays, and `--extra-image CAM=PATH` crops
a full-size 2048x1536 image (e.g. a Blender render of the same camera; blend frame = kp3d index + 1) with
the same window (`*_clean.png`, `*_blender.png`; requires a single `--frames` value).
Red and magenta do not coincide exactly because the real (omnidir) and Blender (equisolid) lens models
differ (~7 px remains after the fit).

## plot/plot_affine_err_by_camera.py

Per-camera box plot over all sessions of the distance between the real-camera and the Blender-camera
projections of the same keypoints: initial placement (left box) vs. affine fit (right box). Evaluated on
frames not used for the fit. Reads only the NPZ files (no video).

```bash
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
    scripts/05_create_digital_twin/alignment_3D_keypoint/plot/plot_affine_err_by_camera.py --jobs 16
```

Outputs `outputs/reproj_eval/affine_err_by_camera/affine_err_by_camera.{csv,png,pdf}`.

## run/render_initial_placement.py

Renders a session with the animal posed at the initial placement (before the affine fit). The released
session `.blend` is only copied, never edited: the copy is re-keyframed with `blender_animation.py` from an
initial-placement CSV and rendered with `render_range.py` (blend frame = kp3d index + 1).

```bash
.venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/run/render_initial_placement.py \
    --session cj357_20221009_090000 --frames 9585 19170 --camera 23511613 --gpu 2
# crop with the same window as the grid cell
.venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/plot/plot_real_vs_blender_2d.py --session cj357_20221009_090000 \
    --skip-stats --frames 9585 --extra-image 23511613=outputs/reproj_eval/initial_placement/cj357_20221009_090000/render/cj357_20221009_090000_initial_Camera.23511613_09586.png \
    --extra-tag blender_initpose
```
