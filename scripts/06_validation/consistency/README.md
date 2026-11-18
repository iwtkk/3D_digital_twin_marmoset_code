# Validation – consistency of the reproduced environment (Fig. 8a)

Reproduces the Fig. 8a image-similarity values from the Published dataset alone: renders of the digital twin are
compared with the real camera images on the **Full image** and the **Animal region** (terms: `GLOSSARY.md`).
The dataset distributes only the metrics and the frame list
(`3D_digital_twin_data/rendered_environmental_images/`, docs/adr/0002-fig8a-metrics-without-render-images.md);
this folder regenerates the renders and checks the reproduced values against those files.

Paper section: **Technical Validation – Validation of consistency in reproducing the experimental environment**.

## Requirements

- The extracted Published dataset (`scripts/00_setup/extract_dataset.py`), at least: `3D_digital_twin_data/`
  (`rendered_environmental_images/`, `3D_digitaltwin_marmoset_blender_files/`, `3D_digitaltwin_simplified_marmoset/`
  for `simplified_fur`), and `marmoset_keypoint_data/` (`video_data/`, `2D_temporal_keypoint_data/`,
  `3D_temporal_keypoint_data/`) for the 108 Sessions listed in `frames.csv`.
- **Blender 5.2.x** with a CUDA GPU (other versions render slightly different pixels; 5.1 vs 5.2 differ by
  0.45 dB PSNR on the same scene).
- The repository Python environment (`.venv`: numpy, pandas, opencv, scikit-image, imgstore, matplotlib).

## Run

```bash
# everything: 3 body models x 2400 pairs (~75 s per image on one GPU, about 150 GPU hours)
scripts/06_validation/consistency/run_all.sh /path/to/dataset outputs/validation/consistency_fig8a 0,1,2,3 /path/to/blender-5.2

# quick check: two Sessions (16 pairs), Detailed model with fur only
SESSIONS="cj425_20210427_180000 cj425_20210428_080000" MODELS=detailed_fur \
  scripts/06_validation/consistency/run_all.sh /path/to/dataset outputs/validation/consistency_fig8a 0
```

| step | script | output (under WORK_DIR) |
|---|---|---|
| 1. Render | `render_pairs.py` (→ `render_pairs_blender.py` in Blender) | `render/{model}/{session}/Camera.{camera_id}/{frame_3d:07d}.png`, `render/background/Camera.{camera_id}.png` |
| 2. Real frames | `extract_real_frames.py` | `real/{session}/Camera.{camera_id}/{frame_3d:07d}.png` |
| 3. Metrics + verify | `compute_metrics.py --verify` | `metrics/pair_metrics.csv`, `camera_summary.csv`, `metrics_animal_{model}.json`, `verify.json` |
| 4. Figures | `visualize.py` | `figures/{model}/by_camera.png`, `by_animal.png`, ... |

Every step skips what already exists, so an interrupted run resumes with the same command.
Shared definitions (paths in the dataset, Animal region, metrics) are in `consistency_core.py`;
inputs and outputs of each script are listed in `io.md`.

## What is rendered

- Scene: `3D_digitaltwin_marmoset_blender_files/{session}.blend` (posed with the Aligned keypoints), camera
  `Camera.{camera_id}`, at the **Blend frame** = 3D frame + 1. Cycles, 500 samples, 2048x1536.
- `detailed_fur`: the scene as distributed. `detailed_nofur`: hair curves hidden.
  `simplified_fur`: the animal collection of `3D_digitaltwin_simplified_marmoset/{session}.blend` appended into the
  same scene (the Simplified-model scenes use a different environment build, so only the animal is taken).
- `background`: the same scene with the animal collections excluded (empty cage), one image per camera. The
  render silhouette is `|render - background| > 25`; the environment is identical in every Session, so the result
  does not depend on which Sessions are run (the published values used the per-camera median of the 2400 renders,
  which differs from the empty-cage render by 0.18 grey levels on average and never by more than 25).

## Real frames

The real image of a pair is `video_frame_index` of `frames.csv`, the frame the published values were computed with.
It was derived from the video timestamps and differs from the 3D keypoint NPZ correspondence
(`npz_video_frame_index` = `frame_num - frame_min`) by −2 / −1 / +1 frames in 24 / 192 / 600 of the 2400 pairs.
`extract_real_frames.py --frame-source npz` uses the NPZ correspondence instead (the values then no longer match
the published ones).

## Tolerance

`compute_metrics.py --verify` passes when, for every metric, at least 95% of the pairs are within the per-pair
tolerance **and** the mean over the pairs differs from the published mean by at most the mean tolerance:

| metric | per pair | mean |
|---|---|---|
| `psnr_full` | 0.05 dB | 0.01 dB |
| `ssim_full` | 0.002 | 0.0005 |
| `psnr_animal` | 0.1 dB | 0.02 dB |
| `ssim_animal` | 0.005 | 0.001 |
| `bbox_iou` | 0.05 | 0.005 |

Two sources of difference: Cycles path tracing is not bit-reproducible across GPUs and drivers, and the render
silhouette uses the empty-cage render instead of the median of all renders. The second one shifts a bbox edge by
a few pixels in a small share of pairs, which moves the Animal region and the IoU of those pairs.

Measured on this machine (Blender 5.2.2, 2026-11-18):

| check | pairs x models | full image | Animal region (share within tolerance; mean difference) | bbox IoU |
|---|---|---|---|---|
| renders behind the published values + empty-cage background | 7200 | identical | PSNR 99.3%, +0.002 dB; SSIM 99.2%, +0.0001 | 99.6%, −0.0004 |
| re-rendered from the Published dataset (2 Sessions, 3 models) | 48 | max 0.0001 dB / 0.00002 | PSNR 100%, +0.001 dB; SSIM 100%, +0.0001 | 100%, −0.0003 |

Re-rendered images match the renders behind the published values at >= 59.5 dB PSNR (median 73.8 dB, mean
absolute difference <= 0.04 grey levels).

## Legacy

The earlier whole-image pipeline (`run_pipeline.py`, `calculate_metrics.py`, ...) was replaced by this folder
on 2026-11-18 and is not part of the public repository.
