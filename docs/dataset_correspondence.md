# Dataset Correspondence

This document maps dataset folders (Zenodo/AWS) to the scripts in this repository.

## Dataset structure

```
3D_digital_twin_marmoset/         (root on Zenodo/AWS)
├── 3D_marmoset_body_data/
│   └── marmoset_body_blender/    # Rigged marmoset .blend file
├── marmoset_keypoint_data/
│   ├── training_data/            # COCO-format annotation + images
│   ├── video_data/               # Multi-camera synchronized videos
│   ├── 2D_temporal_keypoint/     # Filtered 2D keypoints (kp2d_f.pickle)
│   └── 3D_temporal_keypoint/     # Reconstructed 3D keypoints (kp3d*.pickle)
├── 3D_environmental_data/
│   └── environment_model_blender/ # Cage 3D model (.blend)
└── 3D_digital_twin_data/
    └── 3D_digitaltwin_marmoset_blender/ # Integrated scene (.blend)
```

## Dataset folder → script mapping

| Dataset folder | Script(s) | Description |
|---|---|---|
| `marmoset_keypoint_data/training_data` | MMDetection / MMPose training | Train bbox detector and pose estimator |
| `marmoset_keypoint_data/video_data` | `scripts/01_prepare_video/` | Input videos for full pipeline |
| `marmoset_keypoint_data/video_data` | `scripts/02_detect_bbox/` | Bounding box detection input |
| `marmoset_keypoint_data/video_data` | `scripts/03_estimate_2d_keypoints/` | 2D keypoint estimation input |
| `marmoset_keypoint_data/2D_temporal_keypoint` | `scripts/04_reconstruct_3d_keypoints/` | 3D reconstruction input |
| `marmoset_keypoint_data/3D_temporal_keypoint` | `animate/BlenderSync/blender_animation.py` | Blender animation input |
| `marmoset_keypoint_data/3D_temporal_keypoint` | `repr_error/` | Reprojection error analysis |
| `3D_environmental_data/environment_model_blender` | `scripts/06_measure_environment/` | Cage model measurement |
| `3D_digital_twin_data/3D_digitaltwin_marmoset_blender` | `animate/BlenderSync/allcamrender.py` | Pseudo-view rendering |
| `3D_marmoset_body_data/marmoset_body_blender` | `animate/BlenderSync/blender_animation.py` | Base rigged model |

## Minimal data for testing

To run a minimal test without downloading the full dataset, use the sample files in `examples/minimal_sample/`:

```
examples/minimal_sample/
├── sample_calibration.toml     # Calibration for 2 cameras
├── sample_kp2d_f.pickle        # 10 frames, 2 cameras, 1 individual
├── sample_kp3d.pickle          # Corresponding 3D keypoints
└── README.md
```

## Downloading the full dataset

```bash
# Download only the sample subset
python scripts/download_data.py --subset sample

# Download trained model weights only
python scripts/download_data.py --subset trained_model

# Download everything
python scripts/download_data.py --subset all
```
