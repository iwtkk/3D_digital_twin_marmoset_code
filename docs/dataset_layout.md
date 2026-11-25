# Dataset layout

`/data/iwata/3D_digital_twin_marmoset_data_work_npz` (30425 files, 1058.7 GB, scanned 2026-11-25T21:25:14)

Generated from `docs/dataset_layout.yaml`. Edit the YAML, not this file.

## Tree

```
├── 3D_digital_twin_marmoset_data/  (17 patterns, 186.0 GB)
│   ├── 3D_digitaltwin_marmoset_blender.blend  ×1, 0.15 GB
│   ├── 3D_digitaltwin_marmoset_blender_files/  (1 patterns, 69.7 GB)
│   │   └── {session}.blend  ×222, 69.74 GB
│   ├── 3D_digitaltwin_marmoset_blender_simplified_3dmarmo.blend  ×1, 0.29 GB
│   ├── 3D_digitaltwin_marmoset_blender_simplified_marmoset/  (1 patterns, 104.5 GB)
│   │   └── {session}.blend  ×222, 104.46 GB
│   ├── rendered_environmental_images/  (10 patterns, 8.7 GB)
│   │   ├── README.md  ×1, 0.0 GB
│   │   ├── camera_summary.csv  ×1, 0.0 GB
│   │   ├── example/  (2 patterns, 0.0 GB)
│   │   │   ├── ssim_map_detailed_fur_full_Cam1_{session}_f{frame}.png  ×1, 0.0 GB
│   │   │   └── ssim_map_detailed_fur_full_Cam1_{session}_f{frame}_colorbar.png  ×1, 0.0 GB
│   │   ├── frames.csv  ×1, 0.0 GB
│   │   ├── metrics_animal_detailed_fur.json  ×1, 0.0 GB
│   │   ├── metrics_animal_detailed_nofur.json  ×1, 0.0 GB
│   │   ├── metrics_animal_simplified_fur.json  ×1, 0.0 GB
│   │   ├── pair_metrics.csv  ×1, 0.0 GB
│   │   └── render/  (1 patterns, 8.7 GB)
│   │       └── detailed_fur/  (1 patterns, 8.7 GB)
│   │           └── {session}/  (1 patterns, 8.7 GB)
│   │               └── Camera.{cam}/  (1 patterns, 8.7 GB)
│   │                   └── {frame}.png  ×2400, 8.68 GB
│   └── rendered_pseudo_egocentric_video_data/  (3 patterns, 2.7 GB)
│       ├── README.md  ×1, 0.0 GB
│       ├── {session}_Eqdst.mp4  ×1, 1.26 GB
│       └── {session}_Pinhole.mp4  ×1, 1.45 GB
├── 3D_environmental_data/  (1 patterns, 0.6 GB)
│   └── environment_measurement.blend  ×1, 0.62 GB
├── 3D_marmoset_body_data/  (8 patterns, 1.3 GB)
│   ├── 3D_marmoset_body_Blender/  (2 patterns, 0.4 GB)
│   │   ├── 3dmarmoset.blend  ×1, 0.18 GB
│   │   └── 3dmarmoset_simplified.blend  ×1, 0.21 GB
│   └── CT_scan_data/  (6 patterns, 0.9 GB)
│       ├── CT_scan_data.nii  ×1, 0.7 GB
│       ├── Marmoset_surface.seg.nrrd  ×1, 0.0 GB
│       ├── Marmoset_surface.stl  ×1, 0.0 GB
│       ├── Segmentation.nii.gz  ×1, 0.0 GB
│       ├── Segmentation.seg.nrrd  ×1, 0.0 GB
│       └── Segmentation.stl  ×1, 0.17 GB
└── marmoset_keypoint_data/  (40 patterns, 870.8 GB)
    ├── 2D_temporal_keypoint_data/  (1 patterns, 76.4 GB)
    │   └── {session}/  (1 patterns, 76.4 GB)
    │       └── {session}_{cam}_{chunk}.json  ×6512, 76.44 GB
    ├── 3D_temporal_keypoint_data/  (1 patterns, 9.4 GB)
    │   └── {session}.npz  ×222, 9.39 GB
    ├── trained_model/  (3 patterns, 1.6 GB)
    │   ├── detection.pth  ×1, 0.65 GB
    │   ├── id.pth  ×1, 0.19 GB
    │   └── pose_marmo20_best_epoch.pth  ×1, 0.76 GB
    ├── training_data/  (31 patterns, 7.4 GB)
    │   ├── 3dkp_gt/  (25 patterns, 3.1 GB)
    │   │   ├── README.txt  ×1, 0.0 GB
    │   │   ├── fig2e_matched_keypoints/  (2 patterns, 0.0 GB)
    │   │   │   ├── SingleAnimal_20.mat  ×1, 0.0 GB
    │   │   │   └── SingleAnimal_20.npz  ×1, 0.0 GB
    │   │   ├── images/  (1 patterns, 3.1 GB)
    │   │   │   └── {name}.png  ×800, 3.12 GB
    │   │   └── labeled_frames/  (21 patterns, 0.0 GB)
    │   │       ├── valbin1_nanim1_nfrm100/  (7 patterns, 0.0 GB)
    │   │       │   ├── calib/  (4 patterns, 0.0 GB)
    │   │       │   │   ├── cam_extrinsic.h5  ×1, 0.0 GB
    │   │       │   │   ├── cam_extrinsic_optim.h5  ×1, 0.0 GB
    │   │       │   │   ├── cam_intrinsic.h5  ×1, 0.0 GB
    │   │       │   │   └── config.yaml  ×1, 0.0 GB
    │   │       │   ├── framedata.csv  ×1, 0.0 GB
    │   │       │   ├── metadata.yaml  ×1, 0.0 GB
    │   │       │   └── {name}.json  ×100, 0.0 GB
    │   │       ├── valbin2_nanim2_nfrm55/  (7 patterns, 0.0 GB)
    │   │       │   ├── calib/  (4 patterns, 0.0 GB)
    │   │       │   │   ├── cam_extrinsic.h5  ×1, 0.0 GB
    │   │       │   │   ├── cam_extrinsic_optim.h5  ×1, 0.0 GB
    │   │       │   │   ├── cam_intrinsic.h5  ×1, 0.0 GB
    │   │       │   │   └── config.yaml  ×1, 0.0 GB
    │   │       │   ├── framedata.csv  ×1, 0.0 GB
    │   │       │   ├── metadata.yaml  ×1, 0.0 GB
    │   │       │   └── {name}.json  ×50, 0.0 GB
    │   │       └── valbin3_nanim3_nfrm35/  (7 patterns, 0.0 GB)
    │   │           ├── calib/  (4 patterns, 0.0 GB)
    │   │           │   ├── cam_extrinsic.h5  ×1, 0.0 GB
    │   │           │   ├── cam_extrinsic_optim.h5  ×1, 0.0 GB
    │   │           │   ├── cam_intrinsic.h5  ×1, 0.0 GB
    │   │           │   └── config.yaml  ×1, 0.0 GB
    │   │           ├── framedata.csv  ×1, 0.0 GB
    │   │           ├── metadata.yaml  ×1, 0.0 GB
    │   │           └── {name}.json  ×35, 0.0 GB
    │   ├── annotations/  (3 patterns, 0.0 GB)
    │   │   ├── single.json  ×1, 0.0 GB
    │   │   ├── single_test.json  ×1, 0.0 GB
    │   │   └── single_train.json  ×1, 0.0 GB
    │   ├── images/  (1 patterns, 2.9 GB)
    │   │   └── {name}.png  ×3382, 2.87 GB
    │   └── pretrained/  (2 patterns, 1.4 GB)
    │       ├── tk_hrnet_w48_coco_384x288_dark_epoch_70.pth  ×1, 0.76 GB
    │       └── tk_yolox_l_8x8_300e_coco_V0p10_best_bbox_mAP_epoch_282.pth  ×1, 0.65 GB
    └── video_data/  (4 patterns, 775.9 GB)
        └── {session}.{cam}/  (4 patterns, 775.9 GB)
            ├── metadata.yaml  ×1776, 0.0 GB
            ├── {chunk}.extra_data.h5  ×1628, 150.56 GB
            ├── {chunk}.mp4  ×6512, 624.34 GB
            └── {chunk}.npz  ×6512, 1.02 GB
```

## Entries

| Path | Files | GB | Description |
|---|---:|---:|---|
| `3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender.blend` | 1 | 0.15 | Base digital twin scene (environment + marmoset model) |
| `3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_files/{session}.blend` | 222 | 69.74 | Digital twin per session (Detailed model), posed with the Aligned keypoints; Blend frame = 3D frame + 1 |
| `3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_simplified_3dmarmo.blend` | 1 | 0.29 | Base scene with the simplified marmoset model |
| `3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_simplified_marmoset/{session}.blend` | 222 | 104.46 | Digital twin per session (Simplified model), posed with the Aligned keypoints; Blend frame = 3D frame + 1 |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/README.md` | 1 | 0.0 | Description of the Fig. 8a image-similarity files (regions, keys, real frames) |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/camera_summary.csv` | 1 | 0.0 | Mean / std of pair_metrics.csv per Camera ID x body model |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/example/ssim_map_detailed_fur_full_Cam1_{session}_f{frame}.png` | 1 | 0.0 | Example full-image SSIM map of one pair (detailed_fur, ID A, Cam 1), grayscale 255 = SSIM 1 |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/example/ssim_map_detailed_fur_full_Cam1_{session}_f{frame}_colorbar.png` | 1 | 0.0 | The example SSIM map with a colorbar and its mean |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/frames.csv` | 1 | 0.0 | Fig. 8a pairs: Session, Camera ID, 3D frame, Blend frame, Video frame index of the real image |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/metrics_animal_detailed_fur.json` | 1 | 0.0 | Animal-region PSNR/SSIM per pair, Detailed model with fur (Aligned pose, Fig. 8a) |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/metrics_animal_detailed_nofur.json` | 1 | 0.0 | Animal-region PSNR/SSIM per pair, Detailed model without fur (Aligned pose) |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/metrics_animal_simplified_fur.json` | 1 | 0.0 | Animal-region PSNR/SSIM per pair, Simplified model with fur (Aligned pose) |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/pair_metrics.csv` | 1 | 0.0 | Full-image and Animal-region PSNR/SSIM and bbox IoU per pair x body model |
| `3D_digital_twin_marmoset_data/rendered_environmental_images/render/detailed_fur/{session}/Camera.{cam}/{frame}.png` | 2400 | 8.68 | Render of every Fig. 8a pair, Detailed model with fur (Aligned pose, Blend frame = 3D frame + 1, 2048x1536 RGB) |
| `3D_digital_twin_marmoset_data/rendered_pseudo_egocentric_video_data/README.md` | 1 | 0.0 | Description of the pseudo first-person videos (cameras, frame mapping, encoding) |
| `3D_digital_twin_marmoset_data/rendered_pseudo_egocentric_video_data/{session}_Eqdst.mp4` | 1 | 1.26 | Pseudo first-person video from the eyes, equidistant fisheye 140 deg (Camera_Eqdst_70), 1000x1000, 24 fps |
| `3D_digital_twin_marmoset_data/rendered_pseudo_egocentric_video_data/{session}_Pinhole.mp4` | 1 | 1.45 | Pseudo first-person video from the eyes, pinhole 82 deg (Camera_Pinhole_orig), 1000x1000, 24 fps |
| `3D_environmental_data/environment_measurement.blend` | 1 | 0.62 | Home-cage environment model (with measured dimensions) |
| `3D_marmoset_body_data/3D_marmoset_body_Blender/3dmarmoset.blend` | 1 | 0.18 | Rigged marmoset model |
| `3D_marmoset_body_data/3D_marmoset_body_Blender/3dmarmoset_simplified.blend` | 1 | 0.21 | Simplified marmoset model |
| `3D_marmoset_body_data/CT_scan_data/CT_scan_data.nii` | 1 | 0.7 | CT volume |
| `3D_marmoset_body_data/CT_scan_data/Marmoset_surface.seg.nrrd` | 1 | 0.0 | Body surface segmentation (3D Slicer) |
| `3D_marmoset_body_data/CT_scan_data/Marmoset_surface.stl` | 1 | 0.0 | Body surface mesh |
| `3D_marmoset_body_data/CT_scan_data/Segmentation.nii.gz` | 1 | 0.0 | Segmentation (NIfTI) |
| `3D_marmoset_body_data/CT_scan_data/Segmentation.seg.nrrd` | 1 | 0.0 | Segmentation (3D Slicer) |
| `3D_marmoset_body_data/CT_scan_data/Segmentation.stl` | 1 | 0.17 | Segmentation mesh |
| `marmoset_keypoint_data/2D_temporal_keypoint_data/{session}/{session}_{cam}_{chunk}.json` | 6512 | 76.44 | 2D detections (bbox + 20 keypoints), one per video chunk |
| `marmoset_keypoint_data/3D_temporal_keypoint_data/{session}.npz` | 222 | 9.39 | 3D keypoints per session: Aligned keypoints (Blender world coordinates, m), Raw keypoints (mm), score, reprojection error, frame_num; see docs/data_format.md |
| `marmoset_keypoint_data/trained_model/detection.pth` | 1 | 0.65 | Bounding box detector (YOLOX) |
| `marmoset_keypoint_data/trained_model/id.pth` | 1 | 0.19 | Individual identification model |
| `marmoset_keypoint_data/trained_model/pose_marmo20_best_epoch.pth` | 1 | 0.76 | 20-keypoint pose estimator (HRNet-W48 + DARK) |
| `marmoset_keypoint_data/training_data/3dkp_gt/README.txt` | 1 | 0.0 | Description of 3dkp_gt (file formats, shapes, units) |
| `marmoset_keypoint_data/training_data/3dkp_gt/fig2e_matched_keypoints/SingleAnimal_20.mat` | 1 | 0.0 | Fig. 2E matched GT/predicted 3D keypoints (1 animal, 100 frames, 20 keypoints), original output |
| `marmoset_keypoint_data/training_data/3dkp_gt/fig2e_matched_keypoints/SingleAnimal_20.npz` | 1 | 0.0 | Fig. 2E matched GT/predicted 3D keypoints (1 animal, 100 frames, 20 keypoints) with session and frame |
| `marmoset_keypoint_data/training_data/3dkp_gt/images/{name}.png` | 800 | 3.12 | Camera images of the labeled frames ({frame_name}.{camera_id}.png; 1 animal, 100 frames x 8 cameras), re-extracted from video_data |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin1_nanim1_nfrm100/calib/cam_extrinsic.h5` | 1 | 0.0 | Camera calibration used for labeling (1 animal set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin1_nanim1_nfrm100/calib/cam_extrinsic_optim.h5` | 1 | 0.0 | Camera calibration used for labeling (1 animal set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin1_nanim1_nfrm100/calib/cam_intrinsic.h5` | 1 | 0.0 | Camera calibration used for labeling (1 animal set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin1_nanim1_nfrm100/calib/config.yaml` | 1 | 0.0 | Camera calibration used for labeling (1 animal set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin1_nanim1_nfrm100/framedata.csv` | 1 | 0.0 | Labeled frame -> session and video frame per camera (1 animal) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin1_nanim1_nfrm100/metadata.yaml` | 1 | 0.0 | Labeling-tool metadata (1 animal) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin1_nanim1_nfrm100/{name}.json` | 100 | 0.0 | Manually labeled 2D (8 cameras) and 3D keypoints, mm (1 animal) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin2_nanim2_nfrm55/calib/cam_extrinsic.h5` | 1 | 0.0 | Camera calibration used for labeling (2 animals set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin2_nanim2_nfrm55/calib/cam_extrinsic_optim.h5` | 1 | 0.0 | Camera calibration used for labeling (2 animals set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin2_nanim2_nfrm55/calib/cam_intrinsic.h5` | 1 | 0.0 | Camera calibration used for labeling (2 animals set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin2_nanim2_nfrm55/calib/config.yaml` | 1 | 0.0 | Camera calibration used for labeling (2 animals set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin2_nanim2_nfrm55/framedata.csv` | 1 | 0.0 | Labeled frame -> session and video frame per camera (2 animals) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin2_nanim2_nfrm55/metadata.yaml` | 1 | 0.0 | Labeling-tool metadata (2 animals) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin2_nanim2_nfrm55/{name}.json` | 50 | 0.0 | Manually labeled 2D (8 cameras) and 3D keypoints, mm (2 animals) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin3_nanim3_nfrm35/calib/cam_extrinsic.h5` | 1 | 0.0 | Camera calibration used for labeling (3 animals set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin3_nanim3_nfrm35/calib/cam_extrinsic_optim.h5` | 1 | 0.0 | Camera calibration used for labeling (3 animals set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin3_nanim3_nfrm35/calib/cam_intrinsic.h5` | 1 | 0.0 | Camera calibration used for labeling (3 animals set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin3_nanim3_nfrm35/calib/config.yaml` | 1 | 0.0 | Camera calibration used for labeling (3 animals set; identical in the three sets) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin3_nanim3_nfrm35/framedata.csv` | 1 | 0.0 | Labeled frame -> session and video frame per camera (3 animals) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin3_nanim3_nfrm35/metadata.yaml` | 1 | 0.0 | Labeling-tool metadata (3 animals) |
| `marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames/valbin3_nanim3_nfrm35/{name}.json` | 35 | 0.0 | Manually labeled 2D (8 cameras) and 3D keypoints, mm (3 animals) |
| `marmoset_keypoint_data/training_data/annotations/single.json` | 1 | 0.0 | COCO-format annotations (all) |
| `marmoset_keypoint_data/training_data/annotations/single_test.json` | 1 | 0.0 | COCO-format annotations (test) |
| `marmoset_keypoint_data/training_data/annotations/single_train.json` | 1 | 0.0 | COCO-format annotations (train) |
| `marmoset_keypoint_data/training_data/images/{name}.png` | 3382 | 2.87 | Training images |
| `marmoset_keypoint_data/training_data/pretrained/tk_hrnet_w48_coco_384x288_dark_epoch_70.pth` | 1 | 0.76 | Pretrained weights of the pose estimator |
| `marmoset_keypoint_data/training_data/pretrained/tk_yolox_l_8x8_300e_coco_V0p10_best_bbox_mAP_epoch_282.pth` | 1 | 0.65 | Pretrained weights of the detector |
| `marmoset_keypoint_data/video_data/{session}.{cam}/metadata.yaml` | 1776 | 0.0 | imgstore metadata |
| `marmoset_keypoint_data/video_data/{session}.{cam}/{chunk}.extra_data.h5` | 1628 | 150.56 | imgstore extra data (timestamps etc.) |
| `marmoset_keypoint_data/video_data/{session}.{cam}/{chunk}.mp4` | 6512 | 624.34 | Video chunk (imgstore, 24 fps) |
| `marmoset_keypoint_data/video_data/{session}.{cam}/{chunk}.npz` | 6512 | 1.02 | imgstore frame_number / frame_time |
