# Dataset overview (for Data Records)

Overview of the four top-level folders of the released dataset
(Harvard Dataverse doi:10.7910/DVN/HZVBH1; restored layout = `/data/iwata/3D_digital_twin_marmoset_data_work_npz`).
Counts and sizes are from `docs/dataset_layout.yaml` (scan 2026-11-25: 30,425 files, 1,058.7 GB; sizes in decimal GB).
Figure / section numbers marked `[?]` still have to be filled in from the manuscript.

## Table

| Top-level folder | Subfolder / file | Contents | File format | Figure / section | Approx. size |
|---|---|---|---|---|---|
| `marmoset_keypoint_data/` | `video_data/` | Synchronized videos from 8 cameras, 222 sessions (`{session}.{camera}/`), split into chunks (imgstore, 24 fps, 2048×1536) with per-frame numbers and timestamps | `.mp4`, `.npz`, `.h5`, `.yaml` | [?] | 776 GB (16,428 files) |
| | `training_data/` | Training set for the detector and pose estimator (3,382 images, COCO annotations, pretrained weights); manually labeled 3D ground truth for 1/2/3 animals (`3dkp_gt/`) and matched GT/predicted keypoints | `.png`, `.json`, `.pth`, `.h5`, `.csv`, `.yaml`, `.npz`, `.mat` | Fig. 2E (`3dkp_gt/`), [?] | 4.3 GB (3,597 files) |
| | `trained_model/` | Trained bounding-box detector (YOLOX), 20-keypoint pose estimator (HRNet-W48 + DARK) and individual identification model | `.pth` | [?] | 1.6 GB (3 files) |
| | `2D_temporal_keypoint_data/` | 2D detections per video chunk (bounding box, 20 keypoints with scores, ID label) | `.json` | [?] | 76 GB (6,512 files) |
| | `3D_temporal_keypoint_data/` | 3D keypoints per session: Aligned keypoints (Blender world coordinates) and Raw keypoints (mm), confidence, reprojection error, head orientation, frame numbers | `.npz` | [?] | 9.4 GB (222 files) |
| `3D_marmoset_body/` | `3dmarmoset.blend`, `3dmarmoset_simplified.blend` | Rigged marmoset body model (IK rig) and its simplified version | `.blend` | Fig. 3, [?] | 0.4 GB (2 files) |
| | `CT_scan_data/` | CT volume of a marmoset, body-surface and segmentation masks and meshes used to build the model | `.nii`, `.nii.gz`, `.seg.nrrd`, `.stl` | [?] | 0.9 GB (6 files) |
| `3D_environmental_body/` | `environment_measurement.blend` | Home-cage environment model with measured dimensions | `.blend` | Table 3, [?] | 0.6 GB (1 file) |
| `3D_digital_twin_data/` | `3D_digitaltwin_marmoset_blender.blend`, `..._simplified_3dmarmo.blend` | Base digital-twin scenes (environment + marmoset model; full and simplified model) | `.blend` | [?] | 0.4 GB (2 files) |
| | `3D_digitaltwin_marmoset_blender_files/` | Digital twin per session (Detailed model), posed with the Aligned keypoints | `.blend` | [?] | 70 GB (222 files) |
| | `3D_digitaltwin_simplified_marmoset/` | Digital twin per session (Simplified model), posed with the Aligned keypoints | `.blend` | [?] | 104 GB (222 files) |
| | `rendered_environmental_images/` | PSNR/SSIM (Full image, Animal region) and bbox IoU between real and rendered images for 2,400 pairs and 3 body models, with the frame list, the 2,400 renders of the Detailed model with fur and an example full-image SSIM map; all renders are reproducible with the code repository | `.png`, `.json`, `.csv`, `.md` | Fig. 8a | 8.7 GB (2,409 files) |
| | `rendered_pseudo_egocentric_video_data/` | Pseudo first-person videos from the eyes of the animal (session cj611_20230226_110000; equidistant fisheye 140 deg and pinhole 82 deg) | `.mp4`, `.md` | [?] | 2.7 GB (3 files) |

Distribution: every folder is split into zips under 2.5 GB with a manifest CSV per prefix; `extract_dataset.py`
(at the top of the dataset) restores the layout above.

## LaTeX

See `docs/dataset_overview.tex` (booktabs; Figure / Section column is provisional).
