# 3D Digital Twin Marmoset

Vocabulary for the released marmoset 3D keypoint / digital-twin dataset and the code that builds it.

## Dataset copies

**Original dataset**:
The read-only source directory `3D_digital_twin_marmoset_data`; never modified.
_Avoid_: raw data, dataset (unqualified)

**Release tree**:
The directory `3D_digital_twin_marmoset_data_work_npz` holding the dataset exactly as it is to be published, in its restored folder layout; its structure is defined by `docs/dataset_layout.yaml`.
_Avoid_: dataset_work, work dir, NPZ dataset

**Upload set**:
The flat set of zips, manifests and `extract_dataset.py` (`3D_digital_twin_marmoset_data_upload_npz`) built from the Release tree and uploaded to the Published dataset.
_Avoid_: upload dir, Dataverse folder

**Published dataset**:
The Harvard Dataverse dataset doi:10.7910/DVN/HZVBH1.
_Avoid_: Dataverse (alone), public data

## 3D keypoints

**Raw keypoints**:
Triangulated 3D keypoints in millimetres in the camera-calibration frame, before any rotation or alignment (`kp3d_mm`).
_Avoid_: uncorrected, pre-correction pose, pickle keypoints

**Aligned keypoints**:
Raw keypoints after the fixed rotation and the per-session affine fitted by minimizing reprojection error, in Blender world coordinates and metres (`kp3d`). The only 3D keypoints released to users.
_Avoid_: corrected, 補正版, affine_reproj version, Blender keypoints

**Blender world coordinates**:
The coordinate frame of the digital-twin scene (metres, floor at z≈0); a frame, not a state of the keypoints.

## Frames and time

**3D frame**:
The index of a 3D keypoint sample within a session, 0..T-1 at 24 fps (`i_frame`).
_Avoid_: frame (unqualified), kp3d index

**Frame number**:
The imgstore `frame_number` of one camera's video matched to a 3D frame (`frame_num`); differs per camera.
_Avoid_: frame id

**Video frame index**:
Position of a frame counted from the start of one camera's video (`frame_num − frame_min`).

**Blend frame**:
The frame of a session's Blender scene timeline; equal to 3D frame + 1.

**Chunk frame index**:
Position of a frame within one video chunk mp4; the index used by the 2D keypoint JSON files.

## Recording

**Animal ID**:
The canonical identifier of an individual marmoset (e.g. `cj611`); used everywhere in the dataset.
_Avoid_: ID A/B/C (outside figures), subject

**Animal label**:
The display name `ID A`/`ID B`/`ID C` used only in figures, mapped one-to-one to Animal IDs.

**Session**:
One continuous recording of one animal, named `{Animal ID}_{YYYYMMDD}_{HHMMSS}`.
_Avoid_: recording, run, video

**Camera ID**:
The camera serial number (e.g. `23506214`); the canonical way to name one of the 8 cameras.
_Avoid_: sequential names such as Camera.1 or cam index (`Camera.{Camera ID}` as a Blender object name is fine)

**Camera label**:
The display name `Cam1`–`Cam8` used in figures, mapped one-to-one to Camera IDs.

## Body model

**Detailed model**:
The full rigged marmoset body built from the CT scan (`3dmarmoset.blend`).
_Avoid_: full model, normal model

**Simplified model**:
A lighter version of the marmoset body for faster rendering (`3dmarmoset_simplified.blend`).

**Fur**:
Whether the rendered body carries hair (with fur / without fur); a render option, not a separate model.

## Image comparison

**Full image**:
The whole camera image when comparing a real image with a render.
_Avoid_: raw (as in `psnr_raw`)

**Animal region**:
The padded rectangle around the animal (union of render silhouette bbox and 2D detection bbox) used for comparison.
_Avoid_: ROI, crop
