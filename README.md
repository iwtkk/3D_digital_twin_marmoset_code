# 3D Digital Twin Marmoset Code

This repository provides code for reconstructing common marmoset body–environment interactions in a 3D digital-twin framework.

The code accompanies the dataset and manuscript:

> **A multimodal dataset for reconstructing common marmoset body–environment interactions in a 3D digital-twin framework**
> Koki Iwata et al.

---

## What this repository provides

| Pipeline step | What this code does | Corresponding paper section |
|---|---|---|
| 2D bounding box detection | MMDetection-based bbox estimation | Sec. 2.3 |
| 2D keypoint estimation | MMPose-based 20-point keypoint estimation | Sec. 2.3 |
| 3D keypoint reconstruction | Multi-view triangulation via anipose | Sec. 2.4 |
| Temporal smoothing | Spatiotemporal filtering of 3D keypoints | Sec. 2.4 |
| Environment measurement | Python scripts for 3D cage model dimensions | Sec. 2.2 |
| Blender pose application | Apply 3D keypoints to rigged marmoset model | Sec. 2.5 |
| Pseudo-view rendering | Render synthetic first-person-view video | Sec. 2.5 |

---

## Dataset

The associated dataset is available at:

- **Zenodo**: [link to be added upon publication]
- **AWS**: [link to be added upon publication]

The dataset consists of four parts:

| Dataset folder | Contents |
|---|---|
| `3D_marmoset_body_data` | 3D body model files (Blender) |
| `marmoset_keypoint_data` | Videos, 2D/3D keypoints, training data |
| `3D_environmental_data` | 3D cage/environment model (Blender) |
| `3D_digital_twin_data` | Integrated digital twin Blender scenes |

See [docs/dataset_correspondence.md](docs/dataset_correspondence.md) for which script uses which dataset folder.

---

## Directory structure

```
3D_digital_twin_marmoset_code/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
│
├── scripts/                        # Pipeline scripts (run in order)
│   ├── 00_setup/                   # Environment setup and data link creation
│   ├── 01_build_3d_marmoset/       # CT → segmentation → rigging (README only)
│   ├── 02_train/                   # MMDetection/MMPose training configs
│   │   └── configs/
│   │       ├── detection/
│   │       └── pose_estimation/
│   ├── 03_run_marmo3dpose/         # 2D inference + 3D reconstruction
│   │   └── marmo3Dpose/            # submodule (Apache 2.0)
│   ├── 04_build_3d_environment/    # Cage model construction (README only)
│   ├── 05_create_digital_twin/     # Apply pose to Blender, render
│   └── 06_validation/              # Technical validation
│       ├── keypoints/              # Estimated 2D/3D keypoints
│       ├── reprojection_error/     # Reprojection error
│       ├── environment_accuracy/   # Geometric accuracy of the 3D environment model
│       └── consistency/            # Consistency of the reconstructed experimental environment (SSIM/PSNR)
│
├── config/                         # Configuration files
│   └── marmo_cj425m/               # Camera calibration for session cj425m
│
└── docs/                           # Documentation
    ├── keypoint_definition.md
    ├── bone_mapping.md
    ├── camera_mapping.md
    ├── data_format.md
    ├── blender_usage.md
    └── install_mmdet_mmpose.md
```

---

## Installation

### Tested environment

| Item | Version |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10.12 |
| CUDA | 11.6 |

### External software

| Software | Version | Purpose |
|---|---|---|
| Blender | 4.5.0 | 3D pose application and rendering |
| MATLAB | R2023b | Camera calibration (optional) |
| 3D Slicer | 4.11 | Environment model measurement |
| SlicerRT toolkit | - | Used with 3D Slicer |

---

## Quick start

Paths below are relative to the repository root unless a step says `cd`.

### Step 1 – Set up dataset and environment

The dataset is distributed in two places with the same folder layout. They differ only in the
format of the 3D keypoints (`marmoset_keypoint_data/3D_temporal_keypoint_data/`):

| Source | 3D keypoints | Use |
|---|---|---|
| AWS S3 `s3://digitaltwin-marmoset/` | `{session}/kp3d_fxdJointLen.pickle` (raw triangulation) | videos, models, Step 2 |
| Harvard Dataverse (doi:10.7910/DVN/HZVBH1) | `{session}.npz` (aligned to the digital twin, ready for Blender) | Step 3 directly |

```bash
# Download the dataset from AWS S3 (no sign-in required)
# (no aws command: pip install awscli)
aws s3 --no-sign-request cp --recursive s3://digitaltwin-marmoset/ ./3D_digital_twin_marmoset_data
# (Harvard Dataverse: download the zip files and restore the layout with
#  python extract_dataset.py --dst ./3D_digital_twin_marmoset_data --gunzip-blend)

# Clone the repository
git clone --recurse-submodules https://github.com/iwtkk/3D_digital_twin_marmoset_code.git
cd 3D_digital_twin_marmoset_code
```

Then set up the environment in one of two ways.

**Option A – virtual environment**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash scripts/00_setup/install.sh
bash scripts/00_setup/setup_data_links.sh /path/to/3D_digital_twin_marmoset_data
```

**Option B – Docker** (NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/))

```bash
# Build (mmcv-full is compiled from source: tens of minutes).
# INSTALL_BLENDER=1 adds Blender (5.2.2) for Step 3; omit it if Blender runs elsewhere.
docker build --build-arg INSTALL_BLENDER=1 -t marmoset-dt .
# Ubuntu 22.04 / CUDA 11.7 variant: docker build -f Dockerfile.ubuntu22 --build-arg INSTALL_BLENDER=1 -t marmoset-dt .

# Start a shell with the repository at /workspace and the dataset at /data
docker run --gpus all -it --rm --shm-size=8g \
  -v "$PWD":/workspace \
  -v /path/to/3D_digital_twin_marmoset_data:/data \
  marmoset-dt

# Inside the container (the Python environment is already active)
bash scripts/00_setup/setup_data_links.sh /data
```

Run Steps 2 and 3 below inside the container. The repository is bind-mounted, so the outputs stay in
`outputs/` on the host. The links made by `setup_data_links.sh` point to `/data/...` and only resolve
inside the container; run it again with the host path when working outside.
(Rootless podman without `/etc/subuid`: add `--build-arg ROOTLESS_NO_SUBUID=1` to the build.)

`setup_data_links.sh` creates these links:

| Link | Target in the dataset |
|---|---|
| `vid` | `marmoset_keypoint_data/video_data/` |
| `outputs/2d` | `marmoset_keypoint_data/2D_temporal_keypoint_data/` |
| `outputs/3d` | `marmoset_keypoint_data/3D_temporal_keypoint_data/` |
| `outputs/digitaltwin` | `3D_digital_twin_data/` |

Session names are `{animal}_{YYYYMMDD}_{HHMMSS}` (e.g. `cj611_20230227_100000`); the video of each
camera is in `vid/{session}.{camera_id}/`.

### Step 2 – 2D + 3D pose estimation (marmo3Dpose)

Steps 2-1 to 2-4 produce from the videos the same NPZ files as those on Harvard Dataverse.
If you use the Dataverse NPZ files, go to Step 3.

**2-1. Link the trained models** (once). marmo3Dpose loads them from `weight/` under its own directory:

```bash
M=scripts/03_run_marmo3dpose/marmo3Dpose
mkdir -p $M/weight
ln -sfn "$(readlink -f vid)/../trained_model/detection.pth"               $M/weight/detection.pth
ln -sfn "$(readlink -f vid)/../trained_model/id.pth"                      $M/weight/id.pth
ln -sfn "$(readlink -f vid)/../trained_model/pose_marmo20_best_epoch.pth" $M/weight/marmo20_pose.pth
```

**2-2. Choose the sessions.** `all.sh` processes every session found in the video directory it is given.
To process only some sessions, link them into a separate directory:

```bash
mkdir -p outputs/vid_selected
ln -sf "$PWD"/vid/cj611_20230227_100000.* outputs/vid_selected/
```

(Pass `$PWD/vid` instead of `outputs/vid_selected` below to process all sessions.)

**2-3. Run 2D inference → 3D reconstruction → visualization.** `all.sh` takes positional arguments
`<video dir> <calibration config> <output dir> <device>` and must be run from the marmo3Dpose directory:

```bash
ROOT=$PWD
cd scripts/03_run_marmo3dpose/marmo3Dpose
NUMBA_NUM_THREADS=1 bash all.sh \
  $ROOT/outputs/vid_selected \
  ./calib/marmo_cj425m/config.yaml \
  $ROOT/outputs/marmo3d \
  cuda:0
cd $ROOT
```

`NUMBA_NUM_THREADS=1` keeps the triangulation from slowing down on many-core machines.
To use several GPUs, split the sessions into several directories in 2-2 and run one `all.sh` per GPU
(`cuda:0`, `cuda:1`, ...) with the same output directory.

Outputs (in `outputs/marmo3d/`):

| Path | Content |
|---|---|
| `2d_v0p8_Dark_fix_20_all/{session}/{session}_{camera_id}_{chunk}.json` | 2D detections and keypoints per video chunk |
| `3d_v0p8_dark_fix_20_all/{session}/kp3d_fxdJointLen.pickle` | 3D keypoints (same format as `outputs/3d` from S3) |
| `3d_v0p8_dark_fix_20_all/{session}/{camera_id}/frame_num.npy` | video frame of each 3D frame |
| `video/3d_v0p8_dark_fix_20_all/` | 3D keypoints drawn on camera 6 |

**2-4. Align the 3D keypoints to the digital twin and write NPZ.** For each session an affine transform
from the triangulation frame to the Blender scene is fitted by minimising the reprojection error between
the real cameras and the Blender cameras, and the result is written as `{session}.npz`
(same content as the Dataverse NPZ; see `scripts/05_create_digital_twin/alignment_3D_keypoint/`).

```bash
IN=outputs/marmo3d/3d_v0p8_dark_fix_20_all   # or outputs/3d for the pickle files from S3
python scripts/05_create_digital_twin/alignment_3D_keypoint/run/affine_align.py \
  --all --input-dir $IN --output-dir outputs/alignment --jobs 8
python scripts/05_create_digital_twin/alignment_3D_keypoint/run/export_aligned_npz.py \
  --all --input-dir $IN --affine-log outputs/alignment/affine_log.npz --output-dir outputs/npz --jobs 8
```

Use `--session <name>` instead of `--all` for a single session. `outputs/npz/{session}.npz` contains:

| Key | Shape | Content |
|---|---|---|
| `kp3d` | (T, 23, 3) | Blender world coordinates [m] (20 keypoints + shoulder_mid, base, center_eye) |
| `kp_names` | (23,) | keypoint names |
| `head_euler_xyz` | (T, 3) | head orientation, Euler XYZ [rad] |
| `kp3d_mm`, `kp3d_score`, `kp3d_err` | (T, 20, ...) | raw triangulation [mm], 2D score, reprojection error [px] |
| `affine_A`, `affine_t`, `rot` | | `kp3d[:, :20] = ((kp3d_mm @ rot.T) / 1000) @ affine_A.T + affine_t` |
| `frame_num`, `frame_min`, `camera_id` | (8, T), (8,), (8,) | video frame of each kp3d frame per camera |

### Step 3 – Apply pose to Blender and render

Blender (checked with 5.2.2; the released `.blend` files are saved by Blender 5.0) must be on `PATH`, or set
`BLENDER=/path/to/blender`. The Docker image includes it when built with `--build-arg INSTALL_BLENDER=1`.
The input is a directory of `{session}.npz`: `outputs/npz` from Step 2, or `outputs/3d` if the dataset
was restored from Harvard Dataverse.

```bash
# Keyframe the pose on a copy of the digital twin: outputs/blender/work_{session}.blend
bash scripts/05_create_digital_twin/run_animate_.sh \
  --blend outputs/digitaltwin/3D_digitaltwin_marmoset_blender.blend \
  --npz-dir outputs/npz \
  --output-dir outputs/blender \
  --all

# Render (outputs/render/{session}/{camera}/{session}_{camera}_{frame:05d}.png)
bash scripts/05_create_digital_twin/run_render.sh \
  --npz-dir outputs/npz \
  --blend-dir outputs/blender \
  --all-csv --frame-start 1 --frame-end 240 \
  --gpu 0
```

- `--session <name>` (repeatable) in `run_animate_.sh` limits the sessions; without `--all` only the first one is processed.
- Blender output of each session is saved in `outputs/blender/logs/work_{session}.log`.
- Cameras: by default `Camera_Eqdst_70` and `Camera_Pinhole_orig` (pseudo first-person views that follow
  the head). Others can be chosen with `--cam` (repeatable): `Camera_Eqdst_100`, `Camera_Pinhole_x2`, and
  `Camera.{camera_id}` (the 8 real cameras).
- Blend frame `i + 1` corresponds to `kp3d[i]`. Use `--all-frames` instead of `--frame-start/--frame-end`
  to render the whole session.

---

## Keypoint definition

20 keypoints are defined for each individual. See [docs/keypoint_definition.md](docs/keypoint_definition.md) for full details.

| Index | Name | Index | Name |
|---|---|---|---|
| 0 | nose | 10 | R wrist |
| 1 | L eye | 11 | L hip |
| 2 | R eye | 12 | R hip |
| 3 | L ear | 13 | L knee |
| 4 | R ear | 14 | R knee |
| 5 | L shoulder | 15 | L ankle |
| 6 | R shoulder | 16 | R ankle |
| 7 | L elbow | 17 | backbone peak |
| 8 | R elbow | 18 | tail_1 |
| 9 | L wrist | 19 | tail_2 |

Array shapes:
- 2D keypoints: `(n_frame, n_keypoint, n_individual, 3, n_camera)` — last dim is `[x, y, confidence]`
- 3D keypoints: `(n_frame, n_keypoint, 3)` — `[X, Y, Z]` in mm

---

## Camera setup

8 synchronized cameras were used. See [docs/camera_mapping.md](docs/camera_mapping.md) for full details.

| Camera name | Serial ID |
|---|---|
| Cam 1 | 23506226 |
| Cam 2 | 23511607 |
| Cam 3 | 23506214 |
| Cam 4 | 23506236 |
| Cam 5 | 23511614 |
| Cam 6 | 23506237 |
| Cam 7 | 23511613 |
| Cam 8 | 23506239 |

---

## Blender rig and IK bone mapping

See [docs/bone_mapping.md](docs/bone_mapping.md) for the full correspondence between 3D keypoints and Blender IK bones.

---

## Dataset correspondence

| Dataset folder | Used by |
|---|---|
| `marmoset_keypoint_data/training_data` | MMDetection / MMPose training |
| `marmoset_keypoint_data/video_data` | Pipeline input videos |
| `marmoset_keypoint_data/2D_temporal_keypoint` | 3D reconstruction input |
| `marmoset_keypoint_data/3D_temporal_keypoint` | Blender animation input |
| `3D_environmental_data/environment_model_blender` | Cage measurement scripts |
| `3D_digital_twin_data/3D_digitaltwin_marmoset_blender` | Blender integrated scenes |

---

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Data**: CC BY 4.0
- **Blender models / rendered assets**: CC BY 4.0
- **Trained model weights**: CC BY 4.0

---

## Citation

If you use this code, please cite:

```bibtex
@article{iwata2025marmoset3d,
  title   = {A multimodal dataset for reconstructing common marmoset body--environment interactions in a 3D digital-twin framework},
  author  = {Iwata, Koki and others},
  journal = {[journal name to be added]},
  year    = {2025},
  doi     = {[DOI to be added]}
}
```

---

## Contact

- **GitHub Issues**: [https://github.com/iwtkk/3D_digital_twin_marmoset_code/issues](https://github.com/iwtkk/3D_digital_twin_marmoset_code/issues)
- **Email**: iwata.koki.sd@gmail.com
