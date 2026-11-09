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

### Step 1 – Set up dataset and environment

```bash
# Download the dataset from AWS S3 (no sign-in required)
aws s3 --no-sign-request cp --recursive s3://digitaltwin-marmoset/ ./3D_digital_twin_marmoset_data 

# Clone the repository 
git clone --recurse-submodules https://github.com/iwtkk/3D_digital_twin_marmoset_code.git
cd 3D_digital_twin_marmoset_code

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
bash scripts/00_setup/install.sh
bash scripts/00_setup/setup_data_links.sh /path/to/3D_digital_twin_marmoset_data
```

### Step 2 – 2D + 3D pose estimation

Run the full pipeline (2D inference → 3D reconstruction → visualization) for a session:

```bash
bash scripts/03_run_marmo3dpose/run_marmo3d.sh \
  --device cuda:0 \
  --sessions dailylife_cj611_20230227_100000
```

Results are written to `outputs/2d/`, `outputs/3d/`, and `outputs/video/3d/`.

For parallel processing across multiple GPUs:

```bash
# Split sessions across 2 GPUs (run each command on a separate terminal)
bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh --device cuda:0 --splits 2 --split_id 1
bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh --device cuda:1 --splits 2 --split_id 2
```

### Step 3 – Apply pose to Blender and render

```bash
#blender 3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend \
# --background \
# --python scripts/05_create_digital_twin/blender_animation.py -- \
# outputs/animate_temp/transformedcsv/session.csv

bash scripts/05_create_digital_twin/run_animate_.sh \
  --blend 3D_digital_twin_marmoset_data/3D_digital_twin_data/3D_digitaltwin_marmoset_blender.blend \
  --output-dir outputs/blender \
  --all

bash scripts/05_create_digital_twin/run_render.sh \
  --blend-dir outputs/blender \
  --all-csv \
  --all-frames \
  --gpu 0
```

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
