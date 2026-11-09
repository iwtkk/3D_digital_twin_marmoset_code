# 00 – Setup

Set up the environment and connect the data directory.

## Tested environment

| Item | Version |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10.12 |
| CUDA | 11.6 |

## 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/00_setup/install.sh
```

`install.sh` runs the following in order:

| Step | Description |
|---|---|
| 1 | PyTorch 1.10.2 + torchvision 0.11.3 (CUDA 11.3) |
| 2 | mmcv-full 1.6.2 |
| 3 | mmdetection (submodule) |
| 4 | mmpose (submodule) |
| 5 | `requirements.txt` |
| 6 | `marmo3Dpose/requirements.txt` |
| + | PyYAML for Blender → `third_party/` |

## 1.5 Extracting the dataset (after downloading)

The released dataset is a single folder holding split zips (`*_partNNN.zip`), `*_manifest.csv` files and `extract_dataset.py` (Harvard Dataverse cannot hold folders).
The zips store the original relative paths, so `extract_dataset.py` restores the original folder layout. The same script is included in the dataset.
It uses only the Python standard library (Python 3.7+), so `.venv` is not needed. It works on Linux, macOS and Windows (on Windows, use `python` or `py` instead of `python3`).

```bash
python3 extract_dataset.py <download_dir> --check          # check that all zips are present
python3 extract_dataset.py <download_dir>                  # extract in place
python3 extract_dataset.py <download_dir> --dst <dataset_dir> --gunzip-blend
python3 extract_dataset.py <download_dir> --component 3D_temporal_keypoint_data   # only some folders
python3 extract_dataset.py <download_dir> --session cj357_20221009_090000
```

- If it stops part way, run it again to continue (files already extracted are detected by size and skipped).
- `--remove-zip` deletes each zip once its contents are verified.
- If a zip still holds a single zip of the same name (the outer wrapper used for the Harvard Dataverse upload), the script unwraps it first.
- The per-session `.blend` files are gzip-compressed. Blender opens them as they are; `--gunzip-blend` decompresses them.
- On Windows, extract to a short path such as `C:\data` (paths inside the dataset are up to 123 characters; Windows limits a full path to 260).

## 2. Creating data links

Connect the actual data directory to the repository with symbolic links.
Run this only once; after that, all scripts work with the same paths.

```bash
bash scripts/00_setup/setup_data_links.sh /path/to/3D_digital_twin_marmoset_data
```

For `DATA_ROOT`, specify a directory containing the following subdirectories:

```
<DATA_ROOT>/
├── marmoset_keypoint_data/
│   ├── video_data/
│   ├── 2D_temporal_keypoint_data/
│   └── 3D_temporal_keypoint_data/
└── 3D_digital_twin_marmoset_data/
    └── digital_twin_marmoset.blend
```

You can also specify it with an environment variable:

```bash
export EZ2USE_DATA_ROOT=/path/to/3D_digital_twin_marmoset_data
bash scripts/00_setup/setup_data_links.sh
```

## 3. Python packages for Blender (if you use Blender)

Blender uses its own Python interpreter, so packages installed with a normal `pip install` are not visible from Blender.
Installing them into `third_party/` lets `src/marmo3d/blender/render.py` load them automatically.

```bash
mkdir -p third_party
pip install --target third_party PyYAML
```

| Package | Purpose |
|---|---|
| `PyYAML` | Loading render_spec.yaml / dataset_spec.yaml (step 05, 06) |

## External tool requirements

| Tool | Installation | Used in steps |
|---|---|---|
| **Blender** (4.5.0 recommended) | [blender.org](https://www.blender.org/download/) | 05, 06 |
| **FFmpeg** (`ffmpeg`, `ffprobe`) | `apt install ffmpeg` | 03, 06 |
