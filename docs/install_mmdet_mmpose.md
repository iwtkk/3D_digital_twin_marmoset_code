# Installing MMDetection and MMPose

The 2D detection and keypoint estimation pipeline uses:
- **MMDetection v2.26.0** — bounding box detection
- **MMPose v0.29.0** — 2D keypoint estimation

These versions are older and require specific installation steps.

## Prerequisites

- Python 3.8 (recommended for mmdet v2.x / mmpose v0.x)
- CUDA 11.x (for GPU support)
- PyTorch 1.13.x

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# OpenMIM (package manager for OpenMMLab)
pip install openmim

# MMDetection v2.26.0
mim install mmdet==2.26.0

# MMPose v0.29.0
mim install mmpose==0.29.0
```

## Trained model weights

Download the trained model checkpoint from Zenodo:

```bash
# Bounding box detector
wget [zenodo_url]/bbox_marmo_best.pth -O checkpoints/bbox_marmo_best.pth

# 2D keypoint estimator (HRNet-W48 + DEKR, 20 keypoints)
wget [zenodo_url]/pose_marmo20_best_epoch.pth -O checkpoints/pose_marmo20_best_epoch.pth
```

## Config files

MMDetection / MMPose config files for the marmoset models are located in:

```
configs/
├── detection/
│   └── [config file for bbox detector]
└── pose_estimation/
    └── hrnet_w48_dekr_marmoset20.py
```

## Running inference

### Bounding box detection

```bash
python scripts/02_detect_bbox/run_detection.py \
  --video /path/to/video.mp4 \
  --config configs/detection/bbox_marmoset.py \
  --checkpoint checkpoints/bbox_marmo_best.pth \
  --output outputs/bbox/session/
```

### 2D keypoint estimation

```bash
python scripts/03_estimate_2d_keypoints/run_inference.py \
  --video /path/to/video.mp4 \
  --bbox_result outputs/bbox/session/ \
  --config configs/pose_estimation/hrnet_w48_dekr_marmoset20.py \
  --checkpoint checkpoints/pose_marmo20_best_epoch.pth \
  --output outputs/2d_keypoints/session/
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'mmdet'`**
- Make sure you have activated the venv: `source .venv/bin/activate`

**CUDA version mismatch**
- Match the PyTorch CUDA version to your `nvidia-smi` output.
- See [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) for the correct install command.

**`AttributeError` in mmpose**
- Ensure mmpose v0.29.0 is installed — v1.x has a completely different API.
