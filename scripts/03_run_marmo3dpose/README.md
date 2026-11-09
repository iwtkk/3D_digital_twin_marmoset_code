# 03 – Run marmo3Dpose

Run the full 2D detection → keypoint estimation → 3D reconstruction pipeline
using [marmo3Dpose](https://github.com/c7h2y/marmo3Dpose).

## License

marmo3Dpose is licensed under the **Apache License 2.0**.
See `marmo3Dpose/LICENSE` for full terms.

## Setup

```bash
# Initialize submodule (first time only)
git submodule update --init --recursive
```

For installing dependencies, see `scripts/00_setup/README.md`.

## Input

- Multi-camera synchronized videos: `vid/{session}.{camera_id}/`
- Camera calibration: `config/marmo_cj425m/config.yaml`
- Trained model checkpoints: `3D_digital_twin_marmoset_data/marmoset_keypoint_data/trained_model/`

## Running the pipeline

Run from the project root using the wrapper scripts below.
Results are saved under `outputs/`.

### Process all sessions at once

```bash
bash scripts/03_run_marmo3dpose/run_marmo3d.sh --device cuda:0
```

### Specific sessions only

```bash
bash scripts/03_run_marmo3dpose/run_marmo3d.sh \
    --device cuda:0 \
    --sessions dailylife_cj611_20230227_100000
```

### Parallel processing on multiple GPUs (split sessions)

```bash
# First of 2 splits (GPU 0)
bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh --device cuda:0 --splits 2 --split_id 1

# Second of 2 splits (GPU 1)
bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh --device cuda:1 --splits 2 --split_id 2
```

### Main options

| Option | Default | Description |
|---|---|---|
| `--device` | `cuda:0` | GPU to use |
| `--sessions` | All sessions | Comma-separated session names |
| `--config` | `./config/marmo_cj425m/config.yaml` | Calibration config |
| `--results` | `./outputs` | Output directory (can be overridden) |

## Output

```
outputs/
├── 2d/{session}/                     # 2D keypoints (JSON)
├── 3d/{session}/
│   ├── calibration.toml
│   ├── kp2d_f.pickle                 # filtered 2D keypoints (n_frame, n_kp, n_individual, 3, n_cam)
│   ├── kp3d.pickle                   # triangulated 3D keypoints (n_frame, n_kp, 3)
│   └── kp3d_fxdJointLen.pickle       # fixed joint length 3D keypoints
└── video/3d/{session}_*.mp4          # visualization videos
```

See [docs/data_format.md](../../docs/data_format.md) for array shapes.
