# 03 – Run marmo3Dpose

Run the full 2D detection → keypoint estimation → 3D reconstruction pipeline
using [marmo3Dpose](https://github.com/c7h2y/marmo3Dpose). This is Step 2 of the
[Quick start](../../README.md#quick-start).

## License

marmo3Dpose is licensed under the **Apache License 2.0**.
See `marmo3Dpose/LICENSE` for full terms.

## Setup

```bash
# Initialize submodule (first time only)
git submodule update --init --recursive
```

For installing dependencies, see `scripts/00_setup/README.md`. The dataset links (`vid`, `outputs/...`) are
made by `scripts/00_setup/setup_data_links.sh`.

## Input

| Input | Path |
|---|---|
| Multi-camera synchronized videos | `vid/{session}.{camera_id}/` (imgstore: `{chunk}.mp4` + `metadata.yaml`) |
| Camera calibration | `marmo3Dpose/calib/marmo_cj425m/config.yaml` (same files as `config/marmo_cj425m/`) |
| Trained models | `marmo3Dpose/weight/{detection,id,marmo20_pose}.pth` (links to the dataset, see below) |

Session names are `{animal}_{YYYYMMDD}_{HHMMSS}` (e.g. `cj611_20230227_100000`).

## Running the pipeline

All commands start from the repository root.

**1. Link the trained models** (once). marmo3Dpose loads them from `weight/` under its own directory:

```bash
M=scripts/03_run_marmo3dpose/marmo3Dpose
mkdir -p $M/weight
ln -sfn "$(readlink -f vid)/../trained_model/detection.pth"               $M/weight/detection.pth
ln -sfn "$(readlink -f vid)/../trained_model/id.pth"                      $M/weight/id.pth
ln -sfn "$(readlink -f vid)/../trained_model/pose_marmo20_best_epoch.pth" $M/weight/marmo20_pose.pth
```

**2. Choose the sessions.** `all.sh` processes every session found in the video directory it is given.
To process only some sessions, link them into a separate directory:

```bash
mkdir -p outputs/vid_selected
ln -sf "$PWD"/vid/cj611_20230227_100000.* outputs/vid_selected/
```

(Pass `$PWD/vid` instead of `outputs/vid_selected` below to process all sessions.)

**3. Run 2D inference → 3D reconstruction → visualization.** `all.sh` takes positional arguments
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

- `NUMBA_NUM_THREADS=1` keeps the triangulation from slowing down on many-core machines.
- Multiple GPUs: split the sessions into several directories in step 2 and run one `all.sh` per GPU
  (`cuda:0`, `cuda:1`, ...) with the same output directory.
- Only the 3D part (from existing 2D results, e.g. `outputs/2d` of the dataset) can be run with
  `process_3d.py`; see the `process_3d.py` call in `all.sh` and set `--label2d_dir` to the 2D results.

## Output

```
outputs/marmo3d/
├── 2d_v0p8_Dark_fix_20_all/{session}/{session}_{camera_id}_{chunk}.json   # 2D detections + keypoints
├── 3d_v0p8_dark_fix_20_all/{session}/
│   ├── calibration.toml, config.toml
│   ├── {camera_id}/alldata.json, frame_num.npy   # per-camera 2D data, video frame of each 3D frame
│   ├── kp2d.pickle, kp2d_f.pickle                # 2D keypoints (raw / filtered)
│   └── kp3d_fxdJointLen.pickle                   # 3D keypoints with fixed joint length (same format as S3)
└── video/3d_v0p8_dark_fix_20_all/                # 3D keypoints drawn on camera 6
```

See [docs/data_format.md](../../docs/data_format.md) for array shapes.

## Next step

Align the 3D keypoints to the digital twin and write `{session}.npz` (Quick start 2-4):

```bash
IN=outputs/marmo3d/3d_v0p8_dark_fix_20_all
python scripts/05_create_digital_twin/alignment_3D_keypoint/run/affine_align.py \
  --all --input-dir $IN --output-dir outputs/alignment --jobs 8
python scripts/05_create_digital_twin/alignment_3D_keypoint/run/export_aligned_npz.py \
  --all --input-dir $IN --affine-log outputs/alignment/affine_log.npz --output-dir outputs/npz --jobs 8
```
