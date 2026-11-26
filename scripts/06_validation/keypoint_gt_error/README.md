# 3D keypoint error against the labeled ground truth (Fig. 2E)

Accuracy of the 3D keypoints against manually labeled 3D keypoints: 1 animal, 100 frames, 20 keypoints.
The data is in the dataset under `marmoset_keypoint_data/training_data/3dkp_gt/`.
Paths are set in `config/keypoint_gt_error.yaml`; the I/O ledger is `io.md`.

## Reproduce Fig. 2E from the dataset

```bash
.venv/bin/python scripts/06_validation/keypoint_gt_error/plot_error_violin.py --dataset-root <dataset>
# -> outputs/validation/keypoint_gt_error/fig2e_error_{violin,box}.{png,pdf}, fig2e_error_summary.csv
```

`<dataset>` is the folder that holds `marmoset_keypoint_data/` after `extract_dataset.py`.
The plot reads `3dkp_gt/matched_keypoints/SingleAnimal_20.npz`
(`--source mat` reads the original `SingleAnimal_20.mat` instead; the result is identical).
Unlabeled keypoints are left out. The original `plotBox.py` set them to 0 instead, which adds false zero errors.

`SingleAnimal_20.mat` was written by `original_code/s20230220_cperror_single.py`. The predictions it read
(`3d_v0p8_dark_fix_20_for_3d_eval`, frame = round(mean of the labeled camera frames) − 24) are no longer
available, so the matching itself cannot be rerun. `fig2e_mat_to_npz.py` recovers the session and frame of
every sample from the labeled JSON files.

## Camera images of the labeled frames

The PNG files used for labeling were not kept. `extract_labeled_images.py` cuts the same frames out of the
dataset videos (`framedata.csv` holds the frame index of each camera within its video):

```bash
.venv/bin/python scripts/06_validation/keypoint_gt_error/extract_labeled_images.py --dataset-root <dataset>
# -> outputs/validation/keypoint_gt_error/labeled_images/valbin1_nanim1_nfrm100/{frame_name}.{camera_id}.png (800)
#    labeled_images/qc/{frame_name}.jpg (labels drawn on the 8 views), labeled_images/extract_log.csv
```

The file names are the ones the labeling tool used, so the images open in the labeling GUI next to the JSON files.

## Files

| File | What it does |
|---|---|
| `plot_error_violin.py` | Violin plot (port of `violin.m` as in `plotVIolin.mlx`) and box plot (as `plotBox.py`) of the per-keypoint error |
| `fig2e_mat_to_npz.py` | Made the dataset `SingleAnimal_20.npz` from the original `.mat` (recovers session / frame) |
| `extract_labeled_images.py` | Re-extracts the 8 camera images of every labeled frame from the dataset videos |
| `original_code/` | Original code, unchanged; its paths point to the authors' machines |

What `original_code/` holds:
- `s20230220_cperror_single.py` writes `SingleAnimal_20.mat`.
- `plotBox.py` and `violin.py` plot it.
- The `.mlx` / `.m` files are an earlier 18-keypoint version (1 / 2 / 3 animals); not reproduced here.
- `violin.m` is by Hoffmann H, 2015, University of Bonn.

## Checked (2026-11-09)

- `fig2e_mat_to_npz.py`:
  - Every sample matches exactly one labeled frame.
  - The sample order equals the original script's.
  - The npz → .mat round trip is exact.
- `plot_error_violin.py`: the summary is identical from the `.npz` and the `.mat`. Median error over all keypoints is 10.33 mm (1,973 labeled keypoints, 100 frames).
