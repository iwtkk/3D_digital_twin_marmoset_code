"""
I/O ledger: 06_validation/keypoint_gt_error

3D keypoint accuracy against manually labeled ground truth (paper Fig. 2E).
The data lives in the dataset under marmoset_keypoint_data/training_data/3dkp_gt/.
Original analysis (read only): path_registry.data.fig2e_compute_error, copied unchanged to original_code/
    s20230220_cperror_single.py -> SingleAnimal_20.mat (1 animal, 20 keypoints; in the dataset)
    plotBox.py + violin.py      -> violin / box plot of SingleAnimal_20.mat
    s20230220_computeError_{SingleAnimal,two_Animal,Three_Animal}.mlx -> {Single,Two,Three}Animals.mat
    plotVIolin.mlx + violin.m   -> violin plot of those (earlier 18-keypoint version; not in the dataset)
Config: config/keypoint_gt_error.yaml (dataset_root, gt_dir, output_dir, figure{stem, n_kp},
        labeled_images{set, video_dir})

================================================================================
Tool: plot_error_violin   (plot_error_violin.py)
================================================================================
Purpose:
    Fig. 2E: violin plot (port of violin.m, MATLAB ksdensity defaults, y 0-40 mm) and box plot (as plotBox.py,
    whiskers 1.5 IQR) of the per-keypoint 3D error, 1 animal x 100 frames x 20 keypoints. Needs only the dataset.
    NaN (unlabeled) is left out (plotBox.py replaced it with 0).

Inputs:
    {dataset_root}/{gt_dir}/matched_keypoints/SingleAnimal_20.npz
        error_mm (100, 20) float64 mm
    or, with --source mat, SingleAnimal_20.mat: annotKp3ds / decodedKp3ds (20, 3, 100) float64 mm

Outputs ({output_dir} = path_registry.outputs.keypoint_gt_error):
    fig2e_error_violin.{png (200 dpi), pdf}, fig2e_error_box.{png (200 dpi), pdf}
    fig2e_error_summary.csv   21 rows (20 keypoints + all) x
        keypoint str, n_labeled int, median_mm, mean_mm, q25_mm, q75_mm, max_mm float

Validation:
    - 2026-11-09: summary identical from the npz and the mat (1,973 labeled keypoints, median 10.33 mm)

Command:
    .venv/bin/python scripts/06_validation/keypoint_gt_error/plot_error_violin.py [--dataset-root <dataset>] [--source mat]

================================================================================
Tool: fig2e_mat_to_npz   (fig2e_mat_to_npz.py)
================================================================================
Purpose:
    Convert SingleAnimal_20.mat to .npz with the identity of every sample (session, labeled frame, prediction
    frame), recovered by exact matching with the labeled JSON. The .mat file is not changed.

Inputs:
    gt_dir:   directory, 3dkp_gt/labeled_frames (path_registry.data.dataset_3dkp_gt)
              valbin1_nanim1_nfrm100/{framedata.csv, {frame_name}.json}
              json keypoints_3d: [1][20][3] float mm (NaN = not labeled)
    mat_dir:  directory, 3dkp_gt/matched_keypoints
              SingleAnimal_20.mat: annotKp3ds, decodedKp3ds (20, 3, 100) float64 mm

Outputs:
    {out_dir}/SingleAnimal_20.npz   (out_dir = mat_dir in the dataset)
        gt_kp3d, pred_kp3d          (100, 20, 3) float64 mm
        error_mm                    (100, 20) float64
        session, frame_name         (100,) str
        pred_frame                  (100,) int64, 0-based = round(mean(labeled_frame_per_camera)) - 24
                                    (Python round, as s20230220_cperror_single.py)
        camera_id (8,) str, labeled_frame_per_camera (100, 8) int64, kp_names (20,) str, source_mat () str

Validation:
    - every sample matches exactly one labeled frame (NaN-aware equality); the sample order equals that of
      s20230220_cperror_single.py; the npz -> .mat round trip is asserted equal; otherwise the script stops
    - pred_frame cannot be re-checked: the prediction it indexes (3d_v0p8_dark_fix_20_for_3d_eval) is gone
    - 2026-11-09: median error 10.328 mm (same as the .mat)

Command:
    .venv/bin/python scripts/06_validation/keypoint_gt_error/fig2e_mat_to_npz.py \
        --gt-dir <dataset_3dkp_gt>/labeled_frames --mat-dir <dataset_3dkp_gt>/matched_keypoints \
        --out-dir <dataset_3dkp_gt>/matched_keypoints

================================================================================
Tool: extract_labeled_images   (extract_labeled_images.py)
================================================================================
Purpose:
    Re-extract the camera images of the labeled frames from the dataset videos. The PNG files the labeling tool
    worked on were not kept. The images are named as the labeling tool saved them, so the folder opens in the
    labeling GUI next to the JSON files. Only the 1-animal set: the 2/3-animal sessions have no videos in the dataset.

Inputs:
    {dataset_root}/{gt_dir}/labeled_frames/{labeled_images.set}/   (set = valbin1_nanim1_nfrm100)
        framedata.csv   index frame_name str; VideoSession str (e.g. dailylife_cj425_20221106_100000);
                        8 columns = camera ids, int = imgstore frame_number - first frame_number of that camera
        {frame_name}.json   keypoints_2d [1][8][20][2] float px (NaN = not labeled); used only for the QC overlay
    {dataset_root}/{labeled_images.video_dir}/{session}.{camera_id}/metadata.yaml   imgstore (mp4 chunks)
        session = the cjNNN_YYYYMMDD_HHMMSS part of VideoSession

Outputs ({output_dir}/labeled_images/, or --out-dir):
    {set}/{frame_name}.{camera_id}.png   (1536, 2048, 3) uint8 BGR, PNG; 100 frames x 8 cameras = 800 files
        = store.get_image(frame_min + framedata value)
    qc/{frame_name}.jpg                  (768, 2048, 3) uint8 JPEG q85; 2 x 4 tiles of 512 x 384, keypoints_2d in red
    extract_log.csv                      800 rows: frame_name, session, camera_id str; frame_index, frame_number,
                                         n_labeled_kp int; timestamp float s

Validation:
    - stops with RESULT: NG if the frame_number returned by imgstore differs from frame_min + framedata value
    - qc/: the labeled keypoints lie on the animal in every camera (visual check, all 100 frames)
    - 2026-11-09: 800 images, frame-number mismatch 0, RESULT: OK; camera timestamp spread per frame <= 0.042 s
      (1 frame at 24 fps, as the labeling tool's get_nearest_image); qc/ checked for all 100 frames: OK; 3.0 GB

Command:
    .venv/bin/python scripts/06_validation/keypoint_gt_error/extract_labeled_images.py [--dataset-root <dataset>]
        [--out-dir <dir>] [--no-qc] [--limit N]

"""
