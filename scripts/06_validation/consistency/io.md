"""
I/O ledger: 06_validation/consistency (Fig. 8a reproduction)

Reproduces the released Fig. 8a metrics (3D_digital_twin_data/rendered_environmental_images/) from the Published
dataset only. DATASET = extracted Published dataset (path_registry.data.dataset_work_npz has the same layout),
WORK = --work-dir (path_registry.outputs.consistency_fig8a). Nothing under DATASET is written.

================================================================================
Tool: consistency_core   (consistency_core.py, shared module)
================================================================================
Purpose:
    Dataset-relative paths, pair list (frames.csv), render silhouette, Animal region, PSNR / SSIM.
Inputs:
    frames.csv: one row per pair
        columns: session_id(str), camera_id(str), frame_3d(int), blend_frame(int), video_frame_index(int),
                 npz_video_frame_index(int), frame_offset(int)
    2D detections: DATASET/marmoset_keypoint_data/2D_temporal_keypoint_data/{session}/{session}_{cam}_{chunk:06d}.json
        [n_frame][n_det][track_id, x1, y1, x2, y2, kp(20x3), id_label, id_score]; frame = video_frame_index % 10000
Outputs: none (functions)

================================================================================
Tool: render_pairs   (render_pairs.py, drives render_pairs_blender.py)
================================================================================
Purpose:
    Render every pair of frames.csv for each body model (detailed_fur, detailed_nofur, simplified_fur).
Inputs:
    DATASET/3D_digital_twin_data/3D_digitaltwin_marmoset_blender_files/{session}.blend   (Aligned pose)
    DATASET/3D_digital_twin_data/3D_digitaltwin_simplified_marmoset/{session}.blend     (simplified_fur only:
        collection NF_Marmoset_lowpoly is appended into the Detailed scene)
    args: --dataset-root --work-dir [--frames-csv] [--models] [--session] --gpu [--samples 500] --blender [--dry-run]
Outputs:
    WORK/render/{model}/{session}/Camera.{camera_id}/{frame_3d:07d}.png
        shape: [1536, 2048, 3], dtype: uint8, RGB PNG (8 bit)
    WORK/render/background/Camera.{camera_id}.png   empty cage (NF_Marmoset* collections excluded), one per camera,
        rendered from the first Session of the selection; [1536, 2048, 3] uint8
    WORK/render_jobs/{model}/{session}.json   [{camera, blend_frame, out}]
    WORK/render_logs/{model}/{session}.log
Side effects:
    - One Blender process per Session x model, CUDA_VISIBLE_DEVICES = one entry of --gpu. Existing PNGs are skipped.
    - The .blend files are opened read-only in memory; nothing is saved.
Validation:
    - Release tree scenes, Blender 5.2.2, cj425_20210426_141115 Camera.23506226 3D frame 9040:
      detailed_fur vs the render behind the published values PSNR 61.8 dB (mean |d| 0.03 grey levels);
      simplified_fur 72.4 dB. Blend frame 9040 (= 3D frame) is wrong (40.3 dB, animal displaced).

================================================================================
Tool: extract_real_frames   (extract_real_frames.py)
================================================================================
Purpose:
    Cut the real frame of every pair out of the published videos.
Inputs:
    DATASET/marmoset_keypoint_data/video_data/{session}.{camera_id}/metadata.yaml (imgstore)
    DATASET/marmoset_keypoint_data/3D_temporal_keypoint_data/{session}.npz   (camera_id, frame_min)
    args: --dataset-root --work-dir [--frames-csv] [--frame-source published|npz] [--session] [--workers]
Outputs:
    WORK/real/{session}/Camera.{camera_id}/{frame_3d:07d}.png
        shape: [1536, 2048, 3], dtype: uint8, BGR as returned by imgstore
Validation:
    - --frame-source published, all 2400 pairs: identical (np.array_equal) to the real images the published values
      were computed with (consistency_roi_files/run_20260527_185342/frames_mapped).

================================================================================
Tool: compute_metrics   (compute_metrics.py)
================================================================================
Purpose:
    Full-image and Animal-region PSNR / SSIM and bbox IoU of every pair x body model, in the published file
    formats; --verify compares with the published pair_metrics.csv.
Inputs:
    WORK/render/{model}/..., WORK/render/background/, WORK/real/..., DATASET 2D detections, frames.csv
    args: --dataset-root --work-dir [--frames-csv] [--models] [--session] [--workers] [--verify [DIR]]
Outputs (WORK/metrics/):
    pair_metrics.csv: one row per pair x model
        columns: session_id, animal_id, camera_id, frame_3d, model, roi, render_bbox, real_bbox ("x1 y1 x2 y2" px,
                 empty if none), bbox_iou(float, NaN without detection), psnr_full, ssim_full, psnr_animal, ssim_animal
    camera_summary.csv: one row per camera_id x model: camera_label, n_pairs, {metric}_mean, {metric}_std
    metrics_animal_{model}.json: {region, model, pose, total_pairs, summary{total, by_session, by_camera}, details[]}
    verify.json: per metric max / mean |difference|, share of pairs within TOLERANCE, result OK / FAILED
Side effects:
    - Exit code 1 when --verify fails; stops if render/background/ lacks a camera.
Validation:
    - see README.md "Tolerance": 7200 pairs x models from the renders behind the published values -> OK;
      2 Sessions re-rendered from the Release tree (render_pairs.py, Blender 5.2.2) -> OK.

================================================================================
Tool: visualize   (visualize.py, unchanged)
================================================================================
Purpose:
    Fig. 8a style plots (distribution, by camera, by animal) of one metrics_animal_{model}.json.
Outputs:
    --output-dir/{distribution,scatter,by_camera,by_animal,by_camera_ID *}.png

================================================================================
Tool: run_all   (run_all.sh)
================================================================================
Purpose:
    render_pairs -> extract_real_frames -> compute_metrics --verify -> visualize, for all or SESSIONS / MODELS.
    run_all.sh DATASET_ROOT WORK_DIR [GPUS] [BLENDER]
"""
