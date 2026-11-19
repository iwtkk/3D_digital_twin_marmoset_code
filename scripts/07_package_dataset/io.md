"""
I/O ledger: 07_package_dataset

Repack the dataset (working copy) into an upload directory that fits the limits of the public
repository (at most 1000 files, each file under 2.5e9 bytes). The source data (working copy and
original) is only read, never modified.

================================================================================
Tool: pack_dataset_zips   (pack_dataset_zips.py)
================================================================================
Purpose:
    pack:  split one folder (--component) or a set of files (--files) into {prefix}_part{NNN}.zip.
           Each zip stays below --max-bytes. Files are assigned to zips by their original size
           (an upper bound of the compressed size), so the limit holds whatever the compression ratio.
           mp4/h5/png/gz/pth/npz/zip are stored uncompressed, everything else is deflated
           (--store stores everything uncompressed).
           --exclude <sub-folder> (repeatable) leaves out a sub-folder that is packed as its own component,
           so a folder can be added to an already uploaded component without changing its zips.
    wrap:  wrap every zip of a flat upload directory in an outer zip of the same name (stored, one member)
           for Harvard Dataverse, which unzips uploaded zips once and so leaves the inner zips as files.
           Other files (manifests, extract_dataset.py) are hard-linked unchanged.
    check: verify the whole upload directory. --dataset-root accepts several roots; each manifest
           must match the source data under one of them (gzipped .blend files are compared with staging).
           A component nested in another manifest's component is left out when the outer one is compared.

Inputs:
    dataset_root:
        type: directory
        path_source: path_registry.data.dataset_work_npz
    component:
        type: directory (relative to dataset_root)
        example: marmoset_keypoint_data/video_data
    files:
        type: list[file] (relative to dataset_root; exclusive with --component)
        note: the manifest's component column holds the file's own path

Outputs:
    zips:
        type: file
        path: {out_dir}/{prefix}_part{NNN}.zip
        path_source: path_registry.outputs.dataset_upload_npz (all files directly in it)
        content: paths inside the zip are relative to dataset_root (extracting every zip in one place
                 restores the original layout)
        size: < max_bytes (default 2,500,000,000)
    manifest:
        type: file
        path: {out_dir}/{prefix}_manifest.csv
        columns: archive(str), component(str), member(str), size_bytes(int), session(str)
        rows: every file of the component (one row per file)
        note: session is taken from the first path element below the component
              (e.g. cj357_20221009_090000); meaningless for folders that are not per session, such as training_data.

    wrapped (wrap):
        type: file
        path: {out_dir}/{name}.zip, one per zip of upload_dir
        path_source: path_registry.outputs.dataset_upload_dataverse
        content: exactly one stored member {name}.zip, byte-identical to upload_dir/{name}.zip
        size: < max_bytes

Side effects:
    - zips are written as {zip}.tmp and renamed when complete (a rerun continues where it stopped)
    - wrap skips outer zips that already hold the right member; --verify-crc re-reads everything and compares CRCs
    - check writes nothing

================================================================================
Tool: gzip_blend_files   (gzip_blend_files.py)
================================================================================
Purpose:
    gzip {src_dir}/*.blend into {dst_dir}/ under the same names. Blender opens them as they are.
    They are not re-saved by Blender, so the .blend file format version does not change.

Inputs:
    src_dir:
        type: directory
        path_source: path_registry.data.dataset_work_npz
        file_pattern: "3D_digital_twin_data/{3D_digitaltwin_marmoset_blender_files|3D_digitaltwin_simplified_marmoset}/{session}.blend"

Outputs:
    gz_blend:
        type: file (gzip, keeps the .blend extension)
        path: {dst_dir}/{session}.blend
        path_source: path_registry.outputs.dataset_upload_npz_staging
        invariant: md5 of the gunzipped content == md5 of the original (checked with --verify)

Side effects:
    - written as {name}.tmp and renamed when complete; existing outputs are skipped (only verified)

================================================================================
Tool: run_package_dataset   (run_package_dataset.sh)
================================================================================
Purpose:
    Runs pack_dataset_zips(pack) -> gzip_blend_files -> pack of the gzipped .blend files ->
    pack of the single model files -> copy of extract_dataset.py -> pack_dataset_zips(check)
    to build the upload directory.

Outputs (upload_dir; every file directly in it, no sub-folders, for Harvard Dataverse):
    {video_data,2D_temporal_keypoint_data,3D_temporal_keypoint_data,training_data,trained_model,
     rendered_environmental_images,rendered_environmental_images_{animal,env},CT_scan_data}_part{NNN}.zip + {prefix}_manifest.csv
        (rendered_environmental_images_{animal,env} = real vs render PSNR/SSIM per pair in the animal / environment
         region + 1 example image each; pair_metrics.csv and camera_summary.csv are in _env)
    rendered_pseudo_egocentric_video_data_part001.zip + manifest
        (3D_digital_twin_data/rendered_pseudo_egocentric_video_data: pseudo first-person videos of cj611_20230226_110000,
         {Eqdst,Pinhole}.mp4 + README.md; mp4 stored uncompressed)
    training_data_3dkp_gt_part001.zip + manifest
        (marmoset_keypoint_data/training_data/3dkp_gt: 3D keypoint ground truth and the Fig. 2E matched
         keypoints SingleAnimal_20; packed separately, training_data_* leaves it out with --exclude)
    {3D_digitaltwin_marmoset_blender_files,3D_digitaltwin_simplified_marmoset}_part{NNN}.zip + manifest
        (contain the gzipped {session}.blend, stored with --store)
    3D_models_part001.zip + manifest   (3dmarmoset*.blend, environment_measurement.blend, base digital twin .blend files)
    extract_dataset.py                 (copied from scripts/00_setup; restores the folder layout from the zip paths)
    every path inside the zips is relative to dataset_root
Staging (staging_dir = path_registry.outputs.dataset_upload_npz_staging):
    3D_digital_twin_data/{...}/{session}.blend (gzip), the source of the .blend zips.

Validation:
    - check prints RESULT: OK (files <= 1000, every file < 2.5e9, manifest == zip contents == source data)
    - gzip_blend_files --verify reports mismatch=0
    - optionally --crc checks the CRC of every zip

History:
    NPZ version (3D keypoints .csv -> .npz): dataset_work_npz / dataset_upload_npz (path_registry) were made
    with hard links and only 3D_temporal_keypoint_data was repacked and checked:
      pack_dataset_zips.py pack --dataset-root <dataset_work_npz> --component marmoset_keypoint_data/3D_temporal_keypoint_data \
          --prefix 3D_temporal_keypoint_data --out-dir <dataset_upload_npz>/marmoset_keypoint_data
      pack_dataset_zips.py check --upload-dir <dataset_upload_npz> --dataset-root <dataset_work_npz>
    2026-11-06: flattened for Dataverse. check --dataset-root <work_npz> <staging> --crc gave RESULT: OK
      (413 files, 912.64 GB; the 29,731 files inside the zips match dataset_work_npz)
    2026-11-09: added training_data_3dkp_gt (210 files, 1.1 MB zip) as its own component;
      the uploaded training_data_part00{1,2}.zip are unchanged. check --dataset-root <work_npz> <staging>: RESULT: OK
      (419 files incl. rendered_environmental_images_{animal,env}, 912.64 GB)
    2026-11-09 14:31: training_data_3dkp_gt repacked (206 files, 0.78 MB zip): fig2e_matched_keypoints now holds only
      SingleAnimal_20.{mat,npz}; check RESULT: OK (419 files)
    2026-11-10: added rendered_pseudo_egocentric_video_data (README.md + cj611_20230226_110000_{Eqdst,Pinhole}.mp4,
      2 zips, 2.86 GB); the empty 3D_digital_twin_data/rendered_pseudo_video_data was removed. check: OK for every
      manifest except training_data_3dkp_gt, whose source folder was being extended by another task at the time.
      Uploaded to the Dataverse draft (--only 'rendered_pseudo_egocentric_video_data*'): 422 files, sizes match, new 3 MD5 match.

================================================================================
Tool: upload_dataverse   (upload_dataverse.py)
================================================================================
Purpose:
    Upload a flat upload directory to a Harvard Dataverse dataset through the API (dvuploader).
    info: token / dataset / direct-upload check. upload: resumable batch upload. verify: compare file lists.

Inputs:
    upload_dir:
        type: directory (flat)
        path_source: path_registry.outputs.dataset_upload_npz (plain zips; direct upload does not unzip)
                     path_registry.outputs.dataset_upload_dataverse only with --force-native (native upload unzips once)
    pid:     dataset persistent id (doi:10.7910/DVN/...)
    server:  Dataverse URL (default https://dataverse.harvard.edu)
    token:   $DATAVERSE_API_TOKEN or --token-file (default ~/.config/dataverse/token, chmod 600; never in the repository)

Outputs:
    files in the Dataverse dataset (draft version), one per local file, same name, tabIngest=false
        native upload: a wrapper zip is unzipped once by Dataverse -> the inner zip (size = inner size)
        direct upload: nothing is unzipped -> the wrapper itself (size = wrapper size)
    stdout: per-batch progress, RESULT: OK / missing files

Side effects:
    - only adds files; files whose name is already in the dataset are skipped (never replaced or deleted)
    - DVUPLOADER_MAX_PKG_SIZE=1 so dvuploader does not bundle small files (a bundle would be ingested)
    - writes nothing locally

Validation:
    - verify: every local file is in the dataset with the local size or (wrapper) inner size, no extra files;
      --md5 also compares the MD5 stored by Dataverse with the local file
    - 2026-11-08 test on doi:10.7910/DVN/HZVBH1 (direct upload): a wrapped zip stayed wrapped; a plain zip and a CSV
      arrived unchanged (CSV not ingested, sizes equal)

================================================================================
Tool: dataset_layout   (dataset_layout.py)
================================================================================
Purpose:
    Show the dataset layout as YAML / Markdown, and change the layout by editing the YAML.
    snapshot: scan the directory and write docs/dataset_layout.yaml (keeps existing path/remove/description)
    md:       write docs/dataset_layout.md from the YAML (tree + table + empty directories)
    plan:     list the moves implied by current vs path / remove (writes nothing)
    apply:    run the plan (dry-run unless --execute), then regenerate the YAML/MD
    undo:     replay an apply log backwards

Inputs:
    dataset_root:  path_registry.data.dataset_work_npz (dataset_root in the YAML)
    layout_yaml:   docs/dataset_layout.yaml
        entries[]: current(str, pattern of the real paths; do not edit), path(str), remove(bool),
                   count(int), size_gb(float), example(str), description(str)
        placeholders: {session}=cj\d+_\d{8}_\d{6}, {cam}=camera_id from the config, {chunk}=6 digits,
                      {frame}=7 digits, {name}=any basename (used when a directory has more than 20 distinct names)
    camera_config: config/marmo_cj425m/config.yaml

Outputs:
    docs/dataset_layout.yaml, docs/dataset_layout.md
    outputs/dataset_layout/apply_{YYYYmmdd_HHMMSS}.tsv: source<TAB>destination

Side effects:
    - only apply --execute changes dataset_root (rename only; contents and hard-linked originals are unchanged)
    - remove: true moves files to removed_backup (path_registry.data.dataset_work_npz_removed); nothing is deleted
    - does nothing if there are duplicate destinations, existing files, unknown placeholders or changed file counts
    - dataset_upload_npz is not rebuilt automatically (rebuild it with run_package_dataset.sh after a layout change)

================================================================================
Tool: export_fig8a_release   (export_fig8a_release.py)
================================================================================
Purpose:
    Write the Fig. 8a files of the Release tree (3D_digital_twin_data/rendered_environmental_images/) from the
    validation outputs: Aligned-pose rows of the pose x model study and the real frames that study used.
    Only metrics are released (docs/adr/0002-fig8a-metrics-without-render-images.md).

Inputs:
    roi_csv: path_registry.outputs.consistency_roi_files/{run}/metrics_roi.csv
        rows real_source == "mapped"; columns used: session_id, camera_id, frame, raw_frame_index
    pose_fur_csv: path_registry.outputs.consistency_affine_files/{run}/metrics_pose_fur.csv
        rows pose == "affine"
    npz_dir: path_registry.data.dataset_work_npz/marmoset_keypoint_data/3D_temporal_keypoint_data/{session}.npz
        frame_num (C, T) int64, frame_min (C,) int64, camera_id (C,) str
    args: --roi-csv --pose-fur-csv --npz-dir --out-dir

Outputs (--out-dir, path_registry.outputs.fig8a_release):
    frames.csv          2400 rows: session_id, camera_id, frame_3d, blend_frame (= frame_3d + 1), video_frame_index,
                        npz_video_frame_index (= frame_num - frame_min), frame_offset (= video - npz)
    pair_metrics.csv    7200 rows (pair x model): session_id, animal_id, camera_id, frame_3d, model, roi, render_bbox,
                        real_bbox, bbox_iou, psnr_full, ssim_full, psnr_animal, ssim_animal
    camera_summary.csv  24 rows (camera x model): camera_label, n_pairs, mean / std of the 5 metrics
    metrics_animal_{detailed_fur,detailed_nofur,simplified_fur}.json
    (README.md of the folder is written by hand.)

Side effects:
    - session_id loses the `dailylife_` prefix of the validation run.
    - Copying into the Release tree is a separate, manual step.

Validation:
    - Means equal the values of metrics_pose_fur.csv (affine): IoU 0.413 / 0.362 / 0.419,
      SSIM_animal 0.488 / 0.487 / 0.487 (detailed_fur / detailed_nofur / simplified_fur).
    - frame_offset counts -2: 24, -1: 192, 0: 1584, +1: 600.
"""
