#!/usr/bin/env bash
# Build the upload directory from the dataset (working copy) within the limits on file count and file size.
# Harvard Dataverse cannot hold folders, so every file is placed directly in UPLOAD_DIR (no sub-folders).
#   - folders               -> pack_dataset_zips.py pack: zips under 2.5e9 bytes + manifest
#   - per-session .blend    -> gzip_blend_files.py (into STAGING_DIR) -> stored (uncompressed) zips
#   - other single files    -> zips (3D_models, trained_model)
#   - extract_dataset.py    -> extraction script for users (restores the folder layout from paths in the zips)
# Finally pack_dataset_zips.py check verifies the whole directory.
# I/O ledger: scripts/07_package_dataset/io.md
cd "$(git rev-parse --show-toplevel)"
set -euo pipefail

DATASET_ROOT="/data/iwata/3D_digital_twin_marmoset_data_work_npz"            # path_registry.data.dataset_work_npz
UPLOAD_DIR="/data/iwata/3D_digital_twin_marmoset_data_upload_npz"            # path_registry.outputs.dataset_upload_npz
STAGING_DIR="/data/iwata/3D_digital_twin_marmoset_data_upload_npz_staging"   # path_registry.outputs.dataset_upload_npz_staging
MAX_BYTES=2500000000
MAX_FILES=1000
WORKERS=16
CRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [OPTIONS]
  --dataset-root <dir>  source dataset (default: $DATASET_ROOT)
  --upload-dir <dir>    output; every file goes directly in it (default: $UPLOAD_DIR)
  --staging-dir <dir>   where gzipped .blend files are written (default: $STAGING_DIR)
  --max-bytes <n>       size limit per file, files stay below it (default: $MAX_BYTES)
  --max-files <n>       limit on the number of files (default: $MAX_FILES)
  --workers <n>         parallel workers per step (default: $WORKERS)
  --crc                 also check the CRC of every zip at the end (slow, reads all data)
EOF
      exit 0;;
    --dataset-root) DATASET_ROOT="$2"; shift 2;;
    --upload-dir)   UPLOAD_DIR="$2";   shift 2;;
    --staging-dir)  STAGING_DIR="$2";  shift 2;;
    --max-bytes)    MAX_BYTES="$2";    shift 2;;
    --max-files)    MAX_FILES="$2";    shift 2;;
    --workers)      WORKERS="$2";      shift 2;;
    --crc)          CRC="--crc";       shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

PY=.venv/bin/python
TOOLS=scripts/07_package_dataset
mkdir -p "$UPLOAD_DIR"
pack() { "$PY" "$TOOLS/pack_dataset_zips.py" pack --out-dir "$UPLOAD_DIR" --max-bytes "$MAX_BYTES" "$@"; }

# 1) folders to zips: component  zip prefix
PACKS=(
  "marmoset_keypoint_data/video_data                  video_data"
  "marmoset_keypoint_data/2D_temporal_keypoint_data   2D_temporal_keypoint_data"
  "marmoset_keypoint_data/3D_temporal_keypoint_data   3D_temporal_keypoint_data"
  "marmoset_keypoint_data/training_data               training_data  marmoset_keypoint_data/training_data/3dkp_gt"
  "marmoset_keypoint_data/training_data/3dkp_gt     training_data_3dkp_gt"
  "marmoset_keypoint_data/trained_model               trained_model"
  "3D_digital_twin_data/rendered_environmental_images rendered_environmental_images"
  "3D_digital_twin_data/rendered_pseudo_egocentric_video_data rendered_pseudo_egocentric_video_data"
  "3D_marmoset_body/CT_scan_data                      CT_scan_data"
)
for row in "${PACKS[@]}"; do
  # optional 3rd column: a nested component packed separately (added to Dataverse later without changing the outer zips)
  read -r comp prefix excl <<<"$row"
  pack --dataset-root "$DATASET_ROOT" --component "$comp" --prefix "$prefix" --workers "$WORKERS" ${excl:+--exclude "$excl"}
done

# 2) gzip the per-session .blend files into STAGING_DIR, then store them in zips
for d in 3D_digitaltwin_marmoset_blender_files 3D_digitaltwin_simplified_marmoset; do
  "$PY" "$TOOLS/gzip_blend_files.py" --src-dir "$DATASET_ROOT/3D_digital_twin_data/$d" \
    --dst-dir "$STAGING_DIR/3D_digital_twin_data/$d" --workers "$WORKERS" --verify
  pack --dataset-root "$STAGING_DIR" --component "3D_digital_twin_data/$d" --prefix "$d" --store --workers "$WORKERS"
done

# 3) single model files to a zip
pack --dataset-root "$DATASET_ROOT" --prefix 3D_models --workers 4 --files \
  3D_marmoset_body/3dmarmoset.blend \
  3D_marmoset_body/3dmarmoset_simplified.blend \
  3D_environmental_body/environment_measurement.blend \
  3D_digital_twin_data/3D_digitaltwin_marmoset_blender.blend \
  3D_digital_twin_data/3D_digitaltwin_marmoset_blender_simplified_3dmarmo.blend

# 4) put the extraction script for users at the top (the repository copy is the master; overwritten every run)
cp scripts/00_setup/extract_dataset.py "$UPLOAD_DIR/extract_dataset.py"

# 5) verify (manifests of gzipped .blend files are compared with STAGING_DIR)
"$PY" "$TOOLS/pack_dataset_zips.py" check --upload-dir "$UPLOAD_DIR" --dataset-root "$DATASET_ROOT" "$STAGING_DIR" \
  --max-files "$MAX_FILES" --max-bytes "$MAX_BYTES" $CRC
if find "$UPLOAD_DIR" -mindepth 1 -type d | grep -q .; then
  echo "[NG] UPLOAD_DIR contains sub-folders"; exit 1
fi
