#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
# set -euo pipefail

BLEND_BASE="3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend"
ARMATURE_NAME="NF_Armature"
CSV_DIR="outputs/animate_temp/transformedcsv"
COUNT=10
ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --blend)    BLEND_BASE="$2";    shift 2;;
    --armature) ARMATURE_NAME="$2"; shift 2;;
    --dir-path) CSV_DIR="$2";       shift 2;;
    --count)    COUNT="$2";         shift 2;;
    --all)      ALL=1;              shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# render_parallel.py に渡すため保持
CAM_NAMES=("Camera_R" "Camera_L")
FRAME_START=1
FRAME_END=10
cnt=0

for csv in "$CSV_DIR"/*.csv; do
  echo 'start' $csv
  ((cnt++))
  if [[ $ALL -eq 0 ]] && (( cnt > COUNT )); then
    echo "CSV の処理上限 (${COUNT}) に達したので終了します"
    break
  fi
  echo $cnt
  name=$(basename "$csv" .csv)

  echo "=== CSV: $csv (name=${name}) ==="

  # 1) CSV からアニメを仕込んで .blend を複製・上書き保存
  mkdir -p outputs/animate_temp/blenders outputs/blender
  cp "$BLEND_BASE" "outputs/blender/work_${name}.blend"

  blender "outputs/blender/work_${name}.blend" \
    -b -P scripts/05_create_digital_twin/blender_animation.py -- "$csv" --armature "$ARMATURE_NAME"

  # 2) その .blend を各カメラごとに並列レンダリング
  mkdir -p "outputs/render/${name}"

  for CAM_NAME in "${CAM_NAMES[@]}"; do
    echo "  -> Render camera: ${CAM_NAME}"

    python scripts/05_create_digital_twin/render_parallel.py \
      --blend "outputs/blender/work_${name}.blend" \
      --camera "$CAM_NAME" \
      --entire_start "$FRAME_START" \
      --entire_end "$FRAME_END" \
      --out "outputs/render/${name}/${CAM_NAME}" \
      --prefix "${name}_${CAM_NAME}_" \
      --chunk_size 50 \
      --max_jobs 2
  done
done
