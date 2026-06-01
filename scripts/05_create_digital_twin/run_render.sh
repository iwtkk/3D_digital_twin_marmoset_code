#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
# set -euo pipefail

# 使用する .blend はアニメーション適用済みの work_*.blend を想定
# (run_animate*.sh で生成される outputs/blender/work_*.blend)

FRAME_START=1
FRAME_END=10
COUNT=1
ALL=0
ALL_FRAMES=0
DIR_PATH="outputs/animate_temp/transformedcsv"
BLEND_DIR="outputs/blender"
OUTPUT_DIR="outputs/render"
RESOLUTION=""
CAM_NAMES=()
GPU_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --cam <name>        カメラ名（複数指定可、デフォルト: Camera_C Camera_L Camera_R）
  --count <n>         処理するCSV数の上限（デフォルト: 1）
  --all-csv           すべてのCSVを処理
  --frame-start <n>   開始フレーム（デフォルト: 1）
  --frame-end <n>     終了フレーム（デフォルト: 10）
  --csv-dir <dir>     CSVディレクトリ（デフォルト: outputs/animate_temp/transformedcsv）
  --blend-dir <dir>   blendファイルのディレクトリ（デフォルト: outputs/blender）
  --output-dir <dir>  レンダリング結果の保存先（デフォルト: outputs/render）
  --resolution <WxH>  解像度（例: 1920x1080、デフォルト: blendファイルの設定値）
  --gpu <ids>         使用するGPU ID（カンマ区切り、例: 0,1,2）
  --all-frames        CSVの全フレームを処理（--frame-start/--frame-end を上書き）
  -h, --help          このヘルプを表示
EOF
      exit 0;;
    --cam)        CAM_NAMES+=("$2");  shift 2;;
    --count)      COUNT="$2";         shift 2;;
    --all-csv)    ALL=1;              shift;;
    --frame-start) FRAME_START="$2"; shift 2;;
    --frame-end)   FRAME_END="$2";   shift 2;;
    --csv-dir)    DIR_PATH="$2";      shift 2;;
    --blend-dir)  BLEND_DIR="$2";    shift 2;;
    --output-dir) OUTPUT_DIR="$2";        shift 2;;
    --resolution) RESOLUTION="$2";        shift 2;;
    --gpu)        GPU_ARGS=(--gpu "$2");  shift 2;;
    --all-frames) ALL_FRAMES=1;          shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ ${#CAM_NAMES[@]} -eq 0 ]]; then
  CAM_NAMES=("Camera_C" "Camera_L" "Camera_R")
fi

cnt=0

for csv in "$DIR_PATH"/*.csv; do
  echo 'start' $csv
  ((cnt++))
  if [[ $ALL -eq 0 ]] && (( cnt > COUNT )); then
    echo "CSV の処理上限 (${COUNT}) に達したので終了します"
    break
  fi
  echo $cnt
  name=$(basename "$csv" .csv)

  if [[ $ALL_FRAMES -eq 1 ]]; then
    # i_frame は 0 始まり、Blender フレームは 1 始まりなのでヘッダー除いた行数がそのまま終了フレーム
    FRAME_START=1
    FRAME_END=$(( $(wc -l < "$csv") - 1 ))
    echo "  -> 全フレーム: ${FRAME_START}〜${FRAME_END}"
  fi

  mkdir -p "${OUTPUT_DIR}/${name}"

  for CAM_NAME in "${CAM_NAMES[@]}"; do
    echo "  -> Render camera: ${CAM_NAME}"

    python scripts/05_create_digital_twin/render_parallel.py \
      --blend "${BLEND_DIR}/work_${name}.blend" \
      --camera "$CAM_NAME" \
      --entire_start "$FRAME_START" \
      --entire_end "$FRAME_END" \
      --out "${OUTPUT_DIR}/${name}/${CAM_NAME}" \
      --prefix "${name}_${CAM_NAME}_" \
      ${RESOLUTION:+--resolution "$RESOLUTION"} \
      "${GPU_ARGS[@]}"
  done
done
