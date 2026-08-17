#!/usr/bin/env bash
# run_render.sh の depth 版。
# 各 CSV に対応する work_*.blend を開き、指定カメラで Cycles の Depth(Z) パスを
# EXR 出力する（オプションは run_render.sh と同じ。--samples / --npy を追加）。
cd "$(git rev-parse --show-toplevel)"
# set -euo pipefail

FRAME_START=1
FRAME_END=10
COUNT=1
ALL=0
ALL_FRAMES=0
DIR_PATH="outputs/animate_temp/transformedcsv"
BLEND_DIR="outputs/blender"
OUTPUT_DIR="outputs/depth"
RESOLUTION=""
CAM_NAMES=()
GPU_ARGS=()
WORKERS_PER_GPU=""
VERBOSE=0
BACKEND=""
CUDA_PATH=""
SAMPLES=""
TO_NPY=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [OPTIONS]

run_render.sh と同じ要領で、Cycles の Depth(Z) パスを EXR 出力します。

Options:
  --cam <name>        カメラ名（複数指定可、デフォルト: Camera_C Camera_L Camera_R）
  --count <n>         処理するCSV数の上限（デフォルト: 1）
  --all-csv           すべてのCSVを処理
  --frame-start <n>   開始フレーム（デフォルト: 1）
  --frame-end <n>     終了フレーム（デフォルト: 10）
  --csv-dir <dir>     CSVディレクトリ（デフォルト: outputs/animate_temp/transformedcsv）
  --blend-dir <dir>   blendファイルのディレクトリ（デフォルト: outputs/blender）
  --output-dir <dir>  depth の保存先（デフォルト: outputs/depth）
  --resolution <WxH>  解像度（例: 1920x1080、デフォルト: blendファイルの設定値）
  --samples <n>       Cycles サンプル数（depth は 1 で十分、デフォルト: 1）
  --gpu <ids>         使用するGPU ID（カンマ区切り、例: 0,1,2）
  --workers-per-gpu <n>  GPU 1台あたりに同時実行する blender 数（デフォルト: 1）
  --backend <name>    Cycles の GPU バックエンド CUDA / OPTIX（デフォルト: CUDA）
  --cuda-path <dir>   実行時に使う CUDA toolkit パス（デフォルト: /usr/local/cuda-12）
  --npy               レンダー後に EXR を .npy(行0=上端,背景NaN) へ変換（python3+OpenEXR）
  --cold              blender を常駐させず、チャンクごとに起動し直す（従来方式）
  --optimize          Simplify 等の高速化設定を適用（blender_render/tools/README.md 参照）
  --progress          フレーム単位の進捗を表示
  --verbose           Blender の詳細ログを表示
  --all-frames        CSVの全フレームを処理（--frame-start/--frame-end を上書き）
  -h, --help          このヘルプを表示

備考: Cycles の Depth パスは魚眼カメラでレイ距離、ピンホールで平面深度を返す。
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
    --samples)    SAMPLES="$2";           shift 2;;
    --gpu)        GPU_ARGS=(--gpu "$2");  shift 2;;
    --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2;;
    --backend)    BACKEND="$2";          shift 2;;
    --cuda-path)  CUDA_PATH="$2";        shift 2;;
    --npy)        TO_NPY=1;              shift;;
    --cold|--optimize|--progress) EXTRA_ARGS+=("$1"); shift;;
    --verbose)    VERBOSE=1;             shift;;
    --all-frames) ALL_FRAMES=1;          shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ ${#CAM_NAMES[@]} -eq 0 ]]; then
  CAM_NAMES=("Camera_C" "Camera_L" "Camera_R")
fi

cnt=0

for csv in "$DIR_PATH"/*.csv; do
  ((cnt++))
  if [[ $ALL -eq 0 ]] && (( cnt > COUNT )); then
    echo "CSV の処理上限 (${COUNT}) に達したので終了します"
    break
  fi
  name=$(basename "$csv" .csv)
  echo "[${cnt}] $name"

  if [[ $ALL_FRAMES -eq 1 ]]; then
    FRAME_START=1
    FRAME_END=$(( $(wc -l < "$csv") - 1 ))
    echo "  -> 全フレーム: ${FRAME_START}〜${FRAME_END}"
  fi

  mkdir -p "${OUTPUT_DIR}/${name}"

  for CAM_NAME in "${CAM_NAMES[@]}"; do
    echo "  -> Depth camera: ${CAM_NAME}"

    python3 scripts/05_create_digital_twin/render_parallel_depth.py \
      --blend "${BLEND_DIR}/work_${name}.blend" \
      --camera "$CAM_NAME" \
      --entire_start "$FRAME_START" \
      --entire_end "$FRAME_END" \
      --out "${OUTPUT_DIR}/${name}/${CAM_NAME}" \
      --prefix "${name}_${CAM_NAME}_" \
      ${RESOLUTION:+--resolution "$RESOLUTION"} \
      ${SAMPLES:+--samples "$SAMPLES"} \
      ${WORKERS_PER_GPU:+--workers_per_gpu "$WORKERS_PER_GPU"} \
      ${BACKEND:+--backend "$BACKEND"} \
      ${CUDA_PATH:+--cuda-path "$CUDA_PATH"} \
      $([[ $VERBOSE -eq 1 ]] && echo --verbose) \
      "${EXTRA_ARGS[@]}" \
      "${GPU_ARGS[@]}"

    if [[ $TO_NPY -eq 1 ]]; then
      echo "     -> EXR を npy へ変換"
      python3 scripts/05_create_digital_twin/depth_exr_to_npy.py \
        --dir "${OUTPUT_DIR}/${name}/${CAM_NAME}"
    fi
  done
done
