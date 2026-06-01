#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
#et -euo pipefail

BLEND_BASE="3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender.blend"

ARMATURE_NAME="NF_Armature"
CSV_DIR="outputs/animate_temp/transformedcsv"
COUNT=10
ALL=0
MAX_JOBS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --blend <file>      ベースblendファイル（デフォルト: 3D_digital_twin_marmoset_data/...）
  --armature <name>   アーマチュア名（デフォルト: NF_Armature）
  --csv-dir <dir>     CSVディレクトリ（デフォルト: outputs/animate_temp/transformedcsv）
  --count <n>         処理するCSV数の上限（デフォルト: 10）
  --all               すべてのCSVを処理
  -h, --help          このヘルプを表示
EOF
      exit 0;;
    --blend)    BLEND_BASE="$2";    shift 2;;
    --armature) ARMATURE_NAME="$2"; shift 2;;
    --csv-dir)  CSV_DIR="$2";       shift 2;;
    --count)    COUNT="$2";         shift 2;;
    --all)      ALL=1;              shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

pids=()

cleanup() {
  echo "中断シグナルを受信しました。実行中の blender ジョブを終了します..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null
  done
  wait
  echo "すべてのジョブを終了しました"
  exit 1
}
trap cleanup SIGINT SIGTERM

processed=0

mkdir -p outputs/blender

for csv in "$CSV_DIR"/*.csv; do
  name=$(basename "$csv" .csv)
  blend_file="outputs/blender/work_${name}.blend"

  done_file="outputs/blender/work_${name}.done"

  # .done マーカーがあれば処理完了済みとしてスキップ
  if [[ -e "$done_file" ]]; then
    echo "[SKIP] $csv (done: $done_file)"
    continue
  fi

  ((processed++))
  if [[ $ALL -eq 0 ]] && (( processed > COUNT )); then
    echo "CSV の処理上限 (${COUNT}) に達したので終了します"
    break
  fi

  echo "[START] $csv (name=$name)"
  cp "$BLEND_BASE" "$blend_file"
  echo "[CP DONE] $blend_file"

  # blender 完了後に .done を作成するサブシェル
  ( blender "$blend_file" \
      -b -P scripts/05_create_digital_twin/blender_animation.py -- "$csv" --armature "$ARMATURE_NAME" \
    && touch "$done_file" ) &
  pid=$!
  echo "[BLENDER PID] $pid for $csv"

  pids+=("$pid")

  # 並列数が MAX_JOBS に達したら、どれか1つが終わるまで待つ
  if (( ${#pids[@]} >= MAX_JOBS )); then
    wait -n
    new_pids=()
    for p in "${pids[@]}"; do
      kill -0 "$p" 2>/dev/null && new_pids+=("$p")
    done
    pids=("${new_pids[@]}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "すべてのジョブが終了しました"
