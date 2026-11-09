#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
# set -euo pipefail

BLEND_BASE="3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender.blend"

ARMATURE_NAME="NF_Armature"
CSV_DIR="outputs/animate_temp/transformedcsv"
OUTPUT_DIR="outputs/blenders"
COUNT=1
ALL=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --blend <file>      ベースblendファイル（デフォルト: 3D_digital_twin_marmoset_data/...）
  --armature <name>   アーマチュア名（デフォルト: NF_Armature）
  --csv-dir <dir>     CSVディレクトリ（デフォルト: outputs/animate_temp/transformedcsv）
  --output-dir <dir>  出力blendファイルの保存先（デフォルト: outputs/blenders）
  --count <n>         処理するCSV数の上限（デフォルト: 1）
  --all               すべてのCSVを処理
  --verbose           Blenderの詳細出力を表示
  -h, --help          このヘルプを表示
EOF
      exit 0;;
    --blend)      BLEND_BASE="$2";   shift 2;;
    --armature)   ARMATURE_NAME="$2"; shift 2;;
    --csv-dir)    CSV_DIR="$2";      shift 2;;
    --output-dir) OUTPUT_DIR="$2";   shift 2;;
    --count)      COUNT="$2";        shift 2;;
    --all)        ALL=1;             shift;;
    --verbose)    VERBOSE=1;         shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mapfile -t csvs < <(ls "$CSV_DIR"/*.csv 2>/dev/null)
total=${#csvs[@]}
if [[ $ALL -eq 0 ]] && (( total > COUNT )); then
  total=$COUNT
fi

print_progress() {
  local current=$1 total=$2 name=$3
  local bar_width=30
  local filled=$(( bar_width * current / total ))
  local bar=""
  for (( i=0; i<filled; i++ )); do bar+="="; done
  if (( filled < bar_width )); then
    bar+=">"
    for (( i=filled+1; i<bar_width; i++ )); do bar+=" "; done
  fi
  printf "[%s] %d/%d: %s\n" "$bar" "$current" "$total" "$name"
}

cnt=0

for csv in "${csvs[@]}"; do
  (( cnt++ ))
  if [[ $ALL -eq 0 ]] && (( cnt > COUNT )); then
    echo "CSV の処理上限 (${COUNT}) に達したので終了します"
    break
  fi

  name=$(basename "$csv" .csv)
  print_progress "$cnt" "$total" "$name"

  if [[ $VERBOSE -eq 1 ]]; then
    echo "=== CSV: $csv (name=${name}) ==="
  fi

  mkdir -p "$OUTPUT_DIR"
  cp "$BLEND_BASE" "$OUTPUT_DIR/work_${name}.blend"

  if [[ $VERBOSE -eq 1 ]]; then
    blender "$OUTPUT_DIR/work_${name}.blend" \
      -b -P scripts/05_create_digital_twin/blender_animation.py -- "$csv" --armature "$ARMATURE_NAME"
  else
    blender "$OUTPUT_DIR/work_${name}.blend" \
      -b -P scripts/05_create_digital_twin/blender_animation.py -- "$csv" --armature "$ARMATURE_NAME" \
      > /dev/null 2>&1
  fi
done
