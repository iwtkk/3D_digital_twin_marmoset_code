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
  --blend <file>      Base blend file (default: 3D_digital_twin_marmoset_data/...)
  --armature <name>   Armature name (default: NF_Armature)
  --csv-dir <dir>     CSV directory (default: outputs/animate_temp/transformedcsv)
  --output-dir <dir>  Output directory for blend files (default: outputs/blenders)
  --count <n>         Max number of CSVs to process (default: 1)
  --all               Process all CSVs
  --verbose           Show verbose Blender output
  -h, --help          Show this help
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
    echo "Reached CSV processing limit (${COUNT}); exiting"
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
