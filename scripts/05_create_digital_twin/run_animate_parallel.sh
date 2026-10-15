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
  --blend <file>      Base blend file (default: 3D_digital_twin_marmoset_data/...)
  --armature <name>   Armature name (default: NF_Armature)
  --csv-dir <dir>     CSV directory (default: outputs/animate_temp/transformedcsv)
  --count <n>         Max number of CSVs to process (default: 10)
  --all               Process all CSVs
  -h, --help          Show this help
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
  echo "Interrupt received. Terminating running blender jobs..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null
  done
  wait
  echo "All jobs terminated"
  exit 1
}
trap cleanup SIGINT SIGTERM

processed=0

mkdir -p outputs/blender

for csv in "$CSV_DIR"/*.csv; do
  name=$(basename "$csv" .csv)
  blend_file="outputs/blender/work_${name}.blend"

  done_file="outputs/blender/work_${name}.done"

  # Skip if a .done marker exists (already processed)
  if [[ -e "$done_file" ]]; then
    echo "[SKIP] $csv (done: $done_file)"
    continue
  fi

  ((processed++))
  if [[ $ALL -eq 0 ]] && (( processed > COUNT )); then
    echo "Reached CSV processing limit (${COUNT}); exiting"
    break
  fi

  echo "[START] $csv (name=$name)"
  cp "$BLEND_BASE" "$blend_file"
  echo "[CP DONE] $blend_file"

  # Subshell that creates .done after blender finishes
  ( blender "$blend_file" \
      -b -P scripts/05_create_digital_twin/blender_animation.py -- "$csv" --armature "$ARMATURE_NAME" \
    && touch "$done_file" ) &
  pid=$!
  echo "[BLENDER PID] $pid for $csv"

  pids+=("$pid")

  # When MAX_JOBS jobs are running, wait for one to finish
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

echo "All jobs finished"
