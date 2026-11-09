#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
#et -euo pipefail

BLEND_BASE="3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender.blend"

ARMATURE_NAME="NF_Armature"
CSV_DIR="outputs/animate_temp/transformedcsv"
OUTPUT_DIR="outputs/blender"
COUNT=10
ALL=0
MAX_JOBS=4
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
  --output-dir <dir>  Output directory for blend files (default: outputs/blender)
  --count <n>         Max number of CSVs to process (default: 10)
  --all               Process all CSVs
  --max-jobs <n>      Max number of parallel jobs (default: 4)
  --verbose           Save Blender output to a log file (OUTPUT_DIR/work_NAME.log)
  -h, --help          Show this help
EOF
      exit 0;;
    --blend)      BLEND_BASE="$2";    shift 2;;
    --armature)   ARMATURE_NAME="$2"; shift 2;;
    --csv-dir)    CSV_DIR="$2";       shift 2;;
    --output-dir) OUTPUT_DIR="$2";    shift 2;;
    --count)      COUNT="$2";         shift 2;;
    --all)        ALL=1;              shift;;
    --max-jobs)   MAX_JOBS="$2";      shift 2;;
    --verbose)    VERBOSE=1;          shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$OUTPUT_DIR"

# Count the CSVs to process in advance (excluding skipped ones)
to_process=0
for csv in "$CSV_DIR"/*.csv; do
  name=$(basename "$csv" .csv)
  [[ -e "$OUTPUT_DIR/work_${name}.done" ]] && continue
  (( to_process++ ))
  [[ $ALL -eq 0 ]] && (( to_process >= COUNT )) && break
done

# Temporary directory for per-session done markers
SESSION_DIR=$(mktemp -d)

pids=()
monitor_pid=""

cleanup() {
  [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null
  printf "\n"
  echo "Received interrupt signal. Terminating running blender jobs..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null
  done
  wait
  rm -rf "$SESSION_DIR"
  echo "All jobs terminated"
  exit 1
}
trap cleanup SIGINT SIGTERM

# Progress bar monitor process (not needed in verbose mode)
if [[ $VERBOSE -eq 0 ]] && (( to_process > 0 )); then
  bar_width=30
  (
    while true; do
      done_count=$(ls "$SESSION_DIR"/ 2>/dev/null | wc -l)
      filled=$(( bar_width * done_count / to_process ))
      bar=""
      for (( i=0; i<filled; i++ )); do bar+="="; done
      if (( filled < bar_width )); then
        bar+=">"
        for (( i=filled+1; i<bar_width; i++ )); do bar+=" "; done
      fi
      printf "\r[%s] %d/%d" "$bar" "$done_count" "$to_process"
      (( done_count >= to_process )) && break
      sleep 1
    done
  ) &
  monitor_pid=$!
fi

processed=0

for csv in "$CSV_DIR"/*.csv; do
  name=$(basename "$csv" .csv)
  blend_file="$OUTPUT_DIR/work_${name}.blend"
  done_file="$OUTPUT_DIR/work_${name}.done"
  log_file="$OUTPUT_DIR/work_${name}.log"

  if [[ -e "$done_file" ]]; then
    [[ $VERBOSE -eq 1 ]] && echo "[SKIP] $name"
    continue
  fi

  (( processed++ ))
  if [[ $ALL -eq 0 ]] && (( processed > COUNT )); then
    [[ $VERBOSE -eq 1 ]] && echo "Reached CSV processing limit (${COUNT}); exiting"
    break
  fi

  cp "$BLEND_BASE" "$blend_file"

  if [[ $VERBOSE -eq 1 ]]; then
    echo "[START] $name (log: $log_file)"
    ( blender "$blend_file" \
        -b -P scripts/05_create_digital_twin/blender_animation.py -- "$csv" --armature "$ARMATURE_NAME" \
        > "$log_file" 2>&1 \
      && touch "$done_file" \
      && touch "$SESSION_DIR/$name" \
      && echo "[DONE] $name" ) &
  else
    ( blender "$blend_file" \
        -b -P scripts/05_create_digital_twin/blender_animation.py -- "$csv" --armature "$ARMATURE_NAME" \
        > /dev/null 2>&1 \
      && touch "$done_file" \
      && touch "$SESSION_DIR/$name" ) &
  fi

  pid=$!
  pids+=("$pid")

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

[[ -n "$monitor_pid" ]] && { wait "$monitor_pid" 2>/dev/null; printf "\n"; }
rm -rf "$SESSION_DIR"

echo "All jobs finished"
