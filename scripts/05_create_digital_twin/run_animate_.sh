#!/usr/bin/env bash
# Apply 3D keypoints to a copy of the digital-twin .blend (one work_<session>.blend per input).
#
# Input: Blender-ready NPZ files ({session}.npz from export_aligned_npz.py, or the NPZ files
# downloaded from Harvard Dataverse). CSV input is still accepted with --csv-dir.
#
#   bash scripts/05_create_digital_twin/run_animate_.sh --npz-dir outputs/npz --all
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."   # repository root (does not need git)

BLEND_BASE="outputs/digitaltwin/3D_digitaltwin_marmoset_blender.blend"
ARMATURE_NAME="NF_Armature"
INPUT_DIR="outputs/npz"
EXT="npz"
OUTPUT_DIR="outputs/blender"
BLENDER="${BLENDER:-blender}"
COUNT=1
ALL=0
VERBOSE=0
SESSIONS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --blend <file>      Base blend file (default: ${BLEND_BASE})
  --armature <name>   Armature name (default: NF_Armature)
  --npz-dir <dir>     Directory of {session}.npz (default: outputs/npz)
  --csv-dir <dir>     Use CSV files from this directory instead of NPZ
  --session <name>    Process only this session (repeatable)
  --output-dir <dir>  Output directory for work_<session>.blend (default: outputs/blender)
  --count <n>         Max number of inputs to process (default: 1)
  --all               Process all inputs
  --verbose           Show Blender output on the terminal (it is always saved to <output-dir>/logs/)
  -h, --help          Show this help

Environment: BLENDER=<path to blender executable> (default: blender)
EOF
      exit 0;;
    --blend)      BLEND_BASE="$2";    shift 2;;
    --armature)   ARMATURE_NAME="$2"; shift 2;;
    --npz-dir)    INPUT_DIR="$2"; EXT="npz"; shift 2;;
    --csv-dir)    INPUT_DIR="$2"; EXT="csv"; shift 2;;
    --session)    SESSIONS+=("$2");   shift 2;;
    --output-dir) OUTPUT_DIR="$2";    shift 2;;
    --count)      COUNT="$2";         shift 2;;
    --all)        ALL=1;              shift;;
    --verbose)    VERBOSE=1;          shift;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ ! -f "$BLEND_BASE" ]]; then
  echo "ERROR: base blend not found: $BLEND_BASE" >&2; exit 1
fi
if ! command -v "$BLENDER" >/dev/null 2>&1; then
  echo "ERROR: blender not found ('$BLENDER'). Install Blender or set BLENDER=/path/to/blender" >&2; exit 1
fi

inputs=()
if (( ${#SESSIONS[@]} > 0 )); then
  for s in "${SESSIONS[@]}"; do inputs+=("$INPUT_DIR/$s.$EXT"); done
else
  mapfile -t inputs < <(ls "$INPUT_DIR"/*."$EXT" 2>/dev/null)
fi
if (( ${#inputs[@]} == 0 )); then
  echo "ERROR: no *.$EXT files in $INPUT_DIR" >&2; exit 1
fi
if [[ $ALL -eq 0 ]] && (( ${#SESSIONS[@]} == 0 )) && (( ${#inputs[@]} > COUNT )); then
  inputs=("${inputs[@]:0:COUNT}")
fi

mkdir -p "$OUTPUT_DIR/logs"
total=${#inputs[@]}
cnt=0
failed=()
for f in "${inputs[@]}"; do
  (( cnt++ ))
  name=$(basename "$f")
  name="${name%.$EXT}"
  echo "[$cnt/$total] $name"
  if [[ ! -f "$f" ]]; then
    echo "  ERROR: input not found: $f" >&2; failed+=("$name"); continue
  fi

  out="$OUTPUT_DIR/work_${name}.blend"
  log="$OUTPUT_DIR/logs/work_${name}.log"
  cp "$BLEND_BASE" "$out"
  cmd=("$BLENDER" "$out" -b --python-exit-code 1
       -P scripts/05_create_digital_twin/blender_animation.py -- "$f" --armature "$ARMATURE_NAME")
  if [[ $VERBOSE -eq 1 ]]; then
    "${cmd[@]}" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}
  else
    "${cmd[@]}" > "$log" 2>&1; rc=$?
  fi
  if (( rc != 0 )) || ! grep -q "Blend file saved" "$log"; then
    echo "  ERROR: blender failed (exit $rc). Last lines of $log:" >&2
    tail -n 20 "$log" >&2
    failed+=("$name")
  else
    echo "  -> $out"
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "Failed: ${failed[*]}" >&2; exit 1
fi
