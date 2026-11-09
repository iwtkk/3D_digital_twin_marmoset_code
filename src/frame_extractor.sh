#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
set -euo pipefail

# Usage:
#   ./frame_extractor.sh -n 70 -o /path/to/outdir
#   Options:
#     -n NUM     : frame number to extract (1-based). e.g. 70  -> n=69 (0-based)
#     -o OUTDIR  : output directory (required). Created if missing
#     -p PATTERN : glob for session paths (default: vid/dailylife_cj611_20230226_110000.*)
#     -i INFILE  : input file name inside each subfolder (default: 000000.mp4)
#     --flat     : save flat directly under OUTDIR without subfolders (default: mirror layout)
#
# Example:
#   ./frame_extractor.sh -n 70 -o ~/outs/110000
#   ./frame_extractor.sh -n 71 -o ~/outs/110000 --flat

HUMAN_N=""
OUTDIR=""
PATTERN=vid/dailylife_cj611_20230226_110000.*
INFILE=000000.mp4
FLAT=0

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) HUMAN_N="$2"; shift 2 ;;
    -o) OUTDIR="$2"; shift 2 ;;
    -p) PATTERN="$2"; shift 2 ;;
    -i) INFILE="$2"; shift 2 ;;
    --flat) FLAT=1; shift ;;
    -h|--help)
      grep "^# " "$0" | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# --- Required checks ---
if [[ -z "${HUMAN_N}" ]]; then
  echo "Error: -n <HUMAN_FRAME_NUMBER> is required." >&2
  exit 1
fi
if [[ -z "${OUTDIR}" ]]; then
  echo "Error: -o <OUTDIR> is required." >&2
  exit 1
fi

# Convert 1-based to 0-based
if ! [[ "$HUMAN_N" =~ ^[0-9]+$ ]] || [[ "$HUMAN_N" -le 0 ]]; then
  echo "Error: -n must be a positive integer (1-based)." >&2
  exit 1
fi
FRAME_N=$((HUMAN_N - 1))

mkdir -p "$OUTDIR"

# --- Main ---
shopt -s nullglob
FOUND=0
for d in ${PATTERN}; do
  in="$d/$INFILE"
  [[ -f "$in" ]] || { echo "skip (no $INFILE): $d"; continue; }
  FOUND=1

  base="$(basename "$d")"        # e.g. dailylife_cj611_20230226_110000.23506214

  out="$OUTDIR/${base}_${INFILE%.*}_frame$(printf "%03d" "$HUMAN_N").png"

  # Run ffmpeg (69 = 0-based; the ',' in select is escaped with a backslash)
  ffmpeg -y -i "$in" -vf "select='eq(n\\,$FRAME_N)'" -vsync vfr -frames:v 1 "$out"
  echo "wrote: $out"
done

if [[ $FOUND -eq 0 ]]; then
  echo "No inputs matched. Pattern: $PATTERN  infile: $INFILE" >&2
  exit 2
fi
