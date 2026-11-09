#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
set -euo pipefail

# Usage:
#   ./grab_frame.sh <frame_number(1-based)> <session> <input_dir> [output_dir] [ext]
# Example:
#   ./grab_frame.sh 30 S001 ./videos ./frames mp4

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <frame_number(1-based)> <session> <input_dir> [output_dir] [ext]" >&2
  exit 1
fi

FRAME_1_BASED="$1"            # human counting: 1,2,3... (e.g. 30)
SESSION="$2"                   # session ID (e.g. S001)
IN_DIR="$3"                    # input directory
OUT_DIR="$4"        # output directory (defaults to input)
EXT='mp4'                # input extension (defaults to mp4)

# Target camera IDs
cam_ids=(23511614 23506237 23506226 23506239 23511607 23506236 23511613 23506214)

# Convert to 0-based (ffmpeg frame numbers are 0-based)
if ! [[ "$FRAME_1_BASED" =~ ^[0-9]+$ ]] || [[ "$FRAME_1_BASED" -le 0 ]]; then
  echo "Error: frame_number must be a positive integer (1-based)." >&2
  exit 1
fi
FRAME_ZERO=$((FRAME_1_BASED - 1))
FILTER="select=eq(n\\,$FRAME_ZERO)"

mkdir -p "$OUT_DIR"

for cam_id in "${cam_ids[@]}"; do
  inputvid="$IN_DIR/${SESSION}_${cam_id}_000000.${EXT}"
  outfile="$OUT_DIR/${SESSION}_${cam_id}_f${FRAME_1_BASED}.png"

  if [[ ! -f "$inputvid" ]]; then
    echo "WARN: not found -> $inputvid (skip)" >&2
    continue
  fi

  # Write only the target frame as a single PNG
  ffmpeg -hide_banner -loglevel error -y \
    -i "$inputvid" \
    -vf "$FILTER" \
    -vsync vfr -frames:v 1 \
    "$outfile"

  echo "OK: $outfile"
done
