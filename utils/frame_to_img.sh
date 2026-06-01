#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
set -euo pipefail

# 使い方:
#   ./grab_frame.sh <frame_number(1-based)> <session> <input_dir> [output_dir] [ext]
# 例:
#   ./grab_frame.sh 30 S001 ./videos ./frames mp4

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <frame_number(1-based)> <session> <input_dir> [output_dir] [ext]" >&2
  exit 1
fi

FRAME_1_BASED="$1"            # 人間の数え方：1,2,3...（例: 30）
SESSION="$2"                   # セッションID（例: S001）
IN_DIR="$3"                    # 入力ディレクトリ
OUT_DIR="$4"        # 出力ディレクトリ（省略時は入力と同じ）
EXT='mp4'                # 入力拡張子（省略時 mp4）

# 対象カメラID
cam_ids=(23511614 23506237 23506226 23506239 23511607 23506236 23511613 23506214)

# 0始まりに変換（ffmpegのフレーム番号は0始まり）
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

  # 目的のフレームだけを1枚PNGで書き出し
  ffmpeg -hide_banner -loglevel error -y \
    -i "$inputvid" \
    -vf "$FILTER" \
    -vsync vfr -frames:v 1 \
    "$outfile"

  echo "OK: $outfile"
done
