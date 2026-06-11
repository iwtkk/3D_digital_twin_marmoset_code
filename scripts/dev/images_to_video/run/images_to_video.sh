#!/bin/bash
# セッションディレクトリ内の画像シーケンスをカメラごとに MP4 化する (ffmpeg)。
#
# 引数 (METHOD_SPEC 準拠):
#   --input-dir DIR    画像シーケンスのあるセッションディレクトリ (必須)
#   --output-dir DIR   MP4 の出力先 (既定: scripts/dev/images_to_video_files)
#   --session NAME     ファイル名のセッション接頭辞 (既定: input-dir の basename)
#   --framerate N      フレームレート (既定: 24)
#   --help             このヘルプ
#
# 例:
#   ./images_to_video.sh --input-dir ./camtest/cj611_20230226_110000 --framerate 30

set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR="scripts/dev/images_to_video_files"
SESSION=""
FRAMERATE=24

CAMERAS=(
    Camera_Eqdst_100
    Camera_Eqdst_70
    Camera_Pinhole_orig
    Camera_Pinhole_x2
    Camera.23506214
)

usage() {
    sed -n '2,13p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)  INPUT_DIR="${2:-}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
        --session)    SESSION="${2:-}"; shift 2 ;;
        --framerate)  FRAMERATE="${2:-}"; shift 2 ;;
        --help|-h)    usage 0 ;;
        *) echo "不明な引数: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input-dir は必須です" >&2
    usage 1
fi

INPUT_DIR="${INPUT_DIR%/}"
[[ -z "$SESSION" ]] && SESSION="$(basename "$INPUT_DIR")"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: directory not found: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Session   : $SESSION"
echo "Input     : $INPUT_DIR"
echo "Output    : $OUTPUT_DIR"
echo "Framerate : $FRAMERATE fps"
echo ""

for cam in "${CAMERAS[@]}"; do
    INPUT_PATTERN="${INPUT_DIR}/${cam}/${SESSION}_${cam}_%05d.png"
    OUTPUT="${OUTPUT_DIR}/output_${SESSION}_${cam}.mp4"

    if [[ ! -d "${INPUT_DIR}/${cam}" ]]; then
        echo "[SKIP] ${cam} — directory not found"
        continue
    fi

    echo "[START] ${cam} -> ${OUTPUT}"
    ffmpeg -y -framerate "$FRAMERATE" \
        -i "$INPUT_PATTERN" \
        -c:v libx264 -pix_fmt yuv420p \
        "$OUTPUT" \
        && echo "[DONE]  ${cam}" \
        || echo "[FAIL]  ${cam}"
    echo ""
done
