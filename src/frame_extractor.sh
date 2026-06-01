#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
set -euo pipefail

# 使い方:
#   ./frame_extractor.sh -n 70 -o /path/to/outdir
#   オプション:
#     -n NUM     : 抜き出すフレーム番号（1始まり）。例: 70  → 0始まりで n=69
#     -o OUTDIR  : 出力ディレクトリ（必須）。存在しなければ作成
#     -p PATTERN : セッションパスのグロブ（既定: vid/dailylife_cj611_20230226_110000.*）
#     -i INFILE  : サブフォルダ内の入力ファイル名（既定: 000000.mp4）
#     --flat     : サブフォルダを作らず OUTDIR 直下にフラット保存（既定: ミラー保存）
#
# 例:
#   ./frame_extractor.sh -n 70 -o ~/outs/110000
#   ./frame_extractor.sh -n 71 -o ~/outs/110000 --flat

HUMAN_N=""
OUTDIR=""
PATTERN=vid/dailylife_cj611_20230226_110000.*
INFILE=000000.mp4
FLAT=0

# --- 引数処理 ---
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

# --- 必須チェック ---
if [[ -z "${HUMAN_N}" ]]; then
  echo "Error: -n <HUMAN_FRAME_NUMBER> is required." >&2
  exit 1
fi
if [[ -z "${OUTDIR}" ]]; then
  echo "Error: -o <OUTDIR> is required." >&2
  exit 1
fi

# 1始まり→0始まりに変換
if ! [[ "$HUMAN_N" =~ ^[0-9]+$ ]] || [[ "$HUMAN_N" -le 0 ]]; then
  echo "Error: -n must be a positive integer (1-based)." >&2
  exit 1
fi
FRAME_N=$((HUMAN_N - 1))

mkdir -p "$OUTDIR"

# --- メイン ---
shopt -s nullglob
FOUND=0
for d in ${PATTERN}; do
  in="$d/$INFILE"
  [[ -f "$in" ]] || { echo "skip (no $INFILE): $d"; continue; }
  FOUND=1

  base="$(basename "$d")"        # 例: dailylife_cj611_20230226_110000.23506214

  out="$OUTDIR/${base}_${INFILE%.*}_frame$(printf "%03d" "$HUMAN_N").png"

  # ffmpeg 実行（69=0始まり、selectの「,」はバックスラッシュでエスケープ）
  ffmpeg -y -i "$in" -vf "select='eq(n\\,$FRAME_N)'" -vsync vfr -frames:v 1 "$out"
  echo "wrote: $out"
done

if [[ $FOUND -eq 0 ]]; then
  echo "No inputs matched. Pattern: $PATTERN  infile: $INFILE" >&2
  exit 2
fi
