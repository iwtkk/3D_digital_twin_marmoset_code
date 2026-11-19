#!/usr/bin/env bash
# Reproduce the Fig. 8a metrics from the Published dataset: render -> real frames -> metrics (+ verify) -> figures.
#
#   scripts/06_validation/consistency/run_all.sh DATASET_ROOT WORK_DIR [GPUS] [BLENDER]
#
#   DATASET_ROOT  extracted Published dataset (extract_dataset.py)
#   WORK_DIR      where renders, real frames, metrics and figures are written
#   GPUS          comma-separated GPU indices for Blender (default 0)
#   BLENDER       Blender 5.2.x executable (default: blender)
#
# Environment: SESSIONS="s1 s2" limits every step to these Sessions (quick check); MODELS="detailed_fur" limits
# the body models (default: all three).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
DATA="$(realpath "${1:?DATASET_ROOT}")"; WORK="$(realpath -m "${2:?WORK_DIR}")"; GPUS="${3:-0}"; BLENDER="${4:-blender}"
MODELS="${MODELS:-detailed_fur detailed_nofur simplified_fur}"
SESS=(); [[ -n "${SESSIONS:-}" ]] && SESS=(--session $SESSIONS)

cd "$HERE"
"$BLENDER" --version | head -1
"$PY" render_pairs.py --dataset-root "$DATA" --work-dir "$WORK" --gpu "$GPUS" --blender "$BLENDER" \
    --models $MODELS "${SESS[@]}"
"$PY" extract_real_frames.py --dataset-root "$DATA" --work-dir "$WORK" --workers 8 "${SESS[@]}"
"$PY" compute_metrics.py --dataset-root "$DATA" --work-dir "$WORK" --models $MODELS --workers 8 "${SESS[@]}" --verify
for m in $MODELS; do
    "$PY" visualize.py --metrics "$WORK/metrics/metrics_animal_$m.json" --output-dir "$WORK/figures/$m"
done
echo "done: $WORK/metrics/verify.json, $WORK/figures/"
