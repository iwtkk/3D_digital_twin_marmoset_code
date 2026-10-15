#!/usr/bin/env bash
# ez2use wrapper: run marmo3Dpose/allmulti.sh with results directed to outputs/
#
# Usage (from the project root):
#   bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh [options...]
#
# All allmulti.sh options can be passed through as is:
#   --device cuda:1
#   --splits 4 --split_id 2
#   --sessions session1,session2
#   --config ./config/marmo_cj425m/config.yaml
#   etc.
#
# Override by specifying --results explicitly:
#   bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh --results /other/path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${SCRIPT_DIR}/marmo3Dpose/allmulti.sh" \
    --results "${PROJECT_ROOT}/outputs" \
    "$@"
