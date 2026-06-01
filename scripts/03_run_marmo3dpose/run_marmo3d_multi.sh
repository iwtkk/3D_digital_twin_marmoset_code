#!/usr/bin/env bash
# ez2use 用ラッパー: marmo3Dpose/allmulti.sh を outputs/ に向けて実行する
#
# 使い方（プロジェクトルートから）:
#   bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh [オプション...]
#
# allmulti.sh のすべてのオプションをそのまま渡せる:
#   --device cuda:1
#   --splits 4 --split_id 2
#   --sessions session1,session2
#   --config ./config/marmo_cj425m/config.yaml
#   など
#
# --results を明示指定すると上書きできる:
#   bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh --results /other/path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${SCRIPT_DIR}/marmo3Dpose/allmulti.sh" \
    --results "${PROJECT_ROOT}/outputs" \
    "$@"
