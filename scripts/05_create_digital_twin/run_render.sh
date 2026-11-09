#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
# set -euo pipefail

# 使用する .blend はアニメーション適用済みの work_*.blend を想定
# (run_animate*.sh で生成される outputs/blender/work_*.blend)
#
# 全セッション×全カメラを1本のグローバルキューにまとめ、render_sessions.py が
# GPUスロット単位で動的にレンダリングする。セッション/カメラの境界でGPUが空かない。
#
# デフォルトは warm 運用: blender を GPU スロット数だけ常駐させ、チャンクを流し込む。
# .blend ロード / GPU 初期化 / BVH 構築はセッション切り替え時のみ。
# 従来の「チャンクごとに blender 起動」に戻したい場合は --cold。
# フラグの詳細は: python scripts/05_create_digital_twin/render_sessions.py --help

PY=$(command -v python || command -v python3)
exec "$PY" scripts/05_create_digital_twin/render_sessions.py "$@"
