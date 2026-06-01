#!/usr/bin/env bash
# =============================================================================
# setup_data_links.sh  —  データベースとリポジトリをシンボリックリンクで接続する
# =============================================================================
#
# 【何をするスクリプトか】
#   リポジトリルート（ez2use/）直下と outputs/ 内に、
#   実データへのシンボリックリンクを作成する。
#   一度だけ実行すれば以降のスクリプトはすべて同じパスで動く。
#
# 【作成されるシンボリックリンク（リポジトリルート基準）】
#
#   ez2use/                              ← リポジトリルート
#   ├── vid             -> <DATA_ROOT>/marmoset_keypoint_data/video_data/
#   └── outputs/
#       ├── 2d          -> <DATA_ROOT>/marmoset_keypoint_data/2D_temporal_keypoint_data/
#       ├── 3d          -> <DATA_ROOT>/marmoset_keypoint_data/3D_temporal_keypoint_data/
#       └── render      -> <DATA_ROOT>/3D_digital_twin_marmoset_data/
#
#   ※ vid/, outputs/ は .gitignore 済みのためリンク自体は git 管理されない。
#
# 【使い方】
#   # 引数で指定する場合
#   bash scripts/00_setup/setup_data_links.sh /path/to/3D_digital_twin_marmoset_data
#
#   # 環境変数で指定する場合
#   export EZ2USE_DATA_ROOT=/data/iwata/3D_digital_twin_marmoset_data
#   bash scripts/00_setup/setup_data_links.sh
#
# 【DATA_ROOT について】
#   DATA_ROOT は以下のサブディレクトリを持つディレクトリを指定する:
#     <DATA_ROOT>/marmoset_keypoint_data/
#     <DATA_ROOT>/3D_digital_twin_marmoset_data/
#
# =============================================================================

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DATA_ROOT="${1:-${EZ2USE_DATA_ROOT:?'DATA_ROOT を第1引数か環境変数 EZ2USE_DATA_ROOT で指定してください'}}"
DATA_ROOT="${DATA_ROOT%/}"  # 末尾スラッシュを除去

MKP="${DATA_ROOT}/marmoset_keypoint_data"
DTW="${DATA_ROOT}/3D_digital_twin_marmoset_data"

echo "DATA_ROOT    : ${DATA_ROOT}"
echo "PROJECT_ROOT : $(pwd)"
echo ""

# 実体ディレクトリの存在確認
for d in "${MKP}/video_data" "${MKP}/2D_temporal_keypoint_data" "${MKP}/3D_temporal_keypoint_data" "${DTW}"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: ディレクトリが見つかりません: $d" >&2
        exit 1
    fi
done

# --- リポジトリルート直下 ---
ln -sfn "${MKP}/video_data" vid
echo "作成: $(pwd)/vid"
echo "   -> ${MKP}/video_data"

# --- outputs/ 内 ---
mkdir -p outputs
ln -sfn "${MKP}/2D_temporal_keypoint_data" outputs/2d
echo "作成: $(pwd)/outputs/2d"
echo "   -> ${MKP}/2D_temporal_keypoint_data"

ln -sfn "${MKP}/3D_temporal_keypoint_data" outputs/3d
echo "作成: $(pwd)/outputs/3d"
echo "   -> ${MKP}/3D_temporal_keypoint_data"

ln -sfn "${DTW}" outputs/render
echo "作成: $(pwd)/outputs/render"
echo "   -> ${DTW}"

echo ""
echo "=== 確認 ==="
for link in vid outputs/2d outputs/3d outputs/render; do
    target=$(readlink "$link" 2>/dev/null || echo "(not a symlink)")
    echo "  $link -> $target"
done
