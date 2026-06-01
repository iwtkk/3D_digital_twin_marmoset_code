#!/bin/bash
set -eu
cd "$(git rev-parse --show-toplevel)"

# 1. PyTorch (CUDA 11.3)
echo ">>> [1/6] PyTorch + torchvision"
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 2. mmcv-full
echo ">>> [2/6] mmcv-full"
pip install mmcv-full==1.6.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

# 3. mmdetection / mmpose サブモジュール
echo ">>> [3/6] mmdetection"
pip install -e scripts/02_train/mmdetection

echo ">>> [4/6] mmpose"
pip install -e scripts/02_train/mmpose

# 4. requirements.txt
echo ">>> [5/6] requirements.txt"
pip install -r requirements.txt

# 5. marmo3Dpose の依存
echo ">>> [6/6] marmo3Dpose requirements"
pip install -r scripts/03_run_marmo3dpose/marmo3Dpose/requirements.txt

# 6. Blender 用 PyYAML (third_party/)
echo ">>> [+] Blender 用 PyYAML → third_party/"
mkdir -p third_party
pip install --target third_party PyYAML

echo "インストール完了。"
