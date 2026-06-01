# Training – Bounding Box Detection and Pose Estimation

Train the MMDetection (YOLOX-L) and MMPose (HRNet-W48) models used in the pipeline.
Corresponds to paper section: **Methods – 2D keypoint estimation**

## Setup

Clone the repository with all submodules:

```bash
git clone --recurse-submodules https://github.com/c7h2y/3D_digital_twin_marmoset_code.git
cd 3D_digital_twin_marmoset_code
```

## Setup

動作確認済み環境: Python 3.10.12、CUDA 12.7

> **注意**: `python3` が 3.12 以降を指す場合、torch 1.13.1 の wheel が存在しないため必ず `python3.10` を指定する。

```bash
python -m venv .venv
source .venv/bin/activate

# 1. PyTorch（cu116 ビルドは CUDA 12.x でも動作する）
#    torch==1.10.2+cu113 は配布終了のため 1.13.1+cu116 を使用する
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 2. mmcv（torch + CUDA バージョンに合わせたビルドを指定）
pip install mmcv-full==1.6.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html

# 3. mmdetection / mmpose（サブモジュールから editable インストール）
pip install -e scripts/02_train/mmdetection
pip install -e scripts/02_train/mmpose

# 4. lap は numpy 2.x でビルドが失敗するため、numpy を先に固定してからインストール
pip install "numpy==1.22.4"
pip install "lap==0.4.0"

# 5. その他の依存パッケージ
pip install -r requirements.txt
```

## Config files

Training configs are in the marmo3Dpose submodule:

```
scripts/03_run_marmo3dpose/marmo3Dpose/model/
├── detection/marmo20/
│   └── tk_yolox_l_8x8_300e_coco_V0p11.py   # main config (YOLOX-L)
└── pose/marmo20/
    └── marmo20_tk_hrnet_w48_coco_384x288_dark_v0p10_IDgeneralization.py
```

## Data layout

学習データは Zenodo からダウンロードできます（URL は論文参照）。

**Detection, Pose:**
```
3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/
├── images/
└── annotations/
    ├── single_train.json
    └── single_test.json
```

## Training

### Bounding box detection (YOLOX-L)

```bash
python scripts/02_train/mmdetection/tools/train.py \
    scripts/03_run_marmo3dpose/marmo3Dpose/model/detection/marmo20/tk_yolox_l_8x8_300e_coco_V0p11.py \
    --work-dir outputs/train/detection/ \
    --cfg-options \
        load_from=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/pretrained/tk_yolox_l_8x8_300e_coco_V0p10_best_bbox_mAP_epoch_282.pth \
        data.train.dataset.ann_file=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/annotations/single_train.json \
        data.train.dataset.img_prefix=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/images/ \
        data.val.ann_file=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/annotations/single_test.json \
        data.val.img_prefix=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/images/
```

### Pose estimation (HRNet-W48, 20 keypoints)

```bash
python scripts/02_train/mmpose/tools/train.py \
    scripts/03_run_marmo3dpose/marmo3Dpose/model/pose/marmo20/marmo20_tk_hrnet_w48_coco_384x288_dark_v0p10_IDgeneralization.py \
    --work-dir outputs/train/pose/ \
    --cfg-options \
        load_from=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/pretrained/tk_hrnet_w48_coco_384x288_dark_epoch_70.pth \
        data.train.ann_file=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/annotations/single_train.json \
        data.train.img_prefix=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/images/ \
        data.val.ann_file=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/annotations/single_test.json \
        data.val.img_prefix=./3D_digital_twin_marmoset_data/marmoset_keypoint_data/training_data/images/
```

## Model details

| Model | Architecture | Input size | Classes / keypoints | Epochs |
|---|---|---|---|---|
| Detection | YOLOX-L | 832×832 | 2 (adult, infant) | 50 |
| Pose | HRNet-W48 | 384×288 | 20 keypoints | 210 |

See [docs/keypoint_definition.md](../../docs/keypoint_definition.md) for keypoint indices.

## Pretrained weights

学習済みモデルは Zenodo からダウンロードできます（URL は論文参照）。

ダウンロード後、`3D_digital_twin_marmoset_data/marmoset_keypoint_data/trained_model/` に配置してください。
