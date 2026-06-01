# 00 – Setup

環境のセットアップとデータディレクトリの接続を行います。

## 動作確認済み環境

| 項目 | バージョン |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10.12 |
| CUDA | 11.6 |

## 1. Python 環境

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/00_setup/install.sh
```

`install.sh` は以下を順番に実行します:

| ステップ | 内容 |
|---|---|
| 1 | PyTorch 1.10.2 + torchvision 0.11.3 (CUDA 11.3) |
| 2 | mmcv-full 1.6.2 |
| 3 | mmdetection (サブモジュール) |
| 4 | mmpose (サブモジュール) |
| 5 | `requirements.txt` |
| 6 | `marmo3Dpose/requirements.txt` |
| + | Blender 用 PyYAML → `third_party/` |

## 2. データリンクの作成

実データディレクトリとリポジトリをシンボリックリンクで接続します。
一度だけ実行すれば、以降のスクリプトはすべて同じパスで動作します。

```bash
bash scripts/00_setup/setup_data_links.sh /path/to/3D_digital_twin_marmoset_data
```

`DATA_ROOT` には以下のサブディレクトリを持つディレクトリを指定してください:

```
<DATA_ROOT>/
├── marmoset_keypoint_data/
│   ├── video_data/
│   ├── 2D_temporal_keypoint_data/
│   └── 3D_temporal_keypoint_data/
└── 3D_digital_twin_marmoset_data/
    └── digital_twin_marmoset.blend
```

環境変数で指定することもできます:

```bash
export EZ2USE_DATA_ROOT=/path/to/3D_digital_twin_marmoset_data
bash scripts/00_setup/setup_data_links.sh
```

## 3. Blender 用 Python パッケージ（Blender を使う場合）

Blender は独自の Python インタープリタを使用するため、通常の `pip install` では Blender からパッケージを参照できません。
`third_party/` にインストールすることで `src/marmo3d/blender/render.py` が自動的に読み込みます。

```bash
mkdir -p third_party
pip install --target third_party PyYAML
```

| パッケージ | 用途 |
|---|---|
| `PyYAML` | render_spec.yaml / dataset_spec.yaml の読み込み（step 05, 06） |

## 外部ツール要件

| ツール | インストール方法 | 使用ステップ |
|---|---|---|
| **Blender** (4.5.0 推奨) | [blender.org](https://www.blender.org/download/) | 05, 06 |
| **FFmpeg** (`ffmpeg`, `ffprobe`) | `apt install ffmpeg` | 03, 06 |
