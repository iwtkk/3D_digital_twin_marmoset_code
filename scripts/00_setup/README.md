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

## 1.5 Extracting the dataset (after downloading)

The released dataset is a single folder holding split zips (`*_partNNN.zip`), `*_manifest.csv` files and `extract_dataset.py` (Harvard Dataverse cannot hold folders).
The zips store the original relative paths, so `extract_dataset.py` restores the original folder layout. The same script is included in the dataset.
It uses only the Python standard library (Python 3.7+), so `.venv` is not needed. It works on Linux, macOS and Windows (on Windows, use `python` or `py` instead of `python3`).

```bash
python3 extract_dataset.py <download_dir> --check          # check that all zips are present
python3 extract_dataset.py <download_dir>                  # extract in place
python3 extract_dataset.py <download_dir> --dst <dataset_dir> --gunzip-blend
python3 extract_dataset.py <download_dir> --component 3D_temporal_keypoint_data   # only some folders
python3 extract_dataset.py <download_dir> --session cj357_20221009_090000
```

- If it stops part way, run it again to continue (files already extracted are detected by size and skipped).
- `--remove-zip` deletes each zip once its contents are verified.
- If a zip still holds a single zip of the same name (the outer wrapper used for the Harvard Dataverse upload), the script unwraps it first.
- The per-session `.blend` files are gzip-compressed. Blender opens them as they are; `--gunzip-blend` decompresses them.
- On Windows, extract to a short path such as `C:\data` (paths inside the dataset are up to 123 characters; Windows limits a full path to 260).

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
