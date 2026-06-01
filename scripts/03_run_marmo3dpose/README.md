# 03 – Run marmo3Dpose

Run the full 2D detection → keypoint estimation → 3D reconstruction pipeline
using [marmo3Dpose](https://github.com/c7h2y/marmo3Dpose).

## License

marmo3Dpose is licensed under the **Apache License 2.0**.
See `marmo3Dpose/LICENSE` for full terms.

## Setup

```bash
# Initialize submodule (first time only)
git submodule update --init --recursive
```

依存パッケージのインストールは `scripts/00_setup/README.md` を参照してください。

## Input

- Multi-camera synchronized videos: `vid/{session}.{camera_id}/`
- Camera calibration: `config/marmo_cj425m/config.yaml`
- Trained model checkpoints: `3D_digital_twin_marmoset_data/marmoset_keypoint_data/trained_model/`

## Running the pipeline

プロジェクトルートから以下のラッパースクリプトを使って実行します。
結果は `outputs/` 以下に保存されます。

### 全セッションを一括処理

```bash
bash scripts/03_run_marmo3dpose/run_marmo3d.sh --device cuda:0
```

### 特定セッションのみ

```bash
bash scripts/03_run_marmo3dpose/run_marmo3d.sh \
    --device cuda:0 \
    --sessions dailylife_cj611_20230227_100000
```

### 複数 GPU で並列処理（セッションを分割）

```bash
# 2分割の1つ目（GPU 0番）
bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh --device cuda:0 --splits 2 --split_id 1

# 2分割の2つ目（GPU 1番）
bash scripts/03_run_marmo3dpose/run_marmo3d_multi.sh --device cuda:1 --splits 2 --split_id 2
```

### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--device` | `cuda:0` | 使用する GPU |
| `--sessions` | 全セッション | カンマ区切りのセッション名 |
| `--config` | `./config/marmo_cj425m/config.yaml` | キャリブレーション設定 |
| `--results` | `./outputs` | 出力ディレクトリ（上書き可） |

## Output

```
outputs/
├── 2d/{session}/                     # 2D キーポイント（JSON）
├── 3d/{session}/
│   ├── calibration.toml
│   ├── kp2d_f.pickle                 # filtered 2D keypoints (n_frame, n_kp, n_individual, 3, n_cam)
│   ├── kp3d.pickle                   # triangulated 3D keypoints (n_frame, n_kp, 3)
│   └── kp3d_fxdJointLen.pickle       # fixed joint length 3D keypoints
└── video/3d/{session}_*.mp4          # 可視化動画
```

See [docs/data_format.md](../../docs/data_format.md) for array shapes.
