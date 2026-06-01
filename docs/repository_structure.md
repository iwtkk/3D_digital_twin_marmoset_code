# リポジトリ構造方針

## 現状の問題

| 場所 | 問題 |
|---|---|
| `class_blueprint/` | 名前が論文公開に不適切。リポジトリ直下にあり Python パッケージとして機能していない |
| `src/` | 旧来のスクリプトが雑多に混在。用途が不明確 |
| `animate/` | `scripts/` 外に浮いており、`scripts/05_create_digital_twin/` と重複 |

---

## 目標構造

```
ez2use/
├── src/
│   └── marmo3d/              # class_blueprint/ を移動・改名
│       ├── __init__.py
│       ├── blender/          # Blender レンダリングパイプライン
│       ├── marmo_dataset.py  # データセット読み込み
│       ├── rendering_parallelized.py
│       └── validation/       # バリデーションユーティリティ
├── scripts/
│   ├── 00_setup/             # 環境構築・データダウンロード
│   ├── 01_build_3d_marmoset/ # マーモセット 3D モデル構築
│   ├── 02_train/             # 検出・姿勢推定モデルの学習
│   ├── 03_run_marmo3dpose/   # 2D 検出 → 3D 再構成
│   ├── 04_build_3d_environment/ # ケージ環境の 3D モデル構築
│   ├── 05_create_digital_twin/  # 3D キーポイント → Blender アニメーション
│   └── 06_validation/        # 再投影誤差・SSIM/PSNR 検証
├── config/                   # YAML 設定ファイル群
├── docs/                     # ドキュメント
├── assets/                   # Blender シーンファイル等
└── tests/                    # ユニットテスト
```

---

## 各変更の詳細

### 1. `class_blueprint/` → `src/marmo3d/`

**理由**: 論文公開コードとして `class_blueprint` という名前は不適切。  
`marmo3d` はリポジトリの目的（マーモセット 3D デジタルツイン）を端的に表す。

**移動内容**:
```
class_blueprint/blender/         → src/marmo3d/blender/
class_blueprint/marmo_dataset.py → src/marmo3d/marmo_dataset.py
class_blueprint/rendering_parallelized.py → src/marmo3d/rendering_parallelized.py
class_blueprint/validation/      → src/marmo3d/validation/
```

**パッケージ名の候補**（変更可）:

| 名前 | 特徴 |
|---|---|
| `marmo3d` | ドメインが明確。論文コードとして自然 |
| `dtwin` | デジタルツインにフォーカス |
| `ez2use` | リポジトリ名と統一するが、パッケージ名としてやや冗長 |

→ **暫定: `marmo3d`**（ユーザーの確認を待つ）

### 2. `src/` の旧スクリプト群の整理

現在の `src/` には旧来のバラバラなスクリプトが混在している。

| ファイル | 移動先 |
|---|---|
| `tracer.py`, `plantuml.py` | `src/marmo3d/` 内の dev ユーティリティとして保持 |
| `visualize_*.py`, `plot_*.py` | `scripts/06_validation/` 配下（重複があれば削除） |
| `frame_extractor.sh` | `scripts/03_run_marmo3dpose/` |
| `cropper/` | `scripts/` 配下の適切なステップへ |

### 3. `animate/` → `scripts/05_create_digital_twin/` に統合

`animate/BlenderSync/` の内容は `scripts/05_create_digital_twin/` とほぼ重複している。  
`animate/` を廃止し、`scripts/05_create_digital_twin/` に一本化する。

---

## インポートパスの変更方針

`class_blueprint` → `marmo3d` への改名に伴い、以下のインポート文が変わる:

```python
# Before
import class_blueprint.marmo_dataset as marmo_dataset
from class_blueprint.blender import render_setting

# After
import marmo3d.marmo_dataset as marmo_dataset
from marmo3d.blender import render_setting
```

影響ファイル（主なもの）:
- `scripts/06_validation/consistency/run_pipeline.py`
- `class_blueprint/blender/run_render.py`
- `class_blueprint/rendering_parallelized.py`

---

## 実施順序

1. パッケージ名の確定（`marmo3d` または別名）
2. `class_blueprint/` を `src/marmo3d/` に移動
3. 全インポート文を一括置換
4. `src/` 旧スクリプト群の振り分け
5. `animate/` を `scripts/05_create_digital_twin/` に統合・廃止
6. `run_pipeline.py` 等のパス参照を修正
