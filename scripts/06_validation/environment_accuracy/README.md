# Validation – 3D 環境モデルの幾何精度評価

実験ケージの 3D モデル寸法を実測値と比較し、モデルの幾何的精度を評価します。  
論文セクション対応: **Technical Validation – 3D環境モデルの幾何精度評価**

## 概要

計測スクリプトはここ（`06_validation/environment_accuracy/`）に置いていますが、  
同じスクリプトはモデル構築ステップ（[`scripts/04_build_3d_environment/`](../../04_build_3d_environment/README.md)）でも使用します。  
`04` のREADMEからもこちらを参照しています。

## スクリプト

すべて **Blender の Scripting ワークスペース**（またはテキストエディタ）に貼り付けて実行します。  
CLI からは `blender --background <file.blend> --python <script.py>` でも実行できます。

| スクリプト | 目的 |
|---|---|
| `vertex.py` | Edit モードで選択した頂点のローカル・ワールド座標をコンソールに表示 |
| `vertex.csv.py` | 選択頂点の座標を `selected_vertices.csv` に書き出す（mm 単位） |
| `length.py` | Edit モードで選択したエッジのローカル・ワールド長をコンソールに表示 |
| `object_dimensions.py` | 選択オブジェクトの MaxZ / MinZ を `height.csv` に書き出す（mm 単位） |

## 使い方

### Blender の Scripting ワークスペースから実行

1. Blender で `assets/env/cage_environment.blend` を開く
2. 対象メッシュを選択し、必要に応じて Edit モードに入る
3. 計測したい頂点・エッジ・オブジェクトを選択する
4. Scripting ワークスペースでスクリプトを開いて実行する

### CLI から実行

```bash
blender --background assets/env/cage_environment.blend --python scripts/06_validation/environment_accuracy/object_dimensions.py
```

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `selected_vertices.csv` | 頂点インデックス、ローカル座標、ワールド座標（mm） |
| `height.csv` | オブジェクト名、MaxZ、MinZ（mm） |

出力先は `.blend` ファイルと同じディレクトリです。

## 評価手順

1. 上記スクリプトで 3D モデルの各部寸法を計測・CSV 出力する
2. 3D Slicer 4.11（Markups モジュール）で実物ケージを計測した値と比較する
3. 結果は論文 Table X に記載

## 参考

- 物理計測の手順: 論文 Section 2.X
- カメラ配置: [`docs/camera_mapping.md`](../../../docs/camera_mapping.md)
