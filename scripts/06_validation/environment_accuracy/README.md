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

## Table 3 の自動生成（`make_table3.py`）

手作業（頂点を選んで CSV を出し、xlsx で引き算）をやめ、計測定義を
`config/environment_table3.yaml` に書いて .blend から Table 3 を一発で出す。
**標準の計測対象は公開デジタルツイン**（`3D_digital_twin_data/3D_digitaltwin_marmoset_blender.blend`）。
レンダリングに使っている環境そのものを測るため。

### 手順

```bash
# 1. 計測（既定 = 公開デジタルツイン）。約 30 秒
.venv/bin/python scripts/06_validation/environment_accuracy/make_table3.py

# 2. 結果を見る
cat outputs/dev/environment_table3/digitaltwin/table3.md
```

| 出力 | 中身 |
|---|---|
| `table3.csv` | 19 行。図面値・モデル値・誤差・符号付きの比・一致判定・測ったオブジェクト・注意フラグ |
| `table3.md` | 上の表 + 要約統計（一致数、MAE、最大誤差、比の平均と範囲） |
| `table3.tex` | 原稿 Table 3 と同じ形式の LaTeX 表（booktabs）。論文 No. 順、末尾に avg./max./min. 行。桁は mm が小数 2 桁（10^-3 mm 未満は `<10^{-3}`）、比が小数 3 桁で、比は符号付き（model / blueprint） |
| `summary.json` | 要約統計 + 計測した .blend のパスと md5、Blender のバージョン |
| `measured_raw.json` | Blender が返した生の値（オブジェクトの非表示・modifier・animation の情報つき） |

ほかの版を測るとき:

```bash
.venv/bin/python scripts/06_validation/environment_accuracy/make_table3.py --blend v250430   # 論文 Table 3 の元の版
.venv/bin/python scripts/06_validation/environment_accuracy/make_table3.py --blend dataset   # environment_measurement.blend（旧版）
.venv/bin/python scripts/06_validation/environment_accuracy/make_table3.py --blend /path/to/x.blend --output-dir outputs/dev/environment_table3/x
```

### 仕組み

- `make_table3.py`（.venv）が YAML を読み、`measure_blend.py` を `blender -b` で実行する
  （Blender 同梱の Python に yaml が無いため 2 段構成）。.blend は読むだけで保存しない
- 計測方法は 3 種類（測り方は既存の `object_dimensions.py` / `vertex.csv.py` と同じ）
  - `bbox_extent`: ワールド座標の外接箱の長さ。`geometry: mesh` にすると modifier を通す前のメッシュで測る
    （MIRROR で 2 枚組のドアから 1 枚ぶんを測るのに使う）
  - `vertex_delta`: 2 頂点の座標の差
  - `bbox_gap`: 2 つのオブジェクトの面と面の距離。`a: [Cube, min], b: [Cube.003, max]` なら
    「Cube の下面 − Cube.003 の上面」（論文 No.13 のケージ内の高さ）
  - `bbox_span`: 上のオブジェクトの最大 − 下のオブジェクトの最小
- 版によってオブジェクト名が違う行は `by_blend` で上書きする（ドアの 4 行）
- `blueprint_mm: null` の行は、計測値だけを表に出し、要約統計には入れない（図面値が未確定の項目を仮に入れておくとき用）
- 比は符号付き（model / blueprint）。xlsx の J 列（`1 + ABS(1 - 比)`）は使わない
- ワールド行列は `matrix_world` ではなく `matrix_basis` から計算し直す。非表示のオブジェクトは
  評価されず `matrix_world` が古いままのことがある（250430 版の Cube.002 はスケール未反映で 3707 mm になる）
- note 列の注意フラグ: `hidden`（レンダリングされない）、`matrix_world stale`、`animated`（action を持つ）。
  デジタルツインの Cube.004 は `animated` が出るが、キーフレームは 1 つ（frame 72）で値も保存値と同じ。
  1〜500 フレームで位置が変わらないことを確認済み

### 計測項目の追加・変更

`config/environment_table3.yaml` の `rows` に 1 行足す。オブジェクト名と頂点番号は、Blender で対象を選んで
`vertex.py`（頂点番号と座標を表示）や `object_dimensions.py` で確認する。

```yaml
- {label: 30, paper_no: null, part: "perch length", blueprint_mm: 850, method: bbox_extent, object: Cylinder.019, axis: y}
```

### 版の違い（2026-10-30 確認）

| 版 | 一致 | ドア高さ (#12/#21) | 側面高さ (#16) | 備考 |
|---|---|---|---|---|
| `digitaltwin`（公開デジタルツイン） | 8/19 | 755.0 / 757.6 | 761.0 | 環境は 250430 版がもと |
| `v250430`（論文 Table 3 の元） | 8/19 | 755.0 / 755.0 | 761.0 | ドアは非表示の旧オブジェクトを測っている |
| `dataset`（`environment_measurement.blend` = 250124） | 6/19 | 710.1 / 710.1 | 715.5 | 床を下げる前の旧版 |

## 使い方（手作業スクリプト）

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
