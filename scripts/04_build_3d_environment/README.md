# 04 – 3D 環境モデルの構築

実験ケージの 3D モデルを Blender で構築するステップです。  
モデリング自体は **3D Slicer** と **Blender** を使用し、Python スクリプトは Blender 内部から実行します。

完成した Blender ファイルは `3D_digital_twin_marmoset_data/` 以下に格納されています。

---

## Blender スクリプト

頂点座標・エッジ長・オブジェクト高さを計測するスクリプトは、検証ステップと共用しています。

**→ [`scripts/06_validation/environment_accuracy/`](../06_validation/environment_accuracy/README.md)**

モデル構築中にケージ各部の寸法を確認・記録するためにも同スクリプトを使用します。

---

## ワークフロー概要

1. **3D Slicer** でケージの実寸を計測する
2. **Blender** でケージジオメトリをモデリングする
3. `06_validation/environment_accuracy/` のスクリプトで頂点・エッジ・オブジェクトの座標を確認・記録する
4. カメラを配置する（外部パラメータは `config/marmo_cj425m/cam_extrinsic_optim.h5` を参照）

カメラのシリアル ID と配置については [docs/camera_mapping.md](../../docs/camera_mapping.md) を参照してください。

---

## 出力

- `3D_digital_twin_marmoset_data/environment_model/environment_measurement.blend` — 計測用環境モデル
- `3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend` — マーモセット＋環境の合成シーン
