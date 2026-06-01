# RenderSpec クラスの使い方（render_spec.yaml 対応）

このドキュメントは、`render_spec.yaml` を読み込んで **Blenderレンダリング用の設定（profile / view / 解像度 / 入出力パス）を一貫して扱う**ための `RenderSpec` クラスの使い方をまとめたものです。

---

## 1. 目的

* `render_spec.yaml` を Python から読み込み
* `{PROJECT_ROOT}` などのテンプレートを解決
* `profile` に応じて **view（カメラ一覧）** を生成
* **image / video の出力パス**をフレームごとに生成
* 必要なら **manifest**（実行ログJSON）も保存

---

## 2. 前提・依存

* Python 3.10+ 推奨
* 依存：`PyYAML`

```bash
pip install pyyaml
```

---

## 3. ファイル配置の想定

最低限、次のような構造を想定しています（例）：

```
PROJECT_ROOT/
  render_spec.yaml
  assets/
    base.blend
  config/
    marmoXXXX/
      config.yaml
  tmp/
  outputs/
```

> `env_camera` プロファイルでは `config/marmo*/config.yaml` の `camera_id` と `img_size`（等）を参照する想定です。

---

## 4. クイックスタート

### 4.1 読み込み

```python
from pathlib import Path

spec = RenderSpec.from_yaml("render_spec.yaml")
project_root = Path(".").resolve()
```

### 4.2 base.blend の解決

```python
base_blend_path = spec.resolve_base_blend(project_root)
print(base_blend_path)
```

### 4.3 解像度の取得（profile依存）

```python
w, h = spec.get_resolution(project_root)
print(w, h)
```

### 4.4 view（カメラ）一覧の取得（profile依存）

```python
views = spec.get_views(project_root)
for v in views:
    print(v.view_index, v.view_id, v.camera_object_name, v.camid)
```

* `env_camera`：`config.yaml` の `camera_id` 個数に応じて `Camera.{view_index}` を生成
* `marmo_eyes`：`["camera_l", "camera_r"]` の2つ
* `marmo_an_eye`：単眼（例：`camera_l`）のみ

---

## 5. 入出力パス生成

### 5.1 temp3d の入力パス（フレーム単位）

```python
temp3d = spec.temp3d_path(project_root, session_id="S001", frame=123)
print(temp3d)
```

### 5.2 画像出力パス（frame × view）

```python
animal = "cj611"
session_id = "dailylife_cj611_20250101_120000"
frames = [0, 123, 456]

img_paths = spec.image_paths(project_root, animal, session_id, frames)

for view, paths in img_paths.items():
    print(view.view_id, "->", paths[0])  # 最初のフレームの出力先
```

戻り値はこうなります：

* `Dict[ViewInfo, List[Path]]`

  * Key：view（カメラ）
  * Value：そのカメラでのフレームごとの画像ファイルパス

### 5.3 動画出力パス（viewごと）

```python
vid_paths = spec.video_paths(project_root, animal, session_id, start_frame=0, end_frame=1000)
for view, outpath in vid_paths.items():
    print(view.view_id, "->", outpath)
```

戻り値：

* `Dict[ViewInfo, Path]`（viewごとの動画ファイルパス）

---

## 6. profile を切り替えて使う

`render_spec.yaml` の `rendering.profile` がデフォルトで使われますが、明示的に指定もできます。

```python
views_env = spec.get_views(project_root, profile="env_camera")
views_eyes = spec.get_views(project_root, profile="marmo_eyes")
```

同様に：

```python
spec.get_resolution(project_root, profile="marmo_eyes")
spec.image_paths(project_root, animal, session_id, frames, profile="marmo_eyes")
```

---

## 7. Manifest（実行ログ）を保存する（任意）

レンダリング実行後に「何を出したか」を JSON で残す用途です。

```python
views = spec.get_views(project_root)

m = RenderManifest(
    animal=animal,
    session_id=session_id,
    render_profile=spec.profile,
    view_ids=[v.view_id for v in views],
    frames=[0, 123, 456],  # imageならリスト / videoなら {"start":..,"end":..} でもOK
    kp3d_source_path=str(spec.temp3d_path(project_root, session_id, 0)),
    status="success",  # "skipped" / "failed" なども可
)

manifest_path = spec.manifest_path(project_root, animal, session_id)
m.save_json(manifest_path)
print("saved:", manifest_path)
```

---

## 8. 実運用での典型フロー

1. `RenderSpec.from_yaml()` で spec 読み込み
2. Dataset/Session から kp3d を取り出して temp3d を生成（別処理）
3. `spec.get_views()` で view を列挙
4. `spec.image_paths()` or `spec.video_paths()` で出力パスを決定
5. Blender 実行（別処理）
6. `RenderManifest` を保存

---

## 9. よくある落とし穴

* **env_camera で config が見つからない**

  * `PROJECT_ROOT/config/marmo*/config.yaml` が存在するか確認
* **config の解像度フィールドが違う**

  * `render_spec.yaml` の `resolution.field`（例：`img_size`）と、実際の config のキーを一致させる
* **Camera.{view_index} が Blender の実体名と合わない**

  * `rendering.profiles.env_camera.views.camera_pattern` を Blender 側命名に合わせる

---

必要なら、次はこの `RenderSpec` を受け取って **Blenderを起動して実際にレンダリングする `RenderRunner` クラス**（`pipeline.steps` を順に実行するやつ）も一緒に組めます。
