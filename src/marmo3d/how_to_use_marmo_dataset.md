## marmo_dataset.py の使い方（Markdown）

このドキュメントは、`dataset_spec.yaml` を元に作った `marmo_dataset.py` を使って

* データセットをロードする
* セッション（撮影1単位）を列挙する
* 完全性チェックで弾く（skip log を残す）
* `kp3d` / `kp2d` の成果物パスを取る
* 健康状態（healthy/diseased）で絞る
* 24fps→25fpsのフレーム変換をする

までを一通りできるようにするための手順です。

---

## 1. 前提

### 必要ファイル

プロジェクト内に最低限これがある想定です。

* `dataset_spec.yaml`
* （任意）`health_status.yaml`（`dataset_spec.yaml` で enabled の場合）
* `marmo_dataset.py`

### Python 依存

* Python 3.10+ 推奨
* PyYAML

```bash
pip install pyyaml
```

---

## 2. 基本の使い方

### 2.1 Dataset をロードする

```python
from pathlib import Path
from marmo_dataset import MarmoDataset

PROJECT_ROOT = Path(".").resolve()
spec_path = PROJECT_ROOT / "dataset_spec.yaml"

ds = MarmoDataset.from_yaml(spec_path, project_root=PROJECT_ROOT)
print("video_root:", ds.video_root)
print("results_root:", ds.results_root)
print("results_subdir:", ds.results_subdir)
```

* `video_root` は `video_roots` のうち **存在する最初のもの**が選ばれます
* `results_subdir` は `results_subdirs` の **優先順**で存在するものが選ばれます

---

## 3. セッションを列挙する

### 3.1 まず session_id の一覧を取る（results 側から探索）

```python
session_ids = ds.discover_session_ids()
print("sessions:", len(session_ids))
print(session_ids[:5])
```

> `results_root/results_subdir/*` のディレクトリ名を `session_id` と見なして、`session_id_regex` に一致するものだけを返します。

---

## 4. 完全性チェックつきでセッションを回す

### 4.1 OK なセッションだけ列挙（おすすめ）

```python
for s in ds.iter_sessions(valid_only=True, log_skips=True):
    print(s.session_id, "kp3d:", s.kp3d_path())
```

* `valid_only=True`：完全性チェックに通ったものだけ yield
* `log_skips=True`：落ちたセッションは `skip_log.jsonl` に理由を書きます

### 4.2 “壊れていても” 全部回したい場合

```python
for s in ds.iter_sessions(valid_only=False, log_skips=True):
    rep = s.check_integrity()
    print(s.session_id, rep.ok, rep.failed_checks)
```

---

## 5. セッションを直接指定して扱う

### 5.1 1つの session_id から Session を作る

```python
sid = "dailylife_cj611_20250101_123456"  # 例
s = ds.session(sid)

print("animal:", s.animal)
print("date:", s.date)
print("time:", s.time)
```

---

## 6. 成果物（artifact）を取る

### 6.1 kp3d（優先順位で自動選択）

```python
kp3d = s.kp3d_path()
print(kp3d)
```

`dataset_spec.yaml` の `kp3d.path_templates` に従い候補を集めて、

* `prefer_contains`（例: `"kp3d_fxdJointLen"`) を含むファイル名を優先
* なければ `fallback`（デフォルトは `first_match`）

で1つを選びます。

### 6.2 kp2d（複数候補のうち存在するもの全部）

```python
kp2d_list = s.kp2d_paths()
for p in kp2d_list:
    print(p)
```

---

## 7. 動画（8台カメラ）へのアクセス

### 7.1 cam オブジェクトを作ってフォルダや chunk を触る

```python
cam1 = s.cam("1")   # camid は dataset_spec.yaml の allowed_ids に合わせる
print("folder:", cam1.folder)

print("metadata:", cam1.metadata_path(ds.spec))
print("mp4 chunks:", cam1.mp4_chunks(ds.spec)[:3])
print("npz chunks:", cam1.npz_chunks(ds.spec)[:3])
```

---

## 8. 完全性チェック（Integrity Check）の中身を見る

```python
rep = s.check_integrity()
print("ok:", rep.ok)
print("missing_camids:", rep.missing_camids)
print("missing_files:", rep.missing_files)
print("failed_checks:", rep.failed_checks)
```

典型的に落ちるポイント：

* 8台揃ってない（cam フォルダが無い）
* `metadata.yaml` が無い
* `mp4` はあるのに `npz` 数が一致しない
* `kp3d*.pickle` が無い
* （有効なら）同期ズレが大きい

---

## 9. skip_log.jsonl の見方

`log_skips=True` のとき、`dataset_spec.yaml` の `logging.path` に JSONL で追記されます。

1行の例（イメージ）：

```json
{"timestamp":"2025-12-23T20:01:02+0900","session_id":"...","camid_missing":["7"],"missing_files":[".../metadata.yaml"],"failed_checks":["must_have_all_cameras","must_have_metadata"],"exception":null,"traceback":null}
```

---

## 10. 健康状態（healthy / diseased）で絞る

`dataset_spec.yaml` の `partitioning.health_status.enabled: true` かつ `health_status.yaml` が存在するなら使えます。

### 10.1 セッションの健康状態を取る

```python
print(s.health_status())  # "healthy" / "diseased" / "unknown"
```

### 10.2 Dataset 全体から絞る

```python
sessions = list(ds.iter_sessions(valid_only=True))
diseased = ds.filter(sessions, animal="cj611", health="diseased")
print(len(diseased))
```

---

## 11. 24fps → 25fps のフレーム変換

処理結果が 24fps 前提、保存動画が 25fps 前提のときの変換です。

```python
frame25 = s.frame_24_to_25(240)
print(frame25)
```

---

## 12. よくある使い方サンプル

### 12.1 「kp3d があるセッションだけ」一覧化

```python
sessions = []
for s in ds.iter_sessions(valid_only=True, log_skips=True):
    if s.kp3d_path() is not None:
        sessions.append(s)

print("kp3d sessions:", len(sessions))
```

### 12.2 「ある日付範囲だけ」抽出

```python
sessions = list(ds.iter_sessions(valid_only=True))
subset = ds.filter(sessions, date_range=("20250101", "20250131"))
print(len(subset))
```

---

## 13. 次に足しやすい拡張（おすすめ）

このクラスは “探索と完全性と成果物パス” までを責務にしています。
Blenderレンダや temp3d 書き出しを繋げるなら、`Session` に次のメソッドを追加するのが自然です。

* `sample_frames(n, seed)`
* `export_temp3d(frame, out_template=...)`
* `render_with_blender(render_spec.yaml, frames=...)`

必要なら、あなたの `render_spec.yaml` に合わせてこの拡張部分もそのまま実装して追記します。
