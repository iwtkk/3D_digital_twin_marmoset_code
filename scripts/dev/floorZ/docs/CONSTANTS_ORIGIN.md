# `pkl_to_corrected_csv.py` の定数はどこから来たか

対象コード: `scripts/dev/affine/affine_pkl_csv/pkl_to_corrected_csv.py`（51〜78 行目の定数セクション）

このドキュメントは、各定数が

1. **どこで算出されたか**（出どころ）
2. **どういう問題を経て**（背景・経緯）
3. **どういう問題に対処するために**（目的）

設定されたかをまとめる。関連 doc:
[`KDE_AFFINE_QA.md`](KDE_AFFINE_QA.md)（パラメータの動作詳細）、
[`WHY_PER_SESSION.md`](WHY_PER_SESSION.md)（なぜ垂直だけ session ごとか）。

---

## 全体像：定数は 3 グループに分かれる

| グループ | 定数 | 空間 | 算出単位 | 性質 |
|---|---|---|---|---|
| ① 座標系メタ | `KP_NAMES` | — | — | 入力 pickle の仕様 |
| ② グローバル変換 | `BLENDER_BBOX_SCALE` `BLENDER_CENTER` `KP3D_CENTER` `KP3D_NORM_SCALE` | 計測 mm → Blender m | 全 222 session 共通 | 水平・垂直の土台 |
| ③ 垂直 affine 補正 | `PHYSICAL_HEIGHTS_M` `FLOOR_KP_KEYWORDS` `KDE_BW` `MIN_PEAK_GAP_M` `MAX_MATCH_DIST_M` `MAX_AFFINE_SLOPE_DEV` | Blender bZ (m) | session ごとに適用 | 残差補正 |

処理の流れは「② グローバル変換で全 session 一括スケール → その上に ③ で垂直方向だけ
session 個別の affine を残差補正として乗せる」という二段構成。
詳細な経緯は [`WHY_PER_SESSION.md`](WHY_PER_SESSION.md) を参照。

---

## ① 座標系メタ

### `KP_NAMES`（20 キーポイント名）

```python
KP_NAMES = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", ...]
```

- **出どころ**: 入力 `kp3d_fxdJointLen.pickle` の `kp3d` 配列 `(n_animal, n_frame, 20, 3)` の
  index 順そのまま。`analysis_log.md` §1 のキーポイント一覧に対応。
- **目的**: pickle の生配列に列名を与え、`wrist` / `ankle` などを名前で選べるようにするため。
  CSV 列名へは `sanitize()` で小文字化・`_` 化して変換される。

---

## ② グローバル変換パラメータ（全 222 session 共通）

これらは `kp3d`（計測空間 mm）を Blender 空間（m）へ写す**1 セットの共通アフィン**で、
変換式は次の通り（`scale_and_correct` 内）:

```python
half_span = BLENDER_BBOX_SCALE / 2
bX/bY/bZ = (rotated - KP3D_CENTER) / KP3D_NORM_SCALE * half_span + BLENDER_CENTER
```

座標は先に `R = rot_euler(rx=-90, ry=0, rz=180)` で回転され、
その効果は **(x, y, z) → (−x, −z, −y)**（計測の高さ軸 Y が Blender の bZ に反転して入る）。

### `KP3D_CENTER = [-1131.005, -1072.940, 286.610]`（mm）<br>`KP3D_NORM_SCALE = [449.675, 439.560, 378.880]`（mm, 半スパン）

- **出どころ**: **全 222 session・約 1.4 億点**の inlier（各軸 2–98 percentile でフィルタした点）から
  算出した bbox の中心と半スパン。`analysis_log.md` §2 の統計表が一次ソース。
  - 例: 計測 X は span 899.35mm, center 1131.005mm → 回転後 −1131.005
  - 計測 Y（高さ）は span 757.76mm → 半スパン 378.880 が bZ の `NORM_SCALE`
- **経緯**: 当初の本番スクリプト `kp3d_to_blender_csv.py`（旧パラメータ）に対し、
  全 session を集計し直して `test_scale_kp3d_to_blender.py` で更新した値。
  2–98 percentile を採るのは、`analysis_log.md` §2/§9 にある **X≈−10000 等の三角測量誤差
  （特に cj611 の ~930mm 外れ値クラスター）**が bbox を汚染しないようにするため。
- **目的**: session 間でブレない**水平方向（bX, bY）の共通スケール**を与える。
  現行 NORM_SCALE での joint coverage は 89.4%（`analysis_log.md` §5）。

### `BLENDER_BBOX_SCALE = [0.86658, 0.88791, 0.7679]`（m, full span）<br>`BLENDER_CENTER = [0.0, -0.11142, 1.11765]`（m）

- **出どころ**: 変換先である **Blender 上のケージモデルの bounding box**（実測寸法）。
  計測側の bbox（②前項）をこの箱へ写すための写像先定義。
- **検証**: `analysis_log.md` §3 で kp3d bbox の角を変換した結果が Blender bbox 定義と
  bX/bY/bZ すべて一致（✅）することを確認済み。
- **補足（傾き不要の根拠）**: `analysis_log.md` §4 の PCA 平面フィットで床・壁・天井の傾きが
  最大 1.23° に収まり、回転補正は不要と判断されている。

---

## ③ 垂直 affine 補正パラメータ（session ごとに適用）

### 解決しようとしている問題（共通背景）

Blender 上で個体が床から浮いて見える事象が複数 session で観察された
（`analysis_log.md` §6）。診断の結果、**浮きには 2 種類**あることが分かった:

1. **行動由来** — 個体が 1 時間の大半を棚・パーチ上で過ごす実際の行動（補正してはいけない）。
   「浮き」session でも床に最も近づいた瞬間 `floor_y_p99` は床レベル ≈83mm にあった。
2. **session 由来の系統誤差** — session ごとの別撮りマルチカメラ三角測量で、
   カメラ校正の微小ドリフトにより**垂直のスケール・オフセットが session ごとに少しずつ違う**。

②のグローバル変換だけでは ②由来の残差が消えない。per-session affine 係数の 222 session
ばらつきは intercept std ≈ 34mm、slope std 0.032（bZ=1.5m で ≈47mm）に達し、
「数 mm でも浮くと困る」という要求を満たせない（`WHY_PER_SESSION.md`）。
そこで**垂直軸だけ** session ごとに直線フィット `bZ_corrected = a·bZ + b` を乗せる。

水平が共通で済み垂直だけ個別なのは、水平はケージ底面の広がりが session 間で安定する一方、
垂直は床・足場・棒・天井という**既知の物理高さ 4 点でアンカーできる**ため。

### `PHYSICAL_HEIGHTS_M = [0.7337, 1.0179, 1.3035, 1.5016]`（Blender bZ, m）

- **出どころ**: **Blender ケージモデル内のプラットフォーム 4 か所の実測高さ**
  （`vertical_correction_affine_plan.md` / `vertical_correction_status.md` のケージ構造表）。

  | 場所 | bZ (m) |
  |---|---|
  | 床 (floor) | 0.7337 |
  | 足場 (low perch) | 1.0179 |
  | 棒 (high perch) | 1.3035 |
  | 天井 (ceiling) | 1.5016 |

  床値 0.7337 は `BLENDER_CENTER[2] − BLENDER_BBOX_SCALE[2]/2 = 1.11765 − 0.38395` とも一致し、
  ②のグローバル変換と整合している。
- **目的**: 個体はほぼこの 4 か所のいずれかに静止する、という前提のもと、
  KDE で検出した高さピークを**既知の物理高さにアンカー**して affine 係数を決めるための真値。

### `FLOOR_KP_KEYWORDS = ("wrist", "ankle")`

- **出どころ／経緯**: 接地判定に使うキーポイントの選定
  （`vertical_correction_plan.md`「使うキーポイント」）。
  手首(9,10)・足首(15,16)の四肢末端を使い、**尻尾(tail_1, tail_2)は接地しないので除外**。
- **目的**: 各フレームで `min(wrist/ankle の bZ)` を取り「最も床に近い肢端の高さ」を作る。
  この分布のピークがプラットフォーム高さの推定値になる。

### `KDE_BW = 0.03`（KDE バンド幅）／`MIN_PEAK_GAP_M = 0.10`（m）

- **経緯（GMM → KDE への移行）**: 当初計画（`vertical_correction_plan.md`）は
  **GMM(k=4, BIC で k 自動選択)** でピークを推定する方針だった。しかし
  「BIC が k=4 を選んでも 1 か所に密集する」問題（cj611 で確認）があり、
  **隣接ピーク間隔フィルタ `MIN_PEAK_GAP = 0.10m`** で過剰分割を抑える設計に。
  最終実装ではより単純で安定な **KDE + `find_peaks`** に落ち着き、`MIN_PEAK_GAP_M` は
  `find_peaks(distance=...)` に引き継がれた。
- **値の意味**:
  - `KDE_BW=0.03`: 小さいほど細かい山を拾う。プラットフォーム高さの分離を保つ値。
  - `MIN_PEAK_GAP_M=0.10`: 物理高さ間隔（最小 ≈0.18m）より十分小さく、
    かつノイズ由来の近接ピークを 1 つに統合できる距離。
- **目的**: floor KP 高さ分布から、過剰分割せずプラットフォームのピークだけを取り出す。
  （グリッド点数 2000、ピーク高さ閾値 `max*0.03` は `detect_peaks_kde` 内にハードコード。
  詳細は [`KDE_AFFINE_QA.md`](KDE_AFFINE_QA.md) Q1。）

### `MAX_MATCH_DIST_M = 0.10`（m）

- **経緯（決め打ち → Hungarian マッチング）**: 旧版 `fit_affine` は 2〜3 ピーク時に
  「最低ピーク=床／最高ピーク=天井」と**決め打ち**していた。例えばピークが [0.75, 1.30]m
  （実際は床+棒）でも床+天井として回帰し、**slope が過大推定**された
  （`vertical_correction_status.md`）。
  これを `scipy.optimize.linear_sum_assignment`（Hungarian）で**各ピークを最近傍の物理高さに
  最小コスト割り当て**する方式へ修正。`MAX_MATCH_DIST_M` は、割り当てた対が
  物理高さから 0.10m 超離れていたら誤マッチとして**回帰から除外**する閾値。
- **改善実績**（旧版 → 新版, 全 222 session）:

  | 指標 | 旧版（決め打ち） | 新版（Hungarian） |
  |---|---|---|
  | slope mean（理想 1.0） | 1.2040 | **1.0136** |
  | slope std | 0.2590 | **0.1439** |
  | slope warnings | 136 件 (61%) | **28 件 (13%)** |
  | intercept mean（理想 0） | −0.1928 m | **−0.0259 m** |

- **目的**: 誤った高さアンカーによる過補正を防ぎ、訪れていないプラットフォームを
  回帰から自然に外す。ピークが 1 つ以下なら `offset_only` / `no_peaks` にフォールバック
  （[`KDE_AFFINE_QA.md`](KDE_AFFINE_QA.md) Q2 の分岐表）。

### `MAX_AFFINE_SLOPE_DEV = 0.15`

- **出どころ／目的**: `|slope − 1| > 0.15` を `slope_warn=True` として記録するための閾値
  （補正自体は止めず警告のみ）。新版でも slope max ≈2.09 の外れ session が 28 件残り
  （`cj611_20230212` 等、3 ピーク検出で足場位置がずれるケース）、
  これらを後段で見つけるための品質フラグ。

---

## ④ 出力スケール

### `OUTPUT_SCALE = 1 / 1`

- **意味**: 出力 CSV の全座標に乗じる係数。`1/1` は Blender メートルそのまま、
  `1/1000` は mm 相当（床 bZ=0.7337m → 0.0007337）に縮める。
  KDE affine の計算自体は常に Blender bZ (m) 空間で行い、最後に一括スケールするだけ。
- **目的**: Blender 取り込み側のスケール都合に合わせるための出口調整パラメータ。
  デフォルトは等倍。

---

## 系譜まとめ（このスクリプトに至る経緯）

```
旧 kp3d_to_blender_csv.py（旧グローバルパラメータ）
   │  浮き問題の観察（analysis_log §6）
   ▼
test_scale_kp3d_to_blender.py（全222session再集計でグローバル②を更新
   │  + p99ベース offset のみの簡易垂直補正, a=1固定）
   ▼
vertical_correction_plan.md（GMM(k自動) 方針・尻尾除外・MIN_PEAK_GAP の着想）
   ▼
vertical_correction_affine_plan.md（2フェーズ計画・物理高さ4点テーブル）
   ▼
validate_affine_correction.py（Phase 1 検証: 決め打ち→Hungarian へ修正,
   │  MAX_MATCH_DIST 導入, GMM→KDE）
   ▼
pkl_to_corrected_csv.py（Phase 2: pickle パイプライン統合。本ファイル）
```

旧→新の本質的な違いは **「p99 ベースの offset のみ（a=1 固定）」→「KDE ピーク + Hungarian +
affine 回帰（a, b 両方）」**。前者は session ごとのスケール誤差を直せないが、
後者は床〜天井の複数アンカーで scale と offset を同時補正できる。

---

## 関連ファイル

| パス | 役割 |
|---|---|
| `scripts/dev/analysis_log.md` | グローバル②の算出根拠（§2 統計, §3 bbox 一致, §6 浮き問題） |
| `scripts/dev/vertical_correction_plan.md` | 垂直補正の初版方針（GMM・尻尾除外・MIN_PEAK_GAP の起点） |
| `scripts/dev/vertical_correction_affine_plan.md` | 2 フェーズ計画・物理高さ 4 点テーブル |
| `scripts/dev/vertical_correction_status.md` | 決め打ち→Hungarian 修正の効果実績 |
| `scripts/05_create_digital_twin/test_scale_kp3d_to_blender.py` | グローバル②＋旧 p99 垂直補正の実装（前身） |
| `scripts/dev/affine/validate_affine_correction.py` | Phase 1 検証実装（KDE affine の同一ロジック） |
| [`KDE_AFFINE_QA.md`](KDE_AFFINE_QA.md) | KDE パラメータ・affine 回帰の動作詳細 |
| [`WHY_PER_SESSION.md`](WHY_PER_SESSION.md) | なぜ垂直だけ session ごとに補正するか |
