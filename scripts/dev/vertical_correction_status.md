# 垂直方向 per-session Affine 補正 — 現在の状況

最終更新: 2026-06-08

---

## 背景

Blender 上でマーモセットの個体が数mm単位で床から浮いて見える問題の解決。
浮きの原因は calibration error ではなく、個体がケージ内の棚・パーチ上に滞在する実際の行動であることが判明。
それでも変換後の座標を物理的に正しい高さに合わせるため、per-session 垂直補正を実装する。

---

## ケージの物理構造

```
天井  (ceiling) : Blender bZ = 1501.6 mm = 1.5016 m
棒   (bar)      : Blender bZ = 1303.5 mm = 1.3035 m
足場 (low perch): Blender bZ = 1017.9 mm = 1.0179 m
床   (floor)    : Blender bZ =  733.7 mm = 0.7337 m
```

個体はこの4か所のいずれかにほぼ静止しており、ジャンプは一時的。

---

## 方針（`scripts/dev/vertical_correction_affine_plan.md` より）

### 使うキーポイント
- 手首 (L/R wrist: index 9,10) と 足首 (L/R ankle: index 15,16) のみ
- 尻尾 (tail_1=18, tail_2=19) は除外（接地しないため）

### アルゴリズム概要

```
既存 CSV (bZ in m, 変換済み)
  └─ min(wrist/ankle bZ) per frame
       └─ GMM (BIC で k=1..4 自動選択) + 隣接ピーク間隔フィルタ (MIN_PEAK_GAP=0.10m)
            └─ 検出ピーク → 物理高さに nearest-neighbor 割り当て (Hungarian algorithm)
                 └─ np.polyfit → affine (slope a, intercept b)
                      └─ bZ_corrected = a * bZ + b
```

### フォールバック

| 検出ピーク数 | 処理 |
|---|---|
| 4 | full_affine (全物理高さで回帰) |
| 2〜3 | partial_affine (検出ピークのみで回帰) |
| 1 | offset_only (slope=1 固定、最近傍物理高さへ offset) |
| 0 | 補正なし・警告 |

---

## Phase 1 実行結果（CSV ベース検証）

### 実行スクリプト
`scripts/dev/validate_affine_correction.py`

### 対象データ
`outputs/animate_temp/test_scale/*_scaletest.csv`  — 222 sessions

### ピーク検出の状況

| status | sessions数 | 割合 |
|---|---|---|
| full_affine (4ピーク) | 31 | 14% |
| partial_affine_3peaks | 109 | 49% |
| partial_affine_2peaks | 8 | 4% |
| offset_only (1ピーク) | 74 | 33% |

**全4プラットフォームを訪れるセッションは14%のみ。**  
多くのセッションで個体は一部の足場しか利用しない。

---

## `fit_affine` 関数の修正（旧版 → 新版）

### 問題
旧版は 2〜3 ピークの場合「最低ピーク → 床、最高ピーク → 天井」と決め打ちしていた。
例: ピークが [0.75, 1.30] m の場合、実際は「床 + 棒」なのに「床 + 天井」として回帰
→ slope が過大推定される。

### 修正
`scipy.optimize.linear_sum_assignment` (Hungarian algorithm) で
各ピークを最近傍の物理高さに割り当てる。

```python
cost = np.abs(peaks[:, None] - PHYSICAL_HEIGHTS_M[None, :])
row_ind, col_ind = linear_sum_assignment(cost)
matched_peaks = peaks[row_ind]
matched_phys  = PHYSICAL_HEIGHTS_M[col_ind]
slope, intercept = np.polyfit(matched_peaks, matched_phys, 1)
```

### 結果の変化

| 指標 | 旧版 | 新版 |
|---|---|---|
| slope mean | 1.2040 | **1.0136** (理想=1.0) |
| slope std | 0.2590 | **0.1439** |
| slope max | 3.2112 | **2.0849** |
| slope warnings (|a-1|>0.15) | 136件 (61%) | **28件 (13%)** |
| intercept mean | -0.1928 m | **-0.0259 m** (理想=0) |
| intercept std | 0.2316 | **0.1208** |
| floor residual 絶対値平均 | 4.07 mm | 32.60 mm ← ⚠️ |

### floor residual 悪化の理由（メトリクスの問題）
`floor_residual` は「peaks[0] が床ピークのはず」という旧来の前提で計算している。
新実装では peaks[0] が床以外の物理高さに割り当てられる場合があるため、
床を訪れないセッションでこのメトリクスが大きく出る。
**slope/intercept の改善は本物。floor_residual メトリクスは要修正。**

---

## 残る課題

### 1. slope max = 2.09 が残る（28件の warnings）
`cj611_20230212` など、3ピーク検出セッションで GMM が実際の足場位置を
正確に捉えられていないケースが存在する。
原因調査が必要（ピーク間隔が物理高さの間隔と大きくずれている可能性）。

### 2. floor_residual メトリクスを修正する
正しい定義: 補正後の bZ において、床ピーク（もし存在すれば）が
0.7337 m にどれだけ近いかを測る。
床を訪れないセッションはこのメトリクスを適用しない。

### 3. Phase 2 への移行判断
slope 改善は十分か、残る28件の警告を許容するかを判断してから
`test_scale_kp3d_to_blender.py` の `scale()` 関数に統合する。

---

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `scripts/dev/analysis_log.md` | 全体分析の記録・変換パラメータの根拠 |
| `scripts/dev/vertical_correction_plan.md` | GMM 方針の初版（尻尾除外・k 自動選択） |
| `scripts/dev/vertical_correction_affine_plan.md` | 2フェーズ計画・物理高さ対応テーブル |
| `scripts/dev/validate_affine_correction.py` | Phase 1 実装スクリプト |
| `outputs/dev/affine_validation/affine_summary.csv` | 全222session の slope/intercept/status |
| `outputs/dev/affine_validation/affine_summary_old.csv` | 旧版（決め打ち）の結果（比較用） |
| `outputs/dev/affine_validation/height_dist_grid.png` | 全session 高さ分布グリッド |
| `scripts/05_create_digital_twin/test_scale_kp3d_to_blender.py` | Phase 2 実装対象 |

---

## 次のアクション（未実施）

- [ ] floor_residual メトリクスを修正して再評価
- [ ] slope warnings 28件の原因調査（GMM フィット不良セッションの可視化）
- [ ] Phase 2: `scale()` 関数への統合
- [ ] 全222session 再処理 + before/after 可視化で補正効果確認
