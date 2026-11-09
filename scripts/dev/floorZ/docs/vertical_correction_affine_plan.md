# 垂直方向 per-session Affine 補正 実装計画

## 前提情報

### Blender ケージの物理高さ（実測値）

| 場所 | Blender bZ (mm) | Blender bZ (m) |
|---|---|---|
| 天井 (ceiling) | 1501.6 mm | 1.5016 m |
| 棒 (high perch) | 1303.5 mm | 1.3035 m |
| 足場 (low perch) | 1017.9 mm | 1.0179 m |
| 床 (floor) | 733.7 mm | 0.7337 m |

### 既存の変換パラメータ（`test_scale_kp3d_to_blender.py`）

```python
KP3D_CENTER[2]     = 286.610   # mm, rotated_z の中心
KP3D_NORM_SCALE[2] = 378.880   # mm, rotated_z の半スパン
BLENDER_BBOX_SCALE[2] = 0.7679  # m, full span
BLENDER_CENTER[2]  = 1.11765   # m
half_span[2]       = 0.38395   # m
```

### 座標変換の対応（確認済み）

回転後の座標: `rotated_z = -kp3d_y`（Y軸反転）

物理高さ → rotated_z への逆変換式：

```
rotated_z_mm = (bZ_mm - BLENDER_CENTER_mm) / half_span_mm * KP3D_NORM_SCALE[2] + KP3D_CENTER[2]
             = (bZ_mm - 1117.65) / 383.95 * 378.880 + 286.610
```

| 場所 | Blender bZ (mm) | rotated_z (mm) |
|---|---|---|
| 床 | 733.7 | −92.3 |
| 足場 | 1017.9 | +203.2 |
| 棒 | 1303.5 | +500.1 |
| 天井 | 1501.6 | +665.5 |

### 接地判定に使うキーポイント

- **手首 (wrist)**: index 9 (L), 10 (R)
- **足首 (ankle)**: index 15 (L), 16 (R)
- **尻尾は除外**（tail_1=18, tail_2=19 は接地しないため不適）

---

## アプローチ比較

| | **Option A: CSV ベース** | **Option B: Pickle ベース** |
|---|---|---|
| 入力 | `outputs/animate_temp/test_scale/*.csv` | `outputs/3d/*/kp3d_fxdJointLen.pickle` |
| GMM を実行する空間 | Blender bZ (m) ← 物理高さと直接比較可 | rotated_z (mm) ← 変換して比較 |
| 実装場所 | CSV への後処理スクリプト | `scale()` 関数内に統合 |
| 全 222 session 対応 | 未生成 CSV がある場合は不可 | pickle があれば常に可能 |
| pipeline への組み込み | できない（後処理） | できる（生成時に補正） |
| 実装コスト | 低い（検証・プロトタイプ向き） | やや高い（本番向き） |

---

## 推奨計画：2フェーズ

### Phase 1 — CSV で検証（Option A）

既存の `outputs/animate_temp/test_scale/*.csv` を使ってアルゴリズムを確認する。

**目的**：GMM が安定して4ピークを取れるか・affine 係数が妥当かを検証する。

**処理フロー**：

```
既存 CSV (bZ in m, 変換済み)
  └─ wrist/ankle の _z 列を取得
       └─ min(wrist/ankle bZ) per frame  →  height distribution
            └─ GMM(k=4, BIC で k 自動選択)
                 └─ 4ピーク検出  [p1, p2, p3, p4] m
                      └─ np.polyfit(peaks_m, physical_heights_m, deg=1)
                           └─ affine 係数 (a, b) を確認
                                └─ 補正後の bZ が物理高さに合うか目視確認
```

**実装ファイル（新規作成）**：
`scripts/dev/validate_affine_correction.py`

**確認ポイント**：
- GMM の k が安定して 4 になるか（なりにくい session は何件か）
- affine の slope `a` が 1 に近いか（スケール誤差が小さいか）
- offset `b` の session 間のばらつきがどれくらいか

---

### Phase 2 — Pickle pipeline に統合（Option B）

Phase 1 で検証した方法を `test_scale_kp3d_to_blender.py` の `scale()` 関数に組み込む。

**処理フロー**：

```
pickle → 回転 (Xa @ R.T) → 中間 CSV (rotated mm)
                                  └─ scale() 内で：
                                       [Step 1] wrist/ankle の _z 列から
                                                floor_z_per_frame = min(_z cols) per frame
                                       [Step 2] GMM(k=4) → peaks_mm (rotated_z 空間)
                                       [Step 3] peaks_mm → physical_rotated_z_targets で
                                                np.polyfit → affine (a, b)
                                       [Step 4] rotated_z_corrected = a * rotated_z + b
                                                を scale 関数内で _z 列に適用してから
                                                グローバル Blender スケール変換
                                       [Step 5] GMM が k<4 のみ → 警告 + offset のみ適用
```

**物理高さの rotated_z ターゲット**（定数として定義）：

```python
# Blender 物理高さ (mm) → rotated_z (mm) に変換済み
PLATFORM_HEIGHTS_ROTATED_Z = np.array([-92.3, 203.2, 500.1, 665.5])  # [床, 足場, 棒, 天井]
```

**修正ファイル**：
`scripts/05_create_digital_twin/test_scale_kp3d_to_blender.py`

---

## GMM の設計上の注意点

### k の自動選択
BIC で k=1〜4 を比較。ただし BIC が k=4 を選んでも 1 か所に密集している場合がある（cj611 で確認済み）。

追加チェック：ピーク間の最小間隔が `MIN_PEAK_GAP_MM = 100mm` 以上でなければ過剰分割とみなし k を減らす。

```python
def _valid_peaks(peaks, min_gap=100.0):
    """隣接ピーク間が min_gap mm 以上あるものだけ有効とする"""
    valid = [peaks[0]]
    for p in peaks[1:]:
        if p - valid[-1] >= min_gap:
            valid.append(p)
    return valid
```

### フォールバック戦略

| 検出されたピーク数 | 処理 |
|---|---|
| 4 | 完全 affine 補正 (scale + offset) |
| 2〜3 | 検出ピークのみで regression → offset 補正のみ (`a=1` に固定) |
| 1 | p99 ベース offset のみ（`outputs/dev/correct_y_offset/y_corrections.csv` 参照） |
| 0 | 補正なし・警告出力 |

---

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `scripts/dev/analysis_log.md` | 全体分析の記録・変換パラメータの根拠 |
| `scripts/dev/vertical_correction_plan.md` | GMM 方針の初版（尻尾除外・k 自動選択の根拠） |
| `scripts/dev/correct_y_offset.py` | p99 ベース補正スクリプト（フォールバック用） |
| `outputs/dev/correct_y_offset/y_corrections.csv` | p99 補正量テーブル（フォールバック参照用） |
| `outputs/dev/height_dist_gmm.png` | 12 session の高さ分布可視化（GMM フィット確認済み） |
| `outputs/animate_temp/test_scale/*.csv` | Phase 1 検証用 CSV（変換済み bZ in m） |
| `scripts/05_create_digital_twin/test_scale_kp3d_to_blender.py` | Phase 2 実装対象 |

---

## 次のアクション

1. **Phase 1 を実行**：`scripts/dev/validate_affine_correction.py` を作成・実行して
   GMM の安定性と affine 係数を確認する
2. Phase 1 で問題なければ **Phase 2 へ進む**
3. Phase 2 完了後、全 222 session を再処理して bZ の精度を確認する
