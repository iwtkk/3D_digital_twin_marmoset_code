# 垂直方向 per-session 補正 実装方針

## 背景・問題設定

### ケージ構造
マーモセットケージには4つの高さ基準がある：

1. **床 (floor)** — 最下部
2. **低棚 (low perch)** — 中間・幅は小さい
3. **高棚 (high perch)** — より高く、水平方向に長い棒状
4. **天井 (ceiling)** — 最上部

個体はほぼこの4か所のいずれかに静止しており、フレーム間の移動（ジャンプ）は一時的。

### 問題
- kp3d の垂直軸 (Y) にセッション固有のオフセットが存在する
- Blender 上で個体が数mm単位で浮いて見えるため補正が必要
- 「浮き」は calibration error ではなく行動（棚上）も含むため、
  単純な p99 補正では不十分

### 既存の分析結果
- `scripts/dev/analysis_log.md` に詳細あり
- `outputs/dev/correct_y_offset/y_corrections.csv` に per-session p99 補正量あり（参照用）
- p99 補正量は最大 ±45mm 程度（no_contact 10 sessions を除く）

---

## 実装方針

### 使うキーポイント
- **手首 (L/R wrist)** と **足首 (L/R ankle)** の4点のみ
- 尻尾 (`tail_1`, `tail_2`) は除外（接地しないため床高さ推定に不適）
- kp_index: wrist=[9,10], ankle=[15,16]

### 座標の対応
回転後の rotated_z = -kp3d_y  
→ **rotated_z が小さいほど床に近い**（Y軸が反転）

---

## 処理フロー

```
kp3d raw (mm, T×20×3)
  │
  ├─ 回転 Xa @ R.T  →  rotated (T×20×3)
  │     R = rot_euler(rx=-90, ry=0, rz=180)
  │     効果: (x,y,z) → (-x, -z, -y)
  │
  ├─ [Step 1] 接地候補高さの抽出
  │     floor_z_per_frame = min(rotated_z の wrist/ankle 列, axis=1)   # (T,)
  │     ※ min = rotated_z が最小 = 最も床に近い肢端
  │
  ├─ [Step 2] GMM(k=4) でプラットフォーム高さを推定
  │     from sklearn.mixture import GaussianMixture
  │     gm = GaussianMixture(n_components=4).fit(floor_z_per_frame.reshape(-1,1))
  │     peaks_mm = sorted(gm.means_.ravel())   # [z_floor, z_low, z_high, z_ceil] (mm)
  │     ※ ジャンプ中フレームは各成分の responsibility が低い → 自然に除外
  │
  ├─ [Step 3] Affine 補正係数の計算
  │     物理高さが既知の場合: 線形回帰 peaks_mm → physical_heights_mm
  │       a, b = np.polyfit(peaks_mm, physical_heights_mm, 1)
  │       rotated_z_corrected = a * rotated_z + b
  │
  │     物理高さが未知の場合: 床 (z_floor) を kp3d 床レベル (≈-92mm) に合わせる offset のみ
  │       b = -92.27 - peaks_mm[0]
  │       rotated_z_corrected = rotated_z + b
  │
  └─ [Step 4] グローバルスケーリング適用 (水平方向と共通)
        bZ = (rotated_z_corrected - KP3D_CENTER[2]) / KP3D_NORM_SCALE[2]
             * half_span[2] + BLENDER_CENTER[2]
```

---

## 実装上の判断ポイント

### 1. 物理高さが既知かどうか
| 状況 | 実装 |
|---|---|
| 4か所の物理高さ (mm) が既知 | `np.polyfit` で affine 補正 → scale + offset 両方を補正 |
| 不明 | 床ピーク → `GLOBAL_FLOOR_Y = -92.27mm` に合わせる offset のみ |

物理高さが既知なら affine 補正が精度が高い。

### 2. GMM の k の扱い
- 基本は k=4
- 個体がそのセッションでケージの全高さを使っていない場合、ピークが3つ以下になる可能性
- 対処: BIC で k を 2〜4 の中から自動選択、または k=4 固定で警告

```python
# BIC による k 自動選択の例
best_k, best_bic = 1, np.inf
for k in range(1, 5):
    gm = GaussianMixture(n_components=k, random_state=42).fit(X)
    if gm.bic(X) < best_bic:
        best_k, best_bic = k, gm.bic(X)
```

### 3. no_contact session の扱い
- ピーク数が 1 つしか見つからないなど GMM が収束しない session は `no_contact` フラグ
- フォールバック: `y_corrections.csv` の p99 補正量を使うか、補正なし

### 4. ケージ構造が個体間で異なる場合
- cj425 / cj611 / cj357 でケージが同じか確認が必要
- 異なる場合は individual_id ごとに `physical_heights` を分ける

---

## スクリプト設計

### 対象ファイル
```
scripts/05_create_digital_twin/test_scale_kp3d_to_blender.py
```

### 追加・変更箇所

#### 定数追加
```python
# 垂直補正パラメータ
FLOOR_KP_SUFFIXES = ("wrist", "ankle")   # 接地判定に使うKP
GMM_N_COMPONENTS  = 4                    # プラットフォーム数
GMM_MAX_CORRECTION = 0.05                # 50mm 超の補正は no_contact としてキャップ

# 物理高さ (mm, rotated空間) — 未知の場合は None
# PHYSICAL_HEIGHTS_MM = None
# PHYSICAL_HEIGHTS_MM = [-92.27, -XXX, -XXX, -665.49]  # 既知の場合
PHYSICAL_HEIGHTS_MM = None
```

#### 関数追加: `_estimate_vertical_correction(points_raw)`
```python
def _estimate_vertical_correction(points_raw: pd.DataFrame) -> tuple[float, str]:
    """
    回転後・正規化前の CSV から per-session 垂直補正量 (m) を推定。

    Returns
    -------
    correction_m : float  Blender bZ への加算量 (m)
    status       : str    "ok" / "no_contact" / "gmm_fail"
    """
    # 1. wrist/ankle の _z 列を取得
    floor_z_cols = [c for c in points_raw.columns
                    if c.endswith("_z") and any(kp in c for kp in FLOOR_KP_SUFFIXES)]
    if not floor_z_cols:
        return 0.0, "no_floor_cols"

    # 2. 各フレームの最小 rotated_z (= 最も床に近い肢端)
    floor_z_per_frame = points_raw[floor_z_cols].min(axis=1).values.reshape(-1, 1)

    # 3. GMM で4ピーク推定
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=GMM_N_COMPONENTS, random_state=42)
    gm.fit(floor_z_per_frame)
    peaks_mm = sorted(gm.means_.ravel())   # 小さい順 = 床から天井

    # 4. 床ピーク → 期待値 (GLOBAL_FLOOR_Z_MM) へのオフセット補正
    #    affine が使える場合は np.polyfit(peaks_mm, PHYSICAL_HEIGHTS_MM, 1) に差し替え
    floor_peak_mm = peaks_mm[0]
    GLOBAL_FLOOR_Z_MM = -92.27  # rotated_z での床レベル (= -kp3d_y_floor)

    offset_mm = GLOBAL_FLOOR_Z_MM - floor_peak_mm   # mm

    # Blender m への変換
    half_span = BLENDER_BBOX_SCALE / 2
    correction_m = offset_mm / KP3D_NORM_SCALE[2] * half_span[2]

    status = "ok"
    if abs(correction_m) > GMM_MAX_CORRECTION:
        correction_m = float(np.clip(correction_m, -GMM_MAX_CORRECTION, GMM_MAX_CORRECTION))
        status = "no_contact_capped"

    return correction_m, status
```

#### `scale()` の変更点
- 正規化前に `_estimate_vertical_correction()` を呼び出す
- 水平 (_x, _y) はグローバルパラメータのみ
- 垂直 (_z) はグローバル変換 + per-session correction を加算

---

## 検証方法

### 実装前に確認すべきこと
1. **高さ分布の可視化**:
   `min(wrist/ankle rotated_z) per frame` のヒストグラムを数 session で確認
   → 4ピークが明確に見えるか確認
   ```bash
   python scripts/dev/visualize_height_dist.py --sessions cj425_20210427_190000 cj425_20210426_141115
   ```

2. **GMM のフィット品質**:
   各成分の平均・分散と実際のピーク位置が一致しているか確認

3. **After correction の範囲チェック**:
   補正後の floor KP bZ が Blender 床レベル (0.7337m) に近いか確認

### 期待される補正量
- 正常 session: ±10mm 以下
- 棚上が多い session: 〜±30mm
- no_contact (p99=-500mm など): キャップ適用 → 要手動確認

---

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `scripts/dev/analysis_log.md` | 全体分析の記録・変換パラメータの根拠 |
| `scripts/dev/correct_y_offset.py` | p99ベース補正の分析スクリプト（参考） |
| `outputs/dev/correct_y_offset/y_corrections.csv` | p99補正量テーブル（フォールバック用） |
| `scripts/05_create_digital_twin/test_scale_kp3d_to_blender.py` | 実装対象スクリプト |
