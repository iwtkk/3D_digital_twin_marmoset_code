# kp3d → Blender 変換パイプライン 分析ログ

## 概要

`outputs/3d/` に保存された kp3d pickle ファイルを Blender 用座標に変換するパイプラインの
構築・検証・分析を行った記録。

---

## 1. データ構造

### kp3d pickle
```
outputs/3d/{session}/kp3d_fxdJointLen.pickle
keys: ['kp3d', 'kp3d_score', 'kp3d_err', 'joint_len']
kp3d.shape: (1, T, 20, 3)  # (n_animal, n_frame, n_keypoint, xyz)
```

### キーポイント一覧（index順）
```
0:nose  1:L_eye  2:R_eye  3:L_ear  4:R_ear
5:L_shoulder  6:R_shoulder  7:L_elbow  8:R_elbow
9:L_wrist  10:R_wrist  11:L_hip  12:R_hip
13:L_knee  14:R_knee  15:L_ankle  16:R_ankle
17:backbone_peak  18:tail_1  19:tail_2
```

### 計測空間の座標系
- **X**: 水平方向
- **Y**: 高さ方向（大きい=床側, 小さい=天井側）  
  - Y_floor ≈ +92.27 mm, Y_ceiling ≈ -665.49 mm
- **Z**: 奥行き方向

### セッション一覧
- 3個体: cj425, cj611, cj357
- 計 222 sessions

---

## 2. kp3d の統計

全 222 session・140,581,923 点の inlier（2-98 percentile フィルタ後）から算出。

| 軸 | min | max | span | center(bbox中点) |
|---|---|---|---|---|
| X | 681.33 mm | 1580.68 mm | 899.35 mm | 1131.005 mm |
| Y | -665.49 mm | 92.27 mm | 757.76 mm | -286.610 mm |
| Z | 633.38 mm | 1512.50 mm | 879.12 mm | 1072.940 mm |

### 外れ値の傾向
- 全点の約 10〜47% が 2-98 percentile 外
- cj611 に外れ値が多い傾向
- X 方向に大きく外れた点（X=-10000 付近）が主

---

## 3. 座標変換パイプライン

### Step 1: 回転（計測座標系 → Blender 座標系）

```python
def rot_euler(rx=0, ry=0, rz=0, order="xyz"):
    mats = {"x": rot_x(rx), "y": rot_y(ry), "z": rot_z(rz)}
    R = np.eye(3)
    for ax in order:
        R = mats[ax] @ R
    return R

R = rot_euler(rx=-90, ry=0, rz=180)
Xa = Xa @ R.T   # shape: (T, 20, 3)
```

**回転の効果: (x, y, z) → (-x, -z, -y)**

| 計測軸 | → | Blender軸 |
|---|---|---|
| X | → | -bX |
| Y (高さ) | → | -bZ |
| Z (奥行き) | → | -bY |

### Step 2: 正規化 → Blender スケール変換

```python
# パラメータ（回転後の座標系に対して）
KP3D_CENTER     = np.array([-1131.005, -1072.940,  286.610])  # mm
KP3D_NORM_SCALE = np.array([  449.675,   439.560,  378.880])  # mm (span/2)
BLENDER_BBOX_SCALE = np.array([0.86658, 0.88791, 0.7679])     # m (full span)
BLENDER_CENTER     = np.array([0.0, -0.11142, 1.11765])        # m

half_span = BLENDER_BBOX_SCALE / 2  # [0.43329, 0.44396, 0.38395] m

# 変換式
kp3d_blender = (kp3d_rotated - KP3D_CENTER) / KP3D_NORM_SCALE * half_span + BLENDER_CENTER
```

### Blender bbox との対応確認

| 軸 | kp3d bbox corners → Blender | Blender bbox 定義 | 一致 |
|---|---|---|---|
| bX | [-433.29, 433.29] mm | [-433.29, 433.29] mm | ✅ |
| bY | [-555.37, 332.54] mm | [-555.37, 332.54] mm | ✅ |
| bZ | [733.7, 1501.6] mm | [733.7, 1501.6] mm | ✅ |

### 実装ファイル
- `scripts/05_create_digital_twin/test_scale_kp3d_to_blender.py`
  - `pickletocsv()`: 回転のみ適用、中間CSVに出力
  - `scale()`: 正規化・スケール変換・Blender座標へ

---

## 4. 傾き検証（PCA 平面フィッティング）

ケージの床・天井・壁が座標軸に対して傾いているか確認。

| 面 | 理想軸 | 傾き角 |
|---|---|---|
| 床 (Y bot 10%) | Y軸 | 0.36° |
| 天井 (Y top 10%) | Y軸 | 0.97° |
| 壁 X− | X軸 | 1.23° |
| 壁 X+ | X軸 | 0.58° |
| 壁 Z− | Z軸 | 0.64° |
| 壁 Z+ | Z軸 | 0.22° |

**→ 全面で最大 1.23° 以内。回転補正は不要。**

---

## 5. 正規化スケールの最適化

全 222 session からサンプリングした点で「何%の kp3d が Blender bbox 内に収まるか」を計算。

### Uniform scaling（全軸同率）での最適 KP3D_NORM_SCALE

| 目標 coverage | Scale factor k | KP3D_NORM_SCALE [bX, bY, bZ] |
|---|---|---|
| 90% | 1.003 | [450.975, 440.831, 379.975] |
| 95% | 1.037 | [466.205, 455.719, 392.808] |
| 99% | 1.124 | [505.388, 494.020, 425.822] |

現在使用中の NORM_SCALE（2-98%ile ベース）での joint coverage: **89.4%**

---

## 6. 浮き問題の分析

### 観察
複数 session で Blender 上の個体が床から約 60cm 浮いて見えた。

### 診断手法
各 session で `floor_y_p99 = p99( max(floor_KP_Y per frame) )` を計算。
- floor_KP: 手首(9,10)・足首(15,16)の4点
- 「session 中で個体が最も床に近づいた Y 座標」の推定値

### 結果

| session タイプ | floor_y_p99 | floor_y_median | 解釈 |
|---|---|---|---|
| 正常 (例: cj425_20210427) | +75〜+97 mm | +25〜+75 mm | 個体がほぼ床にいる |
| 浮き (例: cj425_20210426) | +83 mm | -507 mm | p99 は床レベル・median は棚の高さ |

**→ 「浮き」session も p99 は床レベル付近（83mm）。**  
**→ calibration error ではなく、個体が 1 時間の大半を棚・パーチ上で過ごしている実際の行動。**

### no_contact session（10 sessions）
p99 が -50mm を下回り、session 中に一度も床に接触しなかった session。

```
cj357_20221010_160000  (-209mm)
cj425_20210428_130000  (-507mm)
cj425_20210904_100000  (-505mm)
cj425_20210904_160000  (-508mm)
cj611_20210909_100000  (-472mm)
cj611_20210909_110000  (-511mm)
cj611_20210909_160000  (-516mm)
cj611_20210910_090000  (-513mm)
cj611_20230213_100000  (-230mm)
cj611_20230224_150000  (-507mm)
```

### 補正スクリプト
`scripts/dev/correct_y_offset.py`

```bash
python scripts/dev/correct_y_offset.py          # 全 session 分析
python scripts/dev/correct_y_offset.py --apply  # CSV 出力
```

出力: `outputs/dev/correct_y_offset/`
- `y_corrections.csv`: session ごとの補正量
- `floor_estimate_summary.png`: floor_y_p99 / floor_y_median の一覧
- `y_correction_per_session.png`: session ごとの補正量棒グラフ
- `before_after.png`: 補正前後の floor KP Y 分布比較

**補正量は最大でも ±45mm 程度（no_contact を除く）で、Blender 上の影響は微小。**

---

## 9. kp3d 外れ値の逸脱距離分布

### 定義

各キーポイントが global inlier bbox（section 2 の 2–98 percentile 境界）から何 mm 離れているかを
`deviation_mm` として算出。全 243 session・15,232,537 件のイベントを集計。

### スクリプト

```bash
python scripts/dev/analyze_kp3d_outliers.py      # 外れ値検出・JSON 出力
python scripts/dev/plot_outlier_deviation_dist.py  # 分布プロット生成
```

出力: `outputs/analysis_kp3d_outliers/`
- `{session}_outlier.json`: session ごとの外れ値イベント（teleport JSON と同一構造）
- `outlier_summary.json`: 全 session サマリー
- `deviation_dist.png`: 分布プロット（下記参照）

### deviation_dist.png の見方

全 7 パネルで構成。

```
┌─────────────────────┬─────────────────────┐
│ ① 全イベント        │ ② 個体別            │
│   log-log ヒスト    │   log-log ヒスト    │
├─────────────────────┼─────────────────────┤
│ ③ Zoom 0–200 mm     │ ④ 遠方外れ値 >200mm │
│   線形ヒスト        │   線形ヒスト        │
├─────────────────────┼─────────────────────┤
│ ⑤ 近傍 vs 遠方      │ ⑥ 近傍外れ値 CDF   │
│   個体別 棒グラフ   │                     │
├─────────────────────┴─────────────────────┤
│ ⑦ キーポイント別 箱ひげ図（200mm caps）   │
└───────────────────────────────────────────┘
```

**① 全イベント（log-log）**  
x 軸: bbox 境界からの逸脱距離（mm）、y 軸: イベント件数、どちらも対数スケール。
全体の分布レンジ（0.1mm〜12,000mm）を俯瞰するためのパネル。
ただし log スケールの圧縮により 200–500mm のギャップは視覚的に目立たない（→ ③④ を参照）。

**② 個体別（log-log）**  
①と同じスケールで cj357・cj425・cj611 を重ねて表示。
cj611 の 930mm 付近の山が他個体より突出していることが確認できる。

**③ Zoom 0–200 mm（線形）**  
全外れ値の 97.9% を占める近傍外れ値を線形スケールで拡大。
p50 = 15mm・p95 = 69mm で、指数的減衰型の分布。

**④ 遠方外れ値 >200 mm（線形）**  
件数は少ない（2.1%）が、930mm 付近に鋭い単一スパイクが存在する。
これが推定誤差による「跳び」フレームに対応する。

**⑤ 近傍 vs 遠方 棒グラフ（個体別）**  
斜線なし = 近傍外れ値（0–200mm）、斜線あり = 遠方外れ値（>200mm）。
cj611 が遠方外れ値の件数で圧倒的に多いことが一目でわかる。

**⑥ 近傍外れ値の CDF**  
全外れ値に占める累積割合として表示。95% → 69mm、97.5% → 127mm。
これ以上の逸脱を「フィルタアウトすべき誤差」とみなす際の閾値設定に使用できる。

**⑦ キーポイント別 箱ひげ図**  
各キーポイントの deviation_mm を 200mm でキャップして表示（外れ値非表示）。
赤線 = 中央値、箱 = 25–75 percentile。
`backbone_peak` が他と比べて中央値・IQR ともに大きい。

### 分布の概要（全 session 合計）

| 集団 | 範囲 | 件数 | 割合 |
|---|---|---|---|
| 近傍外れ値 | 0–200 mm | 14,907,488 | 97.9% |
| （ギャップ） | 200–500 mm | 8,253 | 0.05% |
| 遠方外れ値 | 500–1,000 mm | 316,423 | 2.1% |
| 極端外れ値 | >1,000 mm | 375 | 0.002% |

### 二峰性構造

分布には 200–500 mm に明確なギャップがあり、2 つの集団に分離する。

**① 近傍外れ値（0–200 mm）**
- 全外れ値の 97.9%
- 中央値 15 mm・p95 = 69 mm と小さく、bbox 境界からわずかに逸脱した点が大半
- 指数的減衰型の分布（小さい逸脱ほど多い）
- 個体ケージの端に近づいたときの正常な行動の記録が主体

**② 遠方外れ値（~930 mm クラスター）**
- 500–1,000 mm 帯に約 316k 件が集中し、ほぼ 930 mm 付近の単一スパイクを形成
- analysis_log section 2 で言及した「X 方向に大きく外れた点」に相当する推定誤差
- **cj611 が大部分を占める**（個体別ログスケールヒストグラムで顕著な山）

### キーポイント別の傾向

- `backbone_peak` が中央値・上側 whisker ともに突出して大きい（Y 方向の上方逸脱が多い）
- `R_hip`, `L_hip`, `tail_1` はイベント件数が多い（全 session 合計でそれぞれ 200 万件超）
- 四肢末端（L/R_ankle, L/R_knee）も件数は多いが 1 イベントあたりの逸脱量は小さい

### session 別の外れ値率

| session タイプ | 代表例 | outlier_frame_rate |
|---|---|---|
| 正常 | cj357_20221009_090000 | 2% 程度 |
| 多い | cj425_20210427_080000 | 73% |
| 全フレーム | cj611_20230227_080000 | 100% |

外れ値率 100% の session が cj611 に複数存在し、それらが遠方外れ値クラスターの主要因。

---

## 7. 残課題・次ステップ

- [ ] `kp3d_to_blender_csv.py` に正式な変換パラメータを反映（現在は test スクリプトのみ更新済み）
- [ ] 代表ポーズ抽出（k-means on 60D pose vector）による姿勢分析
- [ ] no_contact session の扱い方を決定（除外 or global median 補正）
- [ ] kp3d_score を使ったフィルタリングの検討

---

## 8. 主要ファイル一覧

| ファイル | 役割 |
|---|---|
| `outputs/3d/{session}/kp3d_fxdJointLen.pickle` | 入力 kp3d データ |
| `scripts/05_create_digital_twin/kp3d_to_blender_csv.py` | 本番変換スクリプト（旧パラメータ） |
| `scripts/05_create_digital_twin/test_scale_kp3d_to_blender.py` | テスト用変換スクリプト（新パラメータ適用済み） |
| `scripts/dev/correct_y_offset.py` | Y オフセット補正分析スクリプト |
| `outputs/dev/correct_y_offset/y_corrections.csv` | session ごとの補正量テーブル |
| `scripts/dev/analyze_kp3d_outliers.py` | kp3d 外れ値検出スクリプト（global bbox 境界超えイベントを JSON 出力） |
| `scripts/dev/plot_outlier_deviation_dist.py` | 外れ値逸脱距離分布プロット生成スクリプト |
| `outputs/analysis_kp3d_outliers/{session}_outlier.json` | session ごとの外れ値イベント詳細 |
| `outputs/analysis_kp3d_outliers/outlier_summary.json` | 全 session の外れ値サマリー |
| `outputs/analysis_kp3d_outliers/deviation_dist.png` | 逸脱距離の分布プロット |
