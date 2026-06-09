# scripts/dev/ — スクリプト一覧と使い方

作業ディレクトリ: リポジトリルート（`/data/iwata/3D_digital_twin_marmoset_code/`）

---

## affine 補正関連（主要）

### `validate_affine_correction.py`

**Phase 1 検証スクリプト。全222セッションに per-session 垂直 affine 補正を適用し結果を保存する。**

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/validate_affine_correction.py
```

引数なし。実行するだけで全セッションを処理する。

**入力**

```
outputs/animate_temp/test_scale/*_scaletest.csv
  各 CSV は kp3d → Blender 変換済みの bZ 座標 (m) を含む
  wrist/ankle の _z 列をフロアキーポイントとして使用
```

**出力**

| ファイル | 内容 |
|---|---|
| `outputs/dev/affine_validation/affine_summary.csv` | 全セッションの slope/intercept/status/n_peaks/floor_residual_mm |
| `outputs/dev/affine_validation/height_dist_grid.png` | 補正前の全セッション高さ分布グリッド |
| `outputs/dev/affine_validation/affine_summary_plot.png` | slope・intercept・floor残差の分布 |

**アルゴリズム概要**

```
各セッション CSV
  └─ wrist/ankle の _z 列の per-frame min → 高さ分布
       └─ KDE (bw=0.03) + find_peaks でモード検出
            └─ Hungarian algorithm で物理高さ(4か所)に割り当て
                 └─ |peak - physical| > 0.10m のペアを除外
                      └─ np.polyfit → affine (a, b)
                           └─ bZ_corrected = a * bZ + b
```

**フォールバック**

| 有効マッチ数 | status | 処理 |
|---|---|---|
| 4 | full_affine | 全物理高さで線形回帰 |
| 2〜3 | partial_affine_Npeaks | 検出ピークのみで回帰 |
| 1 | offset_only | slope=1.0 固定、最近傍物理高さへ offset |
| 0 | no_peaks | 補正なし (a=1, b=0) |

**重要な定数**

```python
PHYSICAL_HEIGHTS_M = [0.7337, 1.0179, 1.3035, 1.5016]  # 床/足場/棒/天井 (m)
MIN_PEAK_GAP_M     = 0.10   # KDE ピーク間の最小距離
MAX_MATCH_DIST_M   = 0.10   # peak-physical マッチの最大許容距離
KDE_BW             = 0.03   # KDE bandwidth
MAX_AFFINE_SLOPE_DEV = 0.15 # |a-1| がこれを超えると slope_warn=True
```

---

## 補助・調査スクリプト

### `correct_y_offset.py`

**p99 ベースの Y オフセット補正分析（affine 補正の前身）**

```bash
# 全セッション解析
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/correct_y_offset.py

# 単一セッション指定
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/correct_y_offset.py --session cj425_20210427_120000

# 補正値を CSV として保存
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/correct_y_offset.py --apply
```

**入力**: `outputs/3d/*/kp3d_fxdJointLen.pickle`（生 kp3d データ）  
**出力**: `outputs/dev/correct_y_offset/y_corrections.csv`、各種プロット

各セッションの `p99(max(floor_KP_Y per frame))` を床高さの推定値として
Y オフセット補正量を計算する。

結論: p99 補正だけでは床+棚の4ピーク構造を正確に捉えられないため
`validate_affine_correction.py` の affine 補正に置き換えた。

---

### `analyze_kp3d_outliers.py`

**kp3d の外れ値（瞬間移動・異常座標）を解析する**

```bash
python3 scripts/dev/analyze_kp3d_outliers.py
```

**入力**: `outputs/3d/*/kp3d_fxdJointLen.pickle`  
**出力**: `outputs/dev/kp3d_outliers/` 以下にプロットと CSV

---

### `analyze_teleport.py` / `analyze_teleport_v2.py`

**フレーム間の急激な座標変化（テレポート）を検出する**

```bash
python3 scripts/dev/analyze_teleport.py
python3 scripts/dev/analyze_teleport_v2.py  # 改良版
```

**入力**: `outputs/3d/*/kp3d_fxdJointLen.pickle`  
**出力**: テレポート発生フレームの一覧・プロット

---

### `plot_outlier_deviation_dist.py`

**外れ値の偏差分布を可視化する**

```bash
python3 scripts/dev/plot_outlier_deviation_dist.py
```

---

## 実行時の注意

```bash
# sklearn (GaussianMixture) を使うスクリプトは必ずこの環境変数をセット
# (OpenBLAS のスレッド競合によるクラッシュ防止)
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/...
```

`validate_affine_correction.py` は GMM を使わなくなったが、他スクリプトで
`sklearn` を使う場合は同様に設定すること。

---

## 出力ディレクトリ一覧

| ディレクトリ | 生成スクリプト | 内容 |
|---|---|---|
| `outputs/dev/affine_validation/` | `validate_affine_correction.py` | affine 補正の検証結果・プロット一覧 |
| `outputs/dev/correct_y_offset/` | `correct_y_offset.py` | p99 補正量テーブル（フォールバック参照用） |
| `outputs/dev/kp3d_outliers/` | `analyze_kp3d_outliers.py` | 外れ値解析結果 |

プロットファイルの説明は `outputs/dev/affine_validation/IMAGES_README.md` を参照。
