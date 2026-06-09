# KDE パラメータ と affine 回帰 についての Q&A

対象コード: `scripts/dev/affine_pkl_csv/pkl_to_corrected_csv.py`
（`scripts/dev/validate_affine_correction.py` も同一ロジック）

---

## Q1. KDE の変数はどこで決めている？

ファイル冒頭の定数セクション（`pkl_to_corrected_csv.py` の 68〜73 行目）で定義している。

```python
# KDE + affine パラメータ
FLOOR_KP_KEYWORDS  = ("wrist", "ankle")   # 接地判定に使う KP（尻尾は除外）
KDE_BW             = 0.03   # KDE のバンド幅 (gaussian_kde の bw_method)
MIN_PEAK_GAP_M     = 0.10   # 隣接ピークの最小間隔 (m)。これ未満は過剰分割として除外
MAX_MATCH_DIST_M   = 0.10   # ピークと物理高さのマッチ許容距離 (m)。超えたら除外
MAX_AFFINE_SLOPE_DEV = 0.15 # |slope - 1| がこれを超えると slope_warn=True
```

### 各変数が実際に使われる場所（`detect_peaks_kde`、178〜189 行目）

```python
def detect_peaks_kde(vals):
    kde = gaussian_kde(vals, bw_method=KDE_BW)                 # ← KDE_BW
    xgrid = np.linspace(vals.min(), vals.max(), 2000)          # 評価グリッド 2000 点（固定）
    density = kde(xgrid)
    min_dist = max(1, int(MIN_PEAK_GAP_M / (xgrid[1]-xgrid[0])))  # ← MIN_PEAK_GAP_M
    peak_idx, _ = find_peaks(density,
                             height=density.max()*0.03,        # 最大密度の 3%（固定）
                             distance=min_dist)                # ← MIN_PEAK_GAP_M 由来
    return np.sort(xgrid[peak_idx])
```

| 変数 | 役割 | 現在値 |
|---|---|---|
| `KDE_BW` | `gaussian_kde(bw_method=...)`。小さいほど分布の細かい山を拾う | 0.03 |
| グリッド点数 | `np.linspace(..., 2000)` ハードコード | 2000 |
| ピーク高さ閾値 | `height=density.max()*0.03` ハードコード（最大の3%未満は無視） | 0.03 |
| `MIN_PEAK_GAP_M` | `find_peaks(distance=...)` に変換。近すぎるピークを1つに統合 | 0.10 m |

**まとめ**: 調整したい主要パラメータ（`KDE_BW`, `MIN_PEAK_GAP_M`）は定数セクションにある。
グリッド点数 2000 とピーク高さ閾値 0.03 は `detect_peaks_kde` 内にハードコードされている。

---

## Q2. a, b の線形回帰は ax + b？

**はい。`bZ_corrected = a * bZ + b`（傾き a, 切片 b の1次式）。**

`fit_affine`（192〜220 行目）で `np.polyfit(..., 1)` により求める。

```python
slope, intercept = np.polyfit(mp_v, mh_v, 1)   # 1次（直線）フィット
#   mp_v = 検出した KDE ピーク位置 (x)
#   mh_v = それにマッチした物理高さ (y)
#   → y = slope * x + intercept
```

- `x` = 補正前の KDE ピーク位置（Blender bZ, m）
- `y` = 対応する物理高さ（[0.7337, 1.0179, 1.3035, 1.5016] のうちマッチしたもの）
- `np.polyfit(x, y, 1)` の戻り値 `(slope, intercept)` がそのまま `(a, b)`

補正の適用（`scale_and_correct` 内）:

```python
df[col] = a * df[col] + b   # bZ 列に対して
```

### ピーク数による分岐（a, b の決まり方）

| 有効ピーク数 | a, b の決め方 | status |
|---|---|---|
| 4 | `np.polyfit` で全点フィット | full_affine |
| 2〜3 | `np.polyfit` で検出点のみフィット | partial_affine_Npeaks |
| 1 | `a=1.0` 固定、`b = 物理高さ − ピーク`（平行移動のみ） | offset_only |
| 0 | `a=1.0, b=0.0`（補正なし） | no_peaks |

ピークと物理高さの対応付けは Hungarian algorithm（`scipy.optimize.linear_sum_assignment`）で
最小コスト割り当てを行い、距離が `MAX_MATCH_DIST_M`(0.10m) を超えるペアは回帰から除外する。

---

## 補足: なぜ ax+b（affine）なのか

- 単なる平行移動（offset, `a=1` 固定）では「セッションごとのスケール誤差」を補正できない。
- 床と天井など複数の既知高さがあれば、傾き `a`（スケール）と切片 `b`（オフセット）の
  両方を同時に補正できる → より正確に物理高さへ合わせられる。
- ピークが1つしか取れないセッションはスケールを決められないので `a=1` 固定の offset_only に
  フォールバックする。
