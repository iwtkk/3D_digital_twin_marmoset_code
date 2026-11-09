# KDE の推定に使った分布は何か（どのキーポイントの何か）

対象コード: `pkl_to_corrected_csv.py`
関連: [`KDE_AFFINE_QA.md`](KDE_AFFINE_QA.md)（KDE パラメータ）、
[`REGRESSION_EXPLAINED.md`](REGRESSION_EXPLAINED.md)（回帰）。

---

## 結論（一言で）

> KDE に入れているのは **「各フレームの floor KP（手首L/R・足首L/R）の bZ の最小値」=
> floor bZ の分布**（1–99 percentile でトリム済み）。

- **キーポイント**: `wrist`（手首 L/R）と `ankle`（足首 L/R）の **4 点だけ**。
  （nose や tail などは使わない）
- **その 4 点の何**: Blender 高さ軸 **bZ（m）**。
- **集約**: 各フレームで 4 点の **min**（いちばん低い肢端 1 個）を取り、その全フレーム分布を KDE にかける。

---

## どのキーポイントか（4 点の特定）

接地に使うキーポイントは定数で定義:

```python
# pkl_to_corrected_csv.py:69
FLOOR_KP_KEYWORDS = ("wrist", "ankle")
```

CSV 列名でいうと、この 2 語を含む `_z` 列＝次の 4 列にマッチする:

| キーポイント | pickle index | CSV 列 |
|---|---|---|
| L wrist | 9  | `pts_l_wrist_z` |
| R wrist | 10 | `pts_r_wrist_z` |
| L ankle | 15 | `pts_l_ankle_z` |
| R ankle | 16 | `pts_r_ankle_z` |

（説明図スクリプト `make_what_is_happening.py` では同じ 4 点を
`FLOOR_IDX = [9, 10, 15, 16]` として pickle index で直接指定している。）

---

## プログラムのどこにあるか

### ① 分布を作る場所 — `scale_and_correct()`（293〜301 行）

```python
# floor KP の bZ 分布から KDE affine を計算
z_cols = [c for c in df.columns
          if c.endswith("_z") and any(k in c for k in FLOOR_KP_KEYWORDS)]  # ← 4列を選ぶ
if z_cols:
    vals_all = df[z_cols].min(axis=1).values                # ← 各フレームで min = floor bZ
    lo, hi   = np.percentile(vals_all, 1), np.percentile(vals_all, 99)
    vals     = vals_all[(vals_all >= lo) & (vals_all <= hi)] # ← 1–99%ile トリム
    peaks    = detect_peaks_kde(vals)                        # ← この vals を KDE へ
    a, b, status = fit_affine(peaks)
```

ここに来る前に `df` は既に「回転 → グローバル変換」で **bZ（Blender m）** になっている
（284〜291 行のグローバルスケール適用）。

### ② KDE 本体 — `detect_peaks_kde()`（178〜189 行）

```python
def detect_peaks_kde(vals):                 # vals = floor bZ（トリム済み）
    kde = gaussian_kde(vals, bw_method=KDE_BW)
    xgrid = np.linspace(vals.min(), vals.max(), 2000)
    density = kde(xgrid)
    ...
    peak_idx, _ = find_peaks(density, height=density.max()*0.03, distance=min_dist)
    return np.sort(xgrid[peak_idx])         # → プラットフォーム高さの候補ピーク
```

`vals`（= floor bZ 分布）が唯一の入力。KDE はこの 1 次元分布の山（モード）を
プラットフォーム高さの推定値として返す。

> 補足: 説明図 `make_what_is_happening.py` の `floor_series()`（61〜69 行）も
> 同じ計算（回転 → bZ 変換 → `np.nanmin(bz[:, FLOOR_IDX], axis=1)`）をしている。

---

## なぜこの分布を使ったのか

### 1. なぜ手首・足首なのか（接地点だから）

- マーモセットは床・足場・棒・天井のいずれかに **四肢で接地して静止**する。
  接地している肢端（手首・足首）の高さ＝その**プラットフォームの高さ**になる。
- 鼻・目・肩・腰・背・尻尾などは体格や姿勢で床から浮くため、高さの基準にならない。
- とくに **尻尾（tail_1, tail_2）は接地しない**ので `FLOOR_KP_KEYWORDS` から除外している。

### 2. なぜ 4 点の min なのか（最も接地に近い肢を拾う）

- 4 肢のうち実際に接地しているのは一部のこともある（片手を上げている等）。
- フレームごとに **最小 bZ** を取れば「最も下＝最も接地に近い肢端」を選べる。
- これにより「体は持ち上がっていても接地肢があるフレーム」で接地高さを拾える。
  （bZ は回転後の高さで、値が小さいほど床に近い。`min` が床側。）

### 3. なぜ 1–99 percentile でトリムするのか（外れ値除去）

- kp3d は session ごとのマルチカメラ三角測量で、ジャンプ中や推定誤差で
  極端な bZ（`analysis_log.md` §9 の ~930mm 外れ値クラスター等）が混じる。
- 上下 1% を落とすことで、これらが KDE のピーク検出を汚すのを防ぐ。

### 4. なぜ「分布のピーク」を見るのか（補正のアンカー）

- 個体が 4 か所に静止するため、floor bZ の分布は**プラットフォーム高さに山ができる**。
- その山（KDE ピーク）を既知の物理高さ `PHYSICAL_HEIGHTS_M = [0.7337, 1.0179, 1.3035, 1.5016]`
  に合わせる affine `bZ_corrected = a·bZ + b` を引くのが目的。
- 「いま見えている接地高さ」と「あるべき物理高さ」のズレを、この分布から読み取っている
  （詳細は [`REGRESSION_EXPLAINED.md`](REGRESSION_EXPLAINED.md)）。

---

## まとめ表

| 項目 | 内容 |
|---|---|
| KDE への入力 | floor bZ 分布（トリム済み 1D） |
| 構成キーポイント | wrist L/R, ankle L/R の 4 点（`FLOOR_KP_KEYWORDS`） |
| 量 | Blender 高さ bZ（m） |
| 集約 | 各フレームで 4 点の `min` → 全フレーム分布 |
| 前処理 | 1–99 percentile トリム |
| コード | `scale_and_correct()` 293〜301 行 / `detect_peaks_kde()` 178〜189 行 |
| 目的 | 分布のピーク＝プラットフォーム高さを検出し、物理高さに affine で合わせる |
