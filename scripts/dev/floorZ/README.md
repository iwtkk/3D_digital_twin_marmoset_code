# floorZ — 垂直 affine 補正

各フレームの床接地 KP (wrist/ankle) の `min(bZ)` 分布を KDE にかけてピークを検出し、
物理高さ `[0.7337, 1.0179, 1.3035, 1.5016] m`（床/足場/棒/天井）に affine
`bZ_corrected = a·bZ + b` で合わせるパイプライン。

全スクリプトは `--input-dir / --output-dir / --session / --help` を受け付ける
（METHOD_SPEC 準拠）。既定の出力先は run が `scripts/dev/floorZ_files/`、
plot が `scripts/dev/floorZ_plot/`。

## run/（アルゴリズム・データ生成 → floorZ_files/）

| スクリプト | 入力(既定) | 出力 |
|---|---|---|
| `floorz_core.py` | — | 共有ライブラリ（CLI なし） |
| `pkl_to_corrected_csv.py` | `outputs/3d` (pickle) | `csv/`, `corrected/`, `affine_log.csv` |
| `validate_affine.py` | `outputs/animate_temp/test_scale` | `affine_summary.csv` |
| `estimate_floor_y.py` | `outputs/3d` (pickle) | `y_corrections.csv`（旧 kp3d Y 軸手法） |

## plot/（可視化 → floorZ_plot/）

| スクリプト | 入力(既定) | 出力 |
|---|---|---|
| `plot_corrected.py` | `outputs/3d` | `grid_after_correction.png` (+`--before-after`) |
| `plot_before_grid.py` | `../ez2use/.../transformedcsv` | `height_dist_grid_before.png` |
| `plot_before_after_session.py` | `../ez2use/.../transformedcsv` | `before_after_{session}.png` |
| `plot_affine_validation.py` | `outputs/animate_temp/test_scale` | `height_dist_grid.png`, `affine_summary_plot.png` |
| `plot_floor_y.py` | `floorZ_files/y_corrections.csv` + pickle | `floor_estimate_summary.png` 他 |
| `plot_floor_estimate_corrected.py` | `floorZ_files/corrected` | `floor_estimate_corrected.png` |
| `make_affine_regression_plot.py` | `outputs/3d` | `affine_regression_explained.png`（説明図） |
| `make_distribution_kde_check.py` | `outputs/3d` | `what_is_happening_kde_fixed.png`（説明図） |
| `make_what_is_happening.py` | `outputs/3d` | `what_is_happening_kde.png`（説明図） |

KDE bandwidth / マッチ距離などのパラメータと座標変換定数はすべて
`run/floorz_core.py` に集約。アルゴリズムを変えるときはここを編集する。
背景の議論は `docs/` を参照。

実行例:
```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/floorZ/run/pkl_to_corrected_csv.py
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/floorZ/plot/plot_corrected.py --before-after
```
