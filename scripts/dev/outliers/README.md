# outliers — キーポイント位置の外れ値・bbox はみ出し解析

2 系統:
1. **kp3d 外れ値** — 生 kp3d (mm) がグローバル inlier bbox の外に出る (frame,kp) を検出。
2. **Blender bbox はみ出し** — 補正済み CSV（Blender m）が Blender bbox からどれだけ
   はみ出すか／垂直方向のずれをセッション別にランク付け。

全スクリプトは `--input-dir / --output-dir / --session / --help` を受け付ける
（旧 `--csv-dir/--out-dir` も別名として残す）。

## run/（→ outliers_files/）
| スクリプト | 入力(既定) | 出力 |
|---|---|---|
| `analyze_kp3d_outliers.py` | `outputs/3d` (pickle) | `{session}_outlier.json`, `outlier_summary.json` |

## plot/（→ outliers_plot/）
| スクリプト | 入力(既定) | 出力 |
|---|---|---|
| `plot_outlier_deviation_dist.py` | `outliers_files` (JSON) | `deviation_dist.png` |
| `plot_blender_bbox_outlier_dist.py` | `floorZ_files/corrected` (CSV) | `deviation_dist*.png`, `position_dist.png`, `position_hist.npz` |
| `plot_pre_post_compare.py` | `floorZ_files/corrected` + transformedcsv | `position_pre_post.png` |
| `rank_vertical_offset.py` | `floorZ_files/corrected` (CSV) | `vertical_offset_by_session.{csv,png}`, `vertical_dist_by_session.png` |

`plot_blender_bbox_outlier_dist.py` が共有ヘルパー（bbox 定数・位置ヒスト集計）を提供し、
`plot_pre_post_compare.py` と `rank_vertical_offset.py` がそれを import する。
bbox の評価方法は `README.md` 内の本節および各スクリプト docstring を参照。
