# bbox — 補正前 Blender bbox はみ出し分析

補正前 (transformedcsv, global scale のみ適用済み) の Blender 座標が Blender bbox から
どれだけはみ出すかをセッション別に集計する。上下 (z) と水平 (x,y) を分ける。

| 区分 | スクリプト | 入力(既定) | 出力 |
|---|---|---|---|
| run | `analyze_bbox_overflow.py` | `../ez2use/.../transformedcsv` | `bbox_overflow.csv` → `bbox_files/` |
| plot | `plot_bbox_overflow.py` | `bbox_files/bbox_overflow.csv` | `bbox_overflow_top.png` → `bbox_plot/` |

全スクリプトは `--input-dir / --output-dir / --session / --help` 対応。
