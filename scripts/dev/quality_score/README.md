# quality_score — kp3d_score / kp3d_err の分布把握

しきい値判定はせず、検出確信度 score（高いほど良, 0.5 でクランプ）と再投影誤差 err（px,
低いほど良）の分布を記述する。score と err は相補的（低 score=不確実な検出, 高 err=幾何
不整合）なので joint 分布も見る。

| 区分 | スクリプト | 入力(既定) | 出力 |
|---|---|---|---|
| run | `analyze_quality_score.py` | `outputs/3d` | `quality_dist_summary.json`, `quality_subsample.npz` → `quality_score_files/` |
| plot | `plot_quality_score.py` | `quality_score_files` | dist/joint/perkp/heatmap/clamp_nan 図 → `quality_score_plot/` |

全スクリプトは `--input-dir / --output-dir / --session / --help` 対応。
