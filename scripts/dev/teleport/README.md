# teleport — 3D キーポイントのテレポート（ジャンプ）解析

各フレームのキーポイント変位から物理的にあり得ない「跳び」を検出し、2D カメラ可視性と
照合して原因を調べる。v1 は単純閾値 (>100mm/frame)、v2 は二段階判定（孤立ジャンプ +
全体極端）。

全スクリプトは `--input-dir / --output-dir / --session / --help` を受け付ける。
run は計算→JSON、plot は JSON を読んで図を描く（一部 timeline / velocity は pickle も読む）。

## run/（→ teleport_files/ ・ teleport_v2_files/）

| スクリプト | 入力(既定) | 出力 |
|---|---|---|
| `teleport_common.py` | — | 共有ライブラリ（ChunkCache 等, CLI なし） |
| `analyze_teleport.py` | `outputs/3d`, `outputs/2d` | `{session}_teleport.json`, `teleport_summary.json` |
| `analyze_teleport_v2.py` | `outputs/3d`, `outputs/2d` | `{session}_teleport_v2.json`, `teleport_summary_v2.json` |

## plot/（→ teleport_plot/ ・ teleport_v2_plot/）

| スクリプト | 入力(既定) | 出力 |
|---|---|---|
| `plot_teleport.py` | `teleport_files` + `outputs/3d` | heatmap / timeline / camera visibility / jump dist |
| `plot_teleport_v2.py` | `teleport_v2_files` + `outputs/3d` | heatmap / kp freq / jump patterns / camera vis / velocity dist |

実行例:
```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/teleport/run/analyze_teleport_v2.py
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/teleport/plot/plot_teleport_v2.py
```
