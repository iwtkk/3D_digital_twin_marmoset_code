# kinematics — 3D キーポイントの運動学・信号品質解析

動きを「信号」として特徴づける: 速度/加速度/ジャークの分布、PSD（周波数別エネルギー）、
（固定のはずの）ボーン長の安定性。全セッションを streaming で集計する。

3D (kp3d_fxdJointLen) は anipose で時間平滑化済みのため、PSD 高周波端は平滑化フィルタの
ロールオフ（生検出ノイズではない）。

全スクリプトは `--input-dir / --output-dir / --session / --help` を受け付ける。

| 区分 | スクリプト | 入力(既定) | 出力 |
|---|---|---|---|
| run | `kinematics_common.py` | — | 共有ライブラリ（CLI なし） |
| run | `analyze_kinematics.py` | `outputs/3d` | `kinematics_accum.npz`, `kinematics_summary.json` → `kinematics_files/` |
| plot | `plot_kinematics.py` | `kinematics_files` | ヒストグラム/PSD/ボーン CV/時系列 → `kinematics_plot/` |

run は累積配列を NPZ に保存し、plot はそれを読んで描画する（pickle 再読み込み不要）。
`--session` は run では時系列プロット用の代表セッション選択。

実行例:
```bash
python3 scripts/dev/kinematics/run/analyze_kinematics.py --session cj425_20210427_120000
python3 scripts/dev/kinematics/plot/plot_kinematics.py
```
