# scripts/dev — 解析スクリプト

各 target は `docs/METHOD_SPEC.md` の規約に従い `run/`（アルゴリズム・データ生成）と
`plot/`（可視化）に分かれる。全スクリプトは `--input-dir / --output-dir / --session /
--help` を受け付ける。既定の出力先は run が `scripts/dev/{target}_files/`、plot が
`scripts/dev/{target}_plot/`。

| target | 内容 | 詳細 |
|---|---|---|
| `floorZ` | 床接地 KP の KDE+affine による垂直補正 | [README](floorZ/README.md) |
| `teleport` | 物理的にあり得ないキーポイントの跳びの検出 (v1/v2) | [README](teleport/README.md) |
| `kinematics` | 速度/加速度/ジャーク分布・PSD・ボーン長安定性 | [README](kinematics/README.md) |
| `outliers` | kp3d / Blender bbox はみ出しと垂直ずれランキング | [README](outliers/README.md) |
| `bbox` | 補正前 Blender bbox はみ出し集計 | [README](bbox/README.md) |
| `quality_score` | kp3d_score / kp3d_err の分布把握 | [README](quality_score/README.md) |
| `strip` | ファイル名マーカー除去ユーティリティ | [README](strip/README.md) |
| `images_to_video` | 画像シーケンス → MP4 | [README](images_to_video/README.md) |

典型的な流れは run でデータ（CSV/JSON/NPZ）を `{target}_files/` に生成し、plot がそれを
読んで図を `{target}_plot/` に出力する。共有アルゴリズムは各 target の `run/*_core.py` /
`*_common.py` に集約。
