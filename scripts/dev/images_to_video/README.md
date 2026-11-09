# images_to_video — 画像シーケンス → MP4

セッションディレクトリ内のカメラ別 PNG シーケンスを ffmpeg で MP4 化する。

| 区分 | スクリプト | 入力 | 出力 |
|---|---|---|---|
| run | `images_to_video.sh` | `--input-dir`（セッションディレクトリ） | `output_{session}_{cam}.mp4` → `images_to_video_files/` |

`--input-dir / --output-dir / --session / --help` を受け付ける。`--framerate N`（既定 24）あり。
