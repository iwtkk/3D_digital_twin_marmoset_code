# strip — ファイル名マーカー除去ユーティリティ

`work_` / `dailylife_` / `_points.csv_transformedtest` を含むファイル名を素の名前に戻す
in-place リネームユーティリティ。既定は dry-run、`--execute` で実行。

| 区分 | スクリプト | 入力 | 動作 |
|---|---|---|---|
| run | `strip_marked_filenames.py` | `--input-dir DIR` | DIR 内のファイル名を書き換え |

`--input-dir / --output-dir / --session / --help` を受け付ける（in-place のため output-dir /
session は未使用）。`--execute` / `--recursive` あり。
