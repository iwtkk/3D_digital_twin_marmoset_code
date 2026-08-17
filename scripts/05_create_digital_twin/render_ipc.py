"""warm レンダーワーカー（Blender側）とディスパッチャ（Python側）の行JSONプロトコル。

Blender は stdout に大量の進捗ログを吐き、しかも Cycles の進捗行は `\\r` 終端で流れる。
そのため制御メッセージは SENTINEL 付きの1行JSONで区別する。ディスパッチャ側は
universal-newline（`\\r` も行区切りとして扱われる）で読み、SENTINEL で始まる行だけを
プロトコルとして解釈し、それ以外は通常のログとして扱う。

方向ごとのメッセージ:

  dispatcher -> worker (worker の stdin, SENTINEL なしの素の1行JSON)
    {"id": 1, "cmd": "render", "camera": "Camera_C", "start": 1, "end": 50,
     "out": "outputs/render/<session>/Camera_C", "prefix": "<session>_Camera_C_",
     "resolution": "1920x1080", "samples": 128, "overwrite": false}
    {"cmd": "quit"}

  worker -> dispatcher (worker の stdout, SENTINEL 付き)
    {"event": "ready", "pid": 123, "backend": "CUDA", "devices": ["NVIDIA A100"]}
    {"event": "frame", "id": 1, "frame": 7, "sec": 9.8}
    {"id": 1, "status": "ok", "rendered": 40, "skipped": 10, "sec": 320.5}
    {"id": 1, "status": "error", "message": "...", "traceback": "..."}
"""
import json

SENTINEL = "@@RENDER_IPC@@"


def encode(msg):
    """制御メッセージを SENTINEL 付きの1行文字列（改行なし）にする。"""
    return SENTINEL + " " + json.dumps(msg, ensure_ascii=False)


def decode(line):
    """Blender の出力1行を制御メッセージとして解釈する。

    制御メッセージでなければ None を返す（＝通常のログ行）。"""
    line = line.strip()
    if not line.startswith(SENTINEL):
        return None
    try:
        return json.loads(line[len(SENTINEL):])
    except json.JSONDecodeError:
        return None
