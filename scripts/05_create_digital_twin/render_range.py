"""Blender 側のレンダーワーカー（カラー画像 / PNG）。

2つのモードがある。

  batch (従来)  : --start/--end を1回レンダリングして blender を終了する。
  warm (--serve): .blend を読み込んだまま常駐し、stdin から1行1JSONのジョブ
                  （render_ipc.py 参照）を受け取って連続でレンダリングする。

warm 運用の狙いは、チャンクごとに blender を起動し直すことで毎回払っていた
  - .blend のロード（ジオメトリ・テクスチャ）
  - CUDA/OptiX コンテキスト初期化と GPU へのシーン転送
  - BVH 構築（use_persistent_data はプロセス内でしか効かない）
を、ワーカー1本につき1回に減らすこと。カメラ・出力先・フレーム範囲はジョブごとに
切り替えられるので、1つの .blend の全カメラ・全チャンクを同じプロセスで消化できる。

使い方:
  blender work.blend -b -P render_range.py -- --camera Camera_C --start 1 --end 50 \\
      --out outputs/render/s/Camera_C --prefix s_Camera_C_
  blender work.blend -b -P render_range.py -- --serve   # ディスパッチャから使う
"""
import json
import os
import re
import sys
import time
import traceback

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_ipc import encode  # noqa: E402


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_arg_parser(description, default_prefix="frame_", default_samples=32):
    """warm/batch 共通のオプション定義。depth 版（render_range_depth.py）も使う。"""
    import argparse
    p = argparse.ArgumentParser(description=description)

    # --- ジョブ指定（batch モードのみ必須。warm モードでは stdin のジョブが優先） ---
    p.add_argument("--serve", action="store_true",
                   help="warm モード。stdin から1行1JSONのジョブを受け取り常駐する")
    p.add_argument("--camera", default=None, help="Camera object name")
    p.add_argument("--start", type=int, default=None, help="Start frame")
    p.add_argument("--end", type=int, default=None, help="End frame (inclusive)")
    p.add_argument("--out", default=None, help="Output directory")
    p.add_argument("--prefix", default=default_prefix,
                   help=f"Output filename prefix (default: {default_prefix})")
    p.add_argument("--resolution", default=None, help="解像度 WxH（例: 1920x1080）")
    p.add_argument("--overwrite", action="store_true",
                   help="既に出力ファイルがあっても上書きする（デフォルトはスキップ）")

    # --- デバイス ---
    p.add_argument("--backend", default="CUDA", choices=["CUDA", "OPTIX"],
                   help="Cycles の GPU バックエンド（デフォルト: CUDA）")
    p.add_argument("--allow-cpu-fallback", action="store_true",
                   help="GPU デバイスが1つも有効化できなくても CPU で続行する"
                        "（デフォルトはエラーで停止。CPU への無言フォールバックは"
                        "1フレーム15秒が4分になる最大の罠なので既定では許可しない）")
    p.add_argument("--no-persistent-data", dest="persistent_data", action="store_false",
                   help="フレーム間でシーンデータを保持しない（デフォルトは保持）")

    # --- 画に影響する最適化（blender_render/tools/optimize_blend.py と同じ内容） ---
    p.add_argument("--optimize", action="store_true",
                   help="デノイザーOptiX・Simplify・indirect clamp・adaptive threshold・"
                        "ガラスの影オフをまとめて適用する（画が変わるのでオプトイン）")
    p.add_argument("--samples", type=int, default=default_samples,
                   help=f"Cycles サンプル数を上書き（デフォルト: {default_samples}）。"
                        f"0 以下を渡すと .blend の設定をそのまま使う")
    p.add_argument("--denoiser", default=None,
                   choices=["OPTIX", "OPENIMAGEDENOISE", "NONE"],
                   help="デノイザー（省略時は .blend の設定のまま、--optimize 時は OPTIX）")
    p.add_argument("--adaptive-threshold", type=float, default=None,
                   help="Adaptive Sampling の Noise Threshold（--optimize 時の既定 0.03）")
    p.add_argument("--indirect-clamp", type=float, default=None,
                   help="Light Path の Indirect Clamp（--optimize 時の既定 3.0）")
    p.add_argument("--texture-limit", default=None,
                   help="Simplify のビューポート用テクスチャ上限px（--optimize 時の既定 2048）")
    p.add_argument("--texture-limit-render", default=None,
                   help="Simplify のレンダー用テクスチャ上限px（--optimize 時の既定 4096）")
    p.add_argument("--glass-keyword", action="append", default=[],
                   help="このキーワードを名前に含むマテリアルを持つオブジェクトの"
                        "影レイ可視性を切る（複数指定可、既定: glass window）")
    return p


def parse_args(parser):
    """Blender の引数は `--` 以降がユーザー用。"""
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# 制御メッセージ
# --------------------------------------------------------------------------
def emit(msg):
    """制御メッセージを stdout へ1行出力する（Blender のログと混ざっても判別可能）。"""
    print(encode(msg), flush=True)


def _set(obj, attr, value, log):
    """存在しない/型が違う設定を無視しつつ代入する（Blender のバージョン差対策）。"""
    if value is None or not hasattr(obj, attr):
        return False
    try:
        setattr(obj, attr, value)
    except (TypeError, ValueError) as e:
        log.append(f"{attr}={value!r} 設定不可 ({e})")
        return False
    log.append(f"{attr}={value!r}")
    return True


# --------------------------------------------------------------------------
# シーン設定（ワーカー起動時に1回だけ実行する）
# --------------------------------------------------------------------------
def setup_gpu(backend, allow_cpu_fallback=False):
    """Cycles の GPU バックエンドを有効化し、有効になったデバイス名を返す。

    Blender の Preferences 側 compute_device_type が NONE のままだと、シーンで
    device='GPU' にしても静かに CPU にフォールバックする（blender_render/tools/README.md
    §0）。ここで明示的に設定し、GPU が1つも取れなければ既定ではエラーにする。
    """
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = backend
    prefs.get_devices()

    enabled = []
    for device in prefs.devices:
        # GPU デバイスのみ有効化。CPU デバイスは無効にしてシーン同期に回す
        # （複数ワーカーを同一GPUに載せたときの CPU 競合を抑える）
        use_dev = device.type == backend
        device.use = use_dev
        if use_dev:
            enabled.append(device.name)

    if not enabled:
        msg = (f"backend={backend} の GPU デバイスが1つも見つかりません "
               f"（このまま実行すると CPU にフォールバックして数十倍遅くなります）。"
               f"検出デバイス: {[(d.name, d.type) for d in prefs.devices]}")
        if not allow_cpu_fallback:
            raise RuntimeError(msg)
        print("[WARN] " + msg, flush=True)
        scene.cycles.device = 'CPU'
        return enabled

    scene.cycles.device = 'GPU'
    return enabled


def apply_warm_settings(scene, persistent_data=True):
    """画を変えない範囲の高速化設定。warm/batch 問わず常に適用する。"""
    log = []
    # フレーム間でシーンデータ（BVH等）を保持する。プロセスを維持する warm 運用では
    # チャンクをまたいでも効くので、cold 運用より効果が大きい。
    _set(scene.render, "use_persistent_data", persistent_data, log)
    # デノイズとコンポジットを GPU 側で行い、CPU 律速を避ける
    _set(scene.cycles, "denoising_use_gpu", True, log)
    _set(scene.render, "compositor_device", "GPU", log)
    return log


def apply_optimizations(scene, args):
    """blender_render/tools/optimize_blend.py と同じ「画が変わる」最適化。

    --optimize でプリセット値を、個別オプションが指定されていればそちらを適用する。
    optimize_blend.py と同様、カメラカリングとデシメートには触らない（シーン依存で
    画が壊れることが検証済み）。
    """
    log = []
    cy = scene.cycles
    preset = args.optimize

    denoiser = args.denoiser or ("OPTIX" if preset else None)
    if denoiser == "NONE":
        _set(cy, "use_denoising", False, log)
    elif denoiser:
        if _set(cy, "denoiser", denoiser, log):
            _set(cy, "use_denoising", True, log)
        else:
            # OptiX 非対応GPUなら Blender 側の既定（OIDN）のまま
            log.append(f"denoiser={denoiser} は使えないため既定のまま")

    _set(cy, "samples", args.samples, log)
    _set(cy, "adaptive_threshold",
         args.adaptive_threshold if args.adaptive_threshold is not None
         else (0.03 if preset else None), log)
    _set(cy, "sample_clamp_indirect",
         args.indirect_clamp if args.indirect_clamp is not None
         else (3.0 if preset else None), log)

    tex = args.texture_limit or ("2048" if preset else None)
    tex_render = args.texture_limit_render or ("4096" if preset else None)
    if tex or tex_render:
        _set(scene.render, "use_simplify", True, log)
        _set(cy, "texture_limit", tex, log)
        _set(cy, "texture_limit_render", tex_render, log)

    if preset or args.glass_keyword:
        keywords = [k.lower() for k in (args.glass_keyword or ["glass", "window"])]
        n = disable_glass_shadows(keywords)
        log.append(f"ガラス材質の影レイ可視性をオフ: {n} オブジェクト")
    return log


def disable_glass_shadows(keywords):
    """名前がキーワードに一致するマテリアルを持つオブジェクトの影レイ可視性を切る。"""
    # 後方の単語境界だけ要求する（"Flexiglass" は拾い、"Windows Button" は拾わない）
    pats = [re.compile(re.escape(k) + r"\b") for k in keywords]
    glass_mats = {m.name for m in bpy.data.materials
                  if any(p.search(m.name.lower()) for p in pats)}
    n = 0
    for o in bpy.data.objects:
        if o.type != "MESH" or not o.data:
            continue
        if {s.material.name for s in o.material_slots if s.material} & glass_mats:
            o.visible_shadow = False
            n += 1
    return n


def set_resolution(scene, resolution):
    w, h = resolution.split("x")
    scene.render.resolution_x = int(w)
    scene.render.resolution_y = int(h)
    scene.render.resolution_percentage = 100


def set_camera(scene, name):
    if name not in bpy.data.objects:
        raise ValueError(f"Camera '{name}' not found")
    if scene.camera is None or scene.camera.name != name:
        scene.camera = bpy.data.objects[name]


def job_value(job, args, key, default=None):
    """ジョブの値 > CLI の値 > default の優先順で取り出す。"""
    if job.get(key) is not None:
        return job[key]
    v = getattr(args, key, None)
    return default if v is None else v


# --------------------------------------------------------------------------
# レンダリング本体
# --------------------------------------------------------------------------
def render_color_job(job, args):
    """1ジョブ（=1カメラ・連続フレーム範囲）をレンダリングする。"""
    scene = bpy.context.scene
    set_camera(scene, job_value(job, args, "camera"))

    resolution = job_value(job, args, "resolution")
    if resolution:
        set_resolution(scene, resolution)
    samples = job_value(job, args, "samples")
    if samples:
        scene.cycles.samples = int(samples)

    scene.render.image_settings.file_format = 'PNG'

    out_dir = bpy.path.abspath(job_value(job, args, "out"))  # //relative にも対応
    os.makedirs(out_dir, exist_ok=True)
    prefix = job_value(job, args, "prefix", "frame_")
    start, end = int(job["start"]), int(job["end"])
    overwrite = bool(job_value(job, args, "overwrite", False))

    # タイムライン上のレンジも合わせておく
    scene.frame_start, scene.frame_end = start, end

    rendered = skipped = 0
    for frame in range(start, end + 1):
        filepath = os.path.join(out_dir, f"{prefix}{frame:05d}.png")
        if not overwrite and os.path.exists(filepath):
            skipped += 1
            continue
        t0 = time.time()
        scene.frame_set(frame)
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        rendered += 1
        emit({"event": "frame", "id": job.get("id"), "frame": frame,
              "sec": round(time.time() - t0, 2)})
    return {"rendered": rendered, "skipped": skipped}


# --------------------------------------------------------------------------
# warm ループ
# --------------------------------------------------------------------------
def serve(args, render_job, ready_info=None):
    """stdin から1行1JSONのジョブを受け取り、EOF か quit まで処理し続ける。"""
    emit(dict(ready_info or {}, event="ready", pid=os.getpid()))
    while True:
        line = sys.stdin.readline()
        if not line:  # ディスパッチャが stdin を閉じた
            break
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"status": "error", "message": f"不正なジョブ行: {e}"})
            continue
        if job.get("cmd") == "quit":
            break
        t0 = time.time()
        try:
            stats = render_job(job, args)
            emit(dict(stats, id=job.get("id"), status="ok",
                      sec=round(time.time() - t0, 2)))
        except Exception as e:  # 1ジョブの失敗でワーカーを落とさない
            emit({"id": job.get("id"), "status": "error", "message": str(e),
                  "traceback": traceback.format_exc(),
                  "sec": round(time.time() - t0, 2)})


def run(args, render_job, extra_setup=None):
    """batch/warm 共通のエントリポイント。depth 版もこれを使う。"""
    scene = bpy.context.scene
    if args.samples is not None and args.samples <= 0:
        args.samples = None  # .blend の設定をそのまま使う

    devices = setup_gpu(args.backend, args.allow_cpu_fallback)
    log = apply_warm_settings(scene, args.persistent_data)
    log += apply_optimizations(scene, args)
    if extra_setup:
        log += extra_setup(scene, args) or []
    print(f"[backend={args.backend}] enabled GPU devices: {devices}", flush=True)
    for line in log:
        print(f"  - {line}", flush=True)

    if args.serve:
        serve(args, render_job, {"backend": args.backend, "devices": devices})
        return

    missing = [f"--{k}" for k in ("camera", "start", "end", "out")
               if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"batch モードでは {', '.join(missing)} が必要です"
                         f"（常駐させる場合は --serve）")
    stats = render_job({"start": args.start, "end": args.end}, args)
    print(f"done: rendered={stats['rendered']} skipped={stats['skipped']}", flush=True)


def main():
    parser = build_arg_parser(
        "Render a frame range with a specific camera (batch or warm/--serve)")
    run(parse_args(parser), render_color_job)


if __name__ == "__main__":
    main()
