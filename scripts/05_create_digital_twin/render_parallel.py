#!/usr/bin/env python3
"""1つの .blend / 1カメラのフレーム範囲を分割し、GPU 並列でレンダリングする。

デフォルトは warm 運用（render_worker.py）:
  blender を GPU スロット数だけ常駐させ、チャンクを stdin 経由で流し込む。
  .blend ロード・GPU 初期化・BVH 構築がワーカーあたり1回で済む。
--cold を付けると従来どおり 1チャンク = blender 1プロセスで起動し直す。
"""
import argparse
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_worker import (  # noqa: E402
    DEFAULT_CUDA_PATH, auto_threads, make_env, resolve_gpu_list, run_warm_pool,
)

RENDER_SCRIPT = "scripts/05_create_digital_twin/render_range.py"


def build_blender_cmd(
    blend, camera, start, end, out, prefix, resolution=None,
    backend="CUDA", persistent_data=True, script=RENDER_SCRIPT, extra_args=(),
    threads=None,
):
    """cold モード（1チャンク=1プロセス）用の blender コマンド。"""
    cmd = [
        "blender", blend, "-b",
        *(["-t", str(threads)] if threads else []),
        "-P", script, "--",
        "--camera", camera,
        "--start", str(start),
        "--end", str(end),
        "--out", out,
        "--prefix", prefix,
        "--backend", backend,
    ]
    if resolution:
        cmd += ["--resolution", resolution]
    if not persistent_data:
        cmd += ["--no-persistent-data"]
    return cmd + list(extra_args)


def run_one(cmd, gpu_id=None, verbose=False, cuda_path=None):
    """cold モードで blender を1回実行する。"""
    env = make_env(gpu_id, cuda_path)
    tag = f"(GPU={gpu_id})" if gpu_id is not None else ""
    # ここでブロックするのはOK。上はスレッドで並列に動く。
    if verbose:
        print("Running:", " ".join(cmd), tag)
        subprocess.run(cmd, check=True, env=env)
        return
    # 非verbose時は Blender の大量ログを抑制し、失敗時のみ出力する
    result = subprocess.run(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(f"[FAILED {tag}] {' '.join(cmd)}")
        if result.stderr:
            print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def add_worker_args(parser, default_samples=32):
    """warm ワーカー（blender 側 render_range*.py）へ渡す共通オプション。"""
    parser.add_argument("--backend", default="CUDA", choices=["CUDA", "OPTIX"],
                        help="Cycles の GPU バックエンド（デフォルト: CUDA）")
    parser.add_argument("--cuda-path", default=DEFAULT_CUDA_PATH,
                        help=f"このコード実行時にだけ使う CUDA toolkit のパス"
                             f"（グローバル設定は変更しない、デフォルト: {DEFAULT_CUDA_PATH}）")
    parser.add_argument("--no-persistent-data", dest="persistent_data",
                        action="store_false",
                        help="フレーム間でシーンデータを保持しない（デフォルトは保持）")
    parser.add_argument("--optimize", action="store_true",
                        help="デノイザーOptiX・Simplify・indirect clamp などの最適化を"
                             "レンダリング時に適用する（画が変わるのでオプトイン。"
                             "詳細は blender_render/tools/README.md）")
    parser.add_argument("--samples", type=int, default=default_samples,
                        help=f"Cycles サンプル数を上書き（デフォルト: {default_samples}）。"
                             f"0 以下を渡すと .blend の設定をそのまま使う。"
                             f"値は blender_render/tools/run_sample_sweep.sh で検証すること")
    parser.add_argument("--denoiser", default=None,
                        choices=["OPTIX", "OPENIMAGEDENOISE", "NONE"],
                        help="デノイザー（省略時は .blend の設定のまま）")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="GPU が有効化できなくても CPU で続行する（デフォルトはエラー）")
    parser.add_argument("--blender-arg", action="append", default=[],
                        help="render_range*.py へそのまま渡す追加引数（複数指定可）")


def worker_extra_args(args):
    """add_worker_args の内容を blender 側スクリプトの引数列に変換する。"""
    extra = ["--backend", args.backend]
    if not args.persistent_data:
        extra.append("--no-persistent-data")
    if args.optimize:
        extra.append("--optimize")
    if args.samples is not None:
        extra += ["--samples", str(args.samples)]
    if args.denoiser:
        extra += ["--denoiser", args.denoiser]
    if args.allow_cpu_fallback:
        extra.append("--allow-cpu-fallback")
    return extra + list(args.blender_arg)


def add_pool_args(parser):
    """warm プール（ディスパッチャ側）の共通オプション。"""
    parser.add_argument("--threads", type=int, default=None,
                        help="blender 1本あたりの CPU スレッド数（blender -t）。"
                             "省略時は CPU コア数をスロット数で山分けする。0 で無制限"
                             "（無制限だと1プロセスが約17コア相当を占有し、ワーカーを"
                             "並べたときに CPU の取り合いで全体が遅くなる）")
    parser.add_argument("--pin-cpu", action="store_true",
                        help="taskset で各ワーカーを別々のコアに固定する。"
                             "blender は -t を指定しても全コアぶんのスレッドプールを作り"
                             "スピン待ちで CPU を食い潰すため、ワーカーを多数並べる場合は"
                             "これを付けると全体のスループットが安定する")
    parser.add_argument("--cold", action="store_true",
                        help="warm を使わず、チャンクごとに blender を起動し直す（従来方式）")
    parser.add_argument("--restart-every", type=int, default=0,
                        help="ワーカーがこのジョブ数を処理したら blender を起動し直す"
                             "（0=しない。長時間運用でメモリが膨らむ場合の保険）")
    parser.add_argument("--job-timeout", type=float, default=None,
                        help="1チャンクがこの秒数を超えたらワーカーを作り直して再試行")
    parser.add_argument("--progress", action="store_true",
                        help="フレーム単位の進捗も表示する")


def main():
    parser = argparse.ArgumentParser(
        description="entire_start / entire_end を分割して GPU 並列レンダリングする"
                    "（デフォルトは blender 常駐の warm 運用）"
    )

    parser.add_argument("--blend", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--entire_start", type=int, required=True)
    parser.add_argument("--entire_end", type=int, required=True)
    parser.add_argument("--out", default="./renders")
    parser.add_argument("--prefix", default="cam_")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="チャンクサイズ（省略時は warm=スロット×8分割 / cold=スロット均等分割）")
    parser.add_argument(
        "--max_jobs", type=int, default=None,
        help="同時に動かす blender プロセス数（省略時はGPU数×workers_per_gpu）")
    parser.add_argument(
        "--workers_per_gpu", type=int, default=1,
        help="GPU 1台あたりに同時に流す blender プロセス数（デフォルト: 1）。"
             "増やしすぎると VRAM 不足で逆に遅くなるので実測してから上げること")
    parser.add_argument(
        "--gpu", default=None,
        help="使用するGPU ID（カンマ区切り、例: 0,1,2）。指定しない場合は nvidia-smi で自動検出")
    parser.add_argument("--resolution", default=None, help="解像度 WxH（例: 1920x1080）")
    add_worker_args(parser)
    add_pool_args(parser)
    parser.add_argument("--verbose", action="store_true",
                        help="Blender の詳細ログを表示する（デフォルトは抑制）")

    args = parser.parse_args()

    if args.entire_end < args.entire_start:
        raise ValueError("entire_end は entire_start 以上にしてください")
    if args.workers_per_gpu < 1:
        raise ValueError("workers_per_gpu は 1 以上にしてください")

    gpu_list = resolve_gpu_list(args.gpu)
    gpu_count = len(gpu_list)
    parallel_slots = max(1, gpu_count) * args.workers_per_gpu

    if args.max_jobs is None:
        args.max_jobs = parallel_slots

    total_frames = args.entire_end - args.entire_start + 1
    if args.chunk_size is None:
        # warm はチャンクを細かくしても blender の再起動が起きないので、
        # 負荷分散を優先して細かめに割る
        divisor = args.max_jobs * (1 if args.cold else 8)
        args.chunk_size = max(1, math.ceil(total_frames / divisor))

    if args.threads is None:
        args.threads = auto_threads(args.max_jobs)

    print(f"mode: {'cold' if args.cold else 'warm'}, GPU数: {gpu_count}, "
          f"workers_per_gpu: {args.workers_per_gpu}, "
          f"max_jobs: {args.max_jobs}, chunk_size: {args.chunk_size}, "
          f"threads/worker: {args.threads or '無制限'}")

    # まず全ての (start, end) を作る
    chunks = []
    current_start = args.entire_start
    while current_start <= args.entire_end:
        current_end = min(current_start + args.chunk_size - 1, args.entire_end)
        chunks.append((current_start, current_end))
        current_start = current_end + 1

    extra = worker_extra_args(args)

    if args.cold:
        commands = [
            build_blender_cmd(
                args.blend, args.camera, start, end, args.out, args.prefix,
                args.resolution, args.backend, args.persistent_data,
                RENDER_SCRIPT, extra, args.threads,
            )
            for (start, end) in chunks
        ]
        with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
            futures = [
                executor.submit(
                    run_one, cmd,
                    gpu_list[i % len(gpu_list)] if gpu_list else None,
                    args.verbose, args.cuda_path,
                )
                for i, cmd in enumerate(commands)
            ]
            for f in as_completed(futures):
                # エラーがあればここで raise される
                f.result()
        return

    jobs = [
        {
            "blend": args.blend,
            "label": f"{args.camera} {start}-{end}",
            "payload": {
                "camera": args.camera, "start": start, "end": end,
                "out": args.out, "prefix": args.prefix,
                "resolution": args.resolution,
            },
        }
        for (start, end) in chunks
    ]

    total = len(jobs)
    state = {"done": 0}
    lock = threading.Lock()

    def on_result(job, reply, gpu):
        with lock:
            state["done"] += 1
            n = state["done"]
        ok = reply.get("status") == "ok"
        detail = (f"{reply.get('rendered', 0)}枚 / skip {reply.get('skipped', 0)} "
                  f"({reply.get('sec', 0)}s)" if ok else reply.get("message", ""))
        print(f"[{n}/{total}] {'done' if ok else 'FAILED'}: {job['label']} "
              f"(GPU={gpu}) {detail}", flush=True)

    failures = run_warm_pool(
        jobs, args.max_jobs, gpu_list, RENDER_SCRIPT, extra, args.cuda_path,
        verbose=args.verbose, progress=args.progress, on_result=on_result,
        restart_every=args.restart_every, job_timeout=args.job_timeout,
        threads=args.threads, pin_cpu=args.pin_cpu,
    )
    if failures:
        print(f"\n失敗したチャンク {len(failures)} 件:")
        for job, reply in failures:
            print(f"   {job['label']}: {reply.get('message', reply.get('status'))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
