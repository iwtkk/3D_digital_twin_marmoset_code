#!/usr/bin/env python3
"""render_parallel.py の depth 版。

1つの .blend / 1カメラのフレーム範囲を分割し、Cycles の Depth(Z) パスを EXR 出力する。
デフォルトは warm 運用（blender を常駐させてチャンクを流し込む）で、--cold を付けると
従来どおりチャンクごとに blender を起動し直す。CLI は render_parallel.py とほぼ同一。
"""
import argparse
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_parallel import (  # noqa: E402
    add_pool_args, add_worker_args, build_blender_cmd, run_one, worker_extra_args,
)
from render_worker import auto_threads, resolve_gpu_list, run_warm_pool  # noqa: E402

DEPTH_SCRIPT = "scripts/05_create_digital_twin/render_range_depth.py"


def main():
    parser = argparse.ArgumentParser(
        description="entire_start / entire_end を分割して Depth(Z) を GPU 並列で書き出す"
                    "（デフォルトは blender 常駐の warm 運用）"
    )
    parser.add_argument("--blend", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--entire_start", type=int, required=True)
    parser.add_argument("--entire_end", type=int, required=True)
    parser.add_argument("--out", default="./depth")
    parser.add_argument("--prefix", default="cam_")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="チャンクサイズ（省略時は warm=スロット×8分割 / cold=スロット均等分割）")
    parser.add_argument("--max_jobs", type=int, default=None,
                        help="同時に動かす blender プロセス数（省略時はGPU数×workers_per_gpu）")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="GPU 1台あたりの同時 blender 数（デフォルト: 1）")
    parser.add_argument("--gpu", default=None,
                        help="使用するGPU ID（カンマ区切り）。未指定なら nvidia-smi で自動検出")
    parser.add_argument("--resolution", default=None, help="解像度 WxH（例: 1920x1080）")
    # depth は決定的なのでサンプル数の既定は 1（カラーの既定 32 とは別）
    add_worker_args(parser, default_samples=1)
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
        divisor = args.max_jobs * (1 if args.cold else 8)
        args.chunk_size = max(1, math.ceil(total_frames / divisor))

    if args.threads is None:
        args.threads = auto_threads(args.max_jobs)

    print(f"mode: {'cold' if args.cold else 'warm'}, GPU数: {gpu_count}, "
          f"workers_per_gpu: {args.workers_per_gpu}, "
          f"max_jobs: {args.max_jobs}, chunk_size: {args.chunk_size}, "
          f"threads/worker: {args.threads or '無制限'}")

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
                DEPTH_SCRIPT, extra, args.threads,
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
        jobs, args.max_jobs, gpu_list, DEPTH_SCRIPT, extra, args.cuda_path,
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
