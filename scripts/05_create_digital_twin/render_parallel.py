#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_blender_cmd(blend, camera, start, end, out, prefix, resolution=None):
    cmd = [
        "blender",
        blend,
        "-b",
        "-P",
        "scripts/05_create_digital_twin/render_range.py",
        "--",
        "--camera",
        camera,
        "--start",
        str(start),
        "--end",
        str(end),
        "--out",
        out,
        "--prefix",
        prefix,
    ]
    if resolution:
        cmd += ["--resolution", resolution]
    return cmd


def run_one(cmd, gpu_id=None):
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print("Running:", " ".join(cmd), f"(GPU={gpu_id})" if gpu_id is not None else "")
    # Blocking here is fine; callers run in parallel threads.
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Split entire_start / entire_end into chunks and run Blender in parallel"
    )

    parser.add_argument("--blend", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--entire_start", type=int, required=True)
    parser.add_argument("--entire_end", type=int, required=True)
    parser.add_argument("--out", default="./renders")
    parser.add_argument("--prefix", default="cam_")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size (default: split evenly by GPU count)")
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=None,
        help="Number of concurrent blender processes (default: GPU count)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU IDs to use (comma-separated, e.g. 0,1,2). Auto-detected via nvidia-smi if omitted",
    )
    parser.add_argument("--resolution", default=None, help="Resolution WxH (e.g. 1920x1080)")

    args = parser.parse_args()

    if args.entire_end < args.entire_start:
        raise ValueError("entire_end must be >= entire_start")

    gpu_list = [g.strip() for g in args.gpu.split(",")] if args.gpu else []

    if not gpu_list:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        gpu_count = len(result.stdout.strip().splitlines())
    else:
        gpu_count = len(gpu_list)

    if args.max_jobs is None:
        args.max_jobs = gpu_count

    total_frames = args.entire_end - args.entire_start + 1
    if args.chunk_size is None:
        args.chunk_size = math.ceil(total_frames / gpu_count)

    print(f"GPU count: {gpu_count}, max_jobs: {args.max_jobs}, chunk_size: {args.chunk_size}")

    # First build all (start, end) pairs
    jobs = []
    current_start = args.entire_start
    while current_start <= args.entire_end:
        current_end = min(current_start + args.chunk_size - 1, args.entire_end)
        jobs.append((current_start, current_end))
        current_start = current_end + 1

    # Build the command list
    commands = [
        build_blender_cmd(
            args.blend,
            args.camera,
            start,
            end,
            args.out,
            args.prefix,
            args.resolution,
        )
        for (start, end) in jobs
    ]

    # Run in parallel (assign GPUs round-robin per job if specified)
    with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        futures = [
            executor.submit(run_one, cmd, gpu_list[i % len(gpu_list)] if gpu_list else None)
            for i, cmd in enumerate(commands)
        ]
        for f in as_completed(futures):
            # Errors are raised here
            f.result()


if __name__ == "__main__":
    main()

