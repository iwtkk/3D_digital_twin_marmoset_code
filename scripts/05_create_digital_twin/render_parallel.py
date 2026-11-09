#!/usr/bin/env python3
"""Split the frame range of one .blend / one camera and render it in parallel on GPUs.

Default is warm mode (render_worker.py):
  one resident blender per GPU slot, fed with chunks via stdin.
  .blend loading, GPU init and BVH build happen only once per worker.
With --cold, blender is restarted for every chunk as before (1 chunk = 1 blender process).
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
    """blender command for cold mode (1 chunk = 1 process)."""
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
    """Run blender once in cold mode."""
    env = make_env(gpu_id, cuda_path)
    tag = f"(GPU={gpu_id})" if gpu_id is not None else ""
    # Blocking here is OK; callers run in parallel threads.
    if verbose:
        print("Running:", " ".join(cmd), tag)
        subprocess.run(cmd, check=True, env=env)
        return
    # Without verbose, suppress Blender's heavy logging and print it only on failure
    result = subprocess.run(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(f"[FAILED {tag}] {' '.join(cmd)}")
        if result.stderr:
            print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def add_worker_args(parser, default_samples=32):
    """Common options passed to warm workers (blender-side render_range*.py)."""
    parser.add_argument("--backend", default="CUDA", choices=["CUDA", "OPTIX"],
                        help="Cycles GPU backend (default: CUDA)")
    parser.add_argument("--cuda-path", default=DEFAULT_CUDA_PATH,
                        help=f"CUDA toolkit path used only while this code runs"
                             f" (global settings are not changed; default: {DEFAULT_CUDA_PATH})")
    parser.add_argument("--no-persistent-data", dest="persistent_data",
                        action="store_false",
                        help="Do not keep scene data between frames (default: keep)")
    parser.add_argument("--optimize", action="store_true",
                        help="Apply optimizations such as OptiX denoiser, Simplify and indirect clamp "
                             "at render time (opt-in because it changes the image; "
                             "see blender_render/tools/README.md)")
    parser.add_argument("--samples", type=int, default=default_samples,
                        help=f"Override Cycles sample count (default: {default_samples}). "
                             f"A value <= 0 uses the .blend setting as is. "
                             f"Validate the value with blender_render/tools/run_sample_sweep.sh")
    parser.add_argument("--denoiser", default=None,
                        choices=["OPTIX", "OPENIMAGEDENOISE", "NONE"],
                        help="Denoiser (default: keep the .blend setting)")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="Continue on CPU even if no GPU can be enabled (default: error)")
    parser.add_argument("--blender-arg", action="append", default=[],
                        help="Extra arguments passed as is to render_range*.py (repeatable)")


def worker_extra_args(args):
    """Convert the add_worker_args options into an argument list for the blender-side script."""
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
    """Common options for the warm pool (dispatcher side)."""
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads per blender process (blender -t). "
                             "By default, CPU cores are divided evenly among slots. 0 = unlimited "
                             "(unlimited lets one process take about 17 cores, and with many workers "
                             "CPU contention slows everything down)")
    parser.add_argument("--pin-cpu", action="store_true",
                        help="Pin each worker to separate cores with taskset. "
                             "Even with -t, blender creates a thread pool for all cores "
                             "and burns CPU in spin waits, so with many workers "
                             "this option makes overall throughput stable")
    parser.add_argument("--cold", action="store_true",
                        help="Do not use warm mode; restart blender for every chunk (old behavior)")
    parser.add_argument("--restart-every", type=int, default=0,
                        help="Restart blender after a worker has processed this many jobs "
                             "(0 = never; a safeguard against memory growth in long runs)")
    parser.add_argument("--job-timeout", type=float, default=None,
                        help="Recreate the worker and retry if a chunk takes longer than this many seconds")
    parser.add_argument("--progress", action="store_true",
                        help="Also show per-frame progress")


def main():
    parser = argparse.ArgumentParser(
        description="Split entire_start / entire_end and render in parallel on GPUs "
                    "(default: warm mode with resident blender processes)"
    )

    parser.add_argument("--blend", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--entire_start", type=int, required=True)
    parser.add_argument("--entire_end", type=int, required=True)
    parser.add_argument("--out", default="./renders")
    parser.add_argument("--prefix", default="cam_")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Chunk size (default: warm = slots x 8 splits / cold = even split across slots)")
    parser.add_argument(
        "--max_jobs", type=int, default=None,
        help="Number of concurrent blender processes (default: number of GPUs x workers_per_gpu)")
    parser.add_argument(
        "--workers_per_gpu", type=int, default=1,
        help="Number of concurrent blender processes per GPU (default: 1). "
             "Too many causes VRAM shortage and slows down; measure before increasing")
    parser.add_argument(
        "--gpu", default=None,
        help="GPU IDs to use (comma-separated, e.g. 0,1,2). If omitted, detected with nvidia-smi")
    parser.add_argument("--resolution", default=None, help="Resolution WxH (e.g. 1920x1080)")
    add_worker_args(parser)
    add_pool_args(parser)
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose Blender logs (default: suppressed)")

    args = parser.parse_args()

    if args.entire_end < args.entire_start:
        raise ValueError("entire_end must be >= entire_start")
    if args.workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be >= 1")

    gpu_list = resolve_gpu_list(args.gpu)
    gpu_count = len(gpu_list)
    parallel_slots = max(1, gpu_count) * args.workers_per_gpu

    if args.max_jobs is None:
        args.max_jobs = parallel_slots

    total_frames = args.entire_end - args.entire_start + 1
    if args.chunk_size is None:
        # In warm mode, small chunks do not cause blender restarts,
        # so split finely to favor load balancing
        divisor = args.max_jobs * (1 if args.cold else 8)
        args.chunk_size = max(1, math.ceil(total_frames / divisor))

    if args.threads is None:
        args.threads = auto_threads(args.max_jobs)

    print(f"mode: {'cold' if args.cold else 'warm'}, GPUs: {gpu_count}, "
          f"workers_per_gpu: {args.workers_per_gpu}, "
          f"max_jobs: {args.max_jobs}, chunk_size: {args.chunk_size}, "
          f"threads/worker: {args.threads or 'unlimited'}")

    # First build all (start, end) pairs
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
                # Errors are raised here if any
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
        detail = (f"{reply.get('rendered', 0)} frames / skip {reply.get('skipped', 0)} "
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
        print(f"\nFailed chunks ({len(failures)}):")
        for job, reply in failures:
            print(f"   {job['label']}: {reply.get('message', reply.get('status'))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
