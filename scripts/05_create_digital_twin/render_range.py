"""Blender-side render worker (color images / PNG).

There are two modes.

  batch (old)   : render --start/--end once and exit blender.
  warm (--serve): stay resident with the .blend loaded, receive one JSON job per line
                  from stdin (see render_ipc.py) and render them one after another.

The goal of warm mode is to reduce the costs that were paid every time blender was restarted per chunk
  - loading the .blend (geometry, textures)
  - CUDA/OptiX context init and scene upload to the GPU
  - BVH build (use_persistent_data only works within a process)
to once per worker. Camera, output directory and frame range can be switched per job,
so all cameras and chunks of one .blend can be processed in the same process.

Usage:
  blender work.blend -b -P render_range.py -- --camera Camera_C --start 1 --end 50 \\
      --out outputs/render/s/Camera_C --prefix s_Camera_C_
  blender work.blend -b -P render_range.py -- --serve   # used from the dispatcher
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
    """Option definitions shared by warm/batch. Also used by the depth version (render_range_depth.py)."""
    import argparse
    p = argparse.ArgumentParser(description=description)

    # --- Job spec (required only in batch mode; in warm mode stdin jobs take priority) ---
    p.add_argument("--serve", action="store_true",
                   help="Warm mode. Stay resident and receive one JSON job per line from stdin")
    p.add_argument("--camera", default=None, help="Camera object name")
    p.add_argument("--start", type=int, default=None, help="Start frame")
    p.add_argument("--end", type=int, default=None, help="End frame (inclusive)")
    p.add_argument("--out", default=None, help="Output directory")
    p.add_argument("--prefix", default=default_prefix,
                   help=f"Output filename prefix (default: {default_prefix})")
    p.add_argument("--resolution", default=None, help="Resolution WxH (e.g. 1920x1080)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files (default: skip)")

    # --- Device ---
    p.add_argument("--backend", default="CUDA", choices=["CUDA", "OPTIX"],
                   help="Cycles GPU backend (default: CUDA)")
    p.add_argument("--allow-cpu-fallback", action="store_true",
                   help="Continue on CPU even if no GPU device can be enabled "
                        "(default: stop with an error. A silent fallback to CPU "
                        "turns 15 s/frame into 4 min/frame, so it is not allowed by default)")
    p.add_argument("--no-persistent-data", dest="persistent_data", action="store_false",
                   help="Do not keep scene data between frames (default: keep)")

    # --- Optimizations that change the image (same as blender_render/tools/optimize_blend.py) ---
    p.add_argument("--optimize", action="store_true",
                   help="Apply OptiX denoiser, Simplify, indirect clamp, adaptive threshold and "
                        "glass shadow off together (opt-in because it changes the image)")
    p.add_argument("--samples", type=int, default=default_samples,
                   help=f"Override Cycles sample count (default: {default_samples}). "
                        f"A value <= 0 uses the .blend setting as is")
    p.add_argument("--denoiser", default=None,
                   choices=["OPTIX", "OPENIMAGEDENOISE", "NONE"],
                   help="Denoiser (default: keep the .blend setting; OPTIX with --optimize)")
    p.add_argument("--adaptive-threshold", type=float, default=None,
                   help="Adaptive Sampling noise threshold (default with --optimize: 0.03)")
    p.add_argument("--indirect-clamp", type=float, default=None,
                   help="Light Path indirect clamp (default with --optimize: 3.0)")
    p.add_argument("--texture-limit", default=None,
                   help="Simplify viewport texture limit in px (default with --optimize: 2048)")
    p.add_argument("--texture-limit-render", default=None,
                   help="Simplify render texture limit in px (default with --optimize: 4096)")
    p.add_argument("--glass-keyword", action="append", default=[],
                   help="Disable shadow-ray visibility of objects with a material whose name "
                        "contains this keyword (repeatable; default: glass window)")
    return p


def parse_args(parser):
    """Blender arguments after `--` are for the user script."""
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Control messages
# --------------------------------------------------------------------------
def emit(msg):
    """Write one control message line to stdout (distinguishable even when mixed with Blender logs)."""
    print(encode(msg), flush=True)


def _set(obj, attr, value, log):
    """Assign a setting, ignoring missing attributes or type mismatches (for Blender version differences)."""
    if value is None or not hasattr(obj, attr):
        return False
    try:
        setattr(obj, attr, value)
    except (TypeError, ValueError) as e:
        log.append(f"{attr}={value!r} cannot be set ({e})")
        return False
    log.append(f"{attr}={value!r}")
    return True


# --------------------------------------------------------------------------
# Scene setup (run only once when the worker starts)
# --------------------------------------------------------------------------
def setup_gpu(backend, allow_cpu_fallback=False):
    """Enable the Cycles GPU backend and return the names of the enabled devices.

    If compute_device_type in Blender Preferences stays NONE, rendering silently
    falls back to CPU even with device='GPU' in the scene (blender_render/tools/README.md
    §0). Set it explicitly here, and by default raise an error if no GPU is available.
    """
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = backend
    prefs.get_devices()

    # prefs.devices keeps devices that were visible on this machine in the past in userpref
    # (measured: 5 CUDA + 5 OPTIX entries even when only one GPU is visible).
    # Get the devices that really exist with get_devices_for_type().
    try:
        present = list(prefs.get_devices_for_type(backend))
    except AttributeError:      # fallback for old Blender
        present = list(prefs.devices)

    for device in prefs.devices:
        # Enable only GPU devices. Disable CPU devices and leave the CPU for scene sync
        # (reduces CPU contention when several workers share one GPU)
        device.use = (device.type == backend)

    enabled = [d.name for d in present if d.type == backend]

    # has_active_device() is Cycles' final verdict (False means it really runs on CPU)
    try:
        gpu_active = bool(prefs.has_active_device())
    except AttributeError:
        gpu_active = bool(enabled)

    if not enabled or not gpu_active:
        msg = (f"No GPU device is enabled for backend={backend} "
               f"(running as is would fall back to CPU and be tens of times slower). "
               f"Present devices: {[(d.name, d.type) for d in present]}, "
               f"has_active_device={gpu_active}")
        if not allow_cpu_fallback:
            raise RuntimeError(msg)
        print("[WARN] " + msg, flush=True)
        scene.cycles.device = 'CPU'
        return []

    scene.cycles.device = 'GPU'
    return enabled


def apply_warm_settings(scene, persistent_data=True):
    """Speed-up settings that do not change the image. Always applied in both warm and batch mode."""
    log = []
    # Keep scene data (BVH etc.) between frames. In warm mode, where the process is kept,
    # this also works across chunks, so the effect is larger than in cold mode.
    _set(scene.render, "use_persistent_data", persistent_data, log)
    # Denoise and composite on the GPU to avoid being CPU-bound
    _set(scene.cycles, "denoising_use_gpu", True, log)
    _set(scene.render, "compositor_device", "GPU", log)
    return log


def apply_optimizations(scene, args):
    """Image-changing optimizations, same as blender_render/tools/optimize_blend.py.

    --optimize applies preset values; individual options take priority when given.
    Like optimize_blend.py, camera culling and decimation are not touched (verified
    to break the image depending on the scene).
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
            # On GPUs without OptiX support, keep Blender's default (OIDN)
            log.append(f"denoiser={denoiser} unavailable; keeping default")

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
        log.append(f"Shadow-ray visibility off for glass materials: {n} objects")
    return log


def disable_glass_shadows(keywords):
    """Disable shadow-ray visibility of objects with a material whose name matches a keyword."""
    # Require only a trailing word boundary ("Flexiglass" matches, "Windows Button" does not)
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
    """Get a value with priority: job value > CLI value > default."""
    if job.get(key) is not None:
        return job[key]
    v = getattr(args, key, None)
    return default if v is None else v


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_color_job(job, args):
    """Render one job (= one camera, contiguous frame range)."""
    scene = bpy.context.scene
    set_camera(scene, job_value(job, args, "camera"))

    resolution = job_value(job, args, "resolution")
    if resolution:
        set_resolution(scene, resolution)
    samples = job_value(job, args, "samples")
    if samples:
        scene.cycles.samples = int(samples)

    scene.render.image_settings.file_format = 'PNG'

    out_dir = bpy.path.abspath(job_value(job, args, "out"))  # also handles //relative paths
    os.makedirs(out_dir, exist_ok=True)
    prefix = job_value(job, args, "prefix", "frame_")
    start, end = int(job["start"]), int(job["end"])
    overwrite = bool(job_value(job, args, "overwrite", False))

    # Also match the timeline range
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
# Warm loop
# --------------------------------------------------------------------------
def serve(args, render_job, ready_info=None):
    """Receive one JSON job per line from stdin and keep processing until EOF or quit."""
    emit(dict(ready_info or {}, event="ready", pid=os.getpid()))
    while True:
        line = sys.stdin.readline()
        if not line:  # the dispatcher closed stdin
            break
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"status": "error", "message": f"Invalid job line: {e}"})
            continue
        if job.get("cmd") == "quit":
            break
        t0 = time.time()
        try:
            stats = render_job(job, args)
            emit(dict(stats, id=job.get("id"), status="ok",
                      sec=round(time.time() - t0, 2)))
        except Exception as e:  # do not kill the worker because one job failed
            emit({"id": job.get("id"), "status": "error", "message": str(e),
                  "traceback": traceback.format_exc(),
                  "sec": round(time.time() - t0, 2)})


def run(args, render_job, extra_setup=None):
    """Entry point shared by batch/warm. Also used by the depth version."""
    scene = bpy.context.scene
    if args.samples is not None and args.samples <= 0:
        args.samples = None  # use the .blend setting as is

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
        raise SystemExit(f"batch mode requires {', '.join(missing)} "
                         f"(use --serve to run resident)")
    stats = render_job({"start": args.start, "end": args.end}, args)
    print(f"done: rendered={stats['rendered']} skipped={stats['skipped']}", flush=True)


def main():
    parser = build_arg_parser(
        "Render a frame range with a specific camera (batch or warm/--serve)")
    run(parse_args(parser), render_color_job)


if __name__ == "__main__":
    main()
