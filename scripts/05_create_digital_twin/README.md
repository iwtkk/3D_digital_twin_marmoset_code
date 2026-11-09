# 05 – Create Digital Twin

Apply 3D keypoints to the rigged marmoset model in Blender and render
pseudo-first-person-view videos.

## Requirements

- Blender 4.5.0
- Blender scene file: `3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend`
- NVIDIA GPU (CUDA-capable) — rendering runs on the CYCLES engine + CUDA (tested environment: NVIDIA A100-SXM4-40GB × 7, CUDA 12.7)

## Input

- `kp3d_fxdJointLen.pickle` from step 03 (`outputs/3d/{session}/kp3d_fxdJointLen.pickle`)

## Step 1 – Convert pickle to CSV

```bash
# All sessions at once
python scripts/05_create_digital_twin/kp3d_to_blender_csv.py --all

# A single session only
python scripts/05_create_digital_twin/kp3d_to_blender_csv.py dailylife_cj611_20230227_100000
```

## Step 2 – Apply keypoints to Blender rig

Open and run `blender_animation.py` in Blender's Scripting workspace, or run it from the CLI:

```bash
blender 3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend \
  --background \
  --python scripts/05_create_digital_twin/blender_animation.py \
  -- \
  outputs/animate_temp/transformedcsv/{session}.csv
```

Options after `--`:

| Argument | Default | Description |
|---|---|---|
| `<csv_path>` | (required) | Path to the CSV file generated in Step 1 |
| `--armature <name>` | `NF_Armature` | Name of the armature object in the Blender scene |

To use a different armature:

```bash
blender scene.blend \
  --background \
  --python scripts/05_create_digital_twin/blender_animation.py \
  -- \
  outputs/animate_temp/transformedcsv/{session}.csv \
  --armature MyArmature
```

## Step 3 – Render pseudo-view

Run `allcamrender.py` in Blender's Scripting workspace, or use the parallel rendering script:

```bash
python scripts/05_create_digital_twin/render_parallel.py \
  --blend 3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend \
  --camera Camera_R \
  --entire_start 1 \
  --entire_end 3000 \
  --out outputs/render/{session}/Camera_R \
  --chunk_size 100 \
  --max_jobs 4
```

### Warm mode (default)

The rendering scripts default to **warm mode, which keeps blender resident**.
One blender is started per GPU slot, and chunks are fed in via stdin
(protocol in `render_ipc.py`, dispatcher in `render_worker.py`).

- Loading the `.blend` / initializing the CUDA/OptiX context / building the BVH / compiling shaders happens
  **once per worker**. Previously this was paid for every chunk.
- `use_persistent_data` only takes effect within a process, so in warm mode it takes effect across chunks.
- Switching cameras and changing the frame range are done within the process. In `render_sessions.py`,
  workers prefer jobs for the `.blend` they have already loaded, and restart blender **only when moving
  to another session**.
- Measured (A100, 320x320, samples=16): 1st image 70s → 2nd image 20s → then 12-15s/image.
  In cold mode this initial cost was paid again for every chunk.
- If problems occur, `--cold` reverts to the previous "1 chunk = 1 process" mode.

Settings based on the test results in `blender_render/tools/` can also be passed:

| Option | Description |
|---|---|
| `--optimize` | Applies OptiX denoiser, Simplify + texture limit, indirect clamp, adaptive threshold and disabling glass shadows together (opt-in because **the image changes**) |
| `--samples N` | Overrides the Cycles sample count. **Default for color is 32** (1 for depth). `0` or less uses the .blend settings as is. Validate the value with `blender_render/tools/run_sample_sweep.sh` |
| `--denoiser {OPTIX,OPENIMAGEDENOISE,NONE}` | Denoiser selection |
| `--allow-cpu-fallback` | Continue on CPU even if the GPU cannot be enabled (default is to stop with an error) |
| `--threads N` | CPU threads per blender instance (`blender -t`). **Default is automatic (cores ÷ slots × 2, max 16)**. `0` means unlimited. As the measurement table below shows, this is the parameter with the largest effect on throughput |
| `--restart-every N` | Restart blender after the worker processes N jobs (a safeguard for long runs) |
| `--job-timeout SEC` | If a chunk hangs, recreate the worker and retry |
| `--progress` | Show the time taken per frame |
| `--blender-arg X` | Pass arbitrary arguments through to `render_range*.py` |

If the GPU cannot be enabled, Cycles **silently falls back to the CPU and becomes tens of times slower**, so
by default it stops with an error (`blender_render/tools/README.md` §0).

### How to check that the GPU is actually used

Cycles easily ends up in a state where "you think you selected the GPU but it is actually the CPU" (this happens on desktops).
This pipeline has three levels of detection:

1. **At startup**: Get the devices that *actually exist* with `prefs.get_devices_for_type(backend)`,
   and abort startup if `prefs.has_active_device()` is False (`--allow-cpu-fallback` to continue).
   `prefs.devices` is unreliable because devices that were once visible on that machine remain in the userpref
   (measured: 13 entries listed when only one GPU was visible). Only the existing devices are logged.
2. **After the first render**: The dispatcher checks whether the worker PID (and its child processes)
   appear in `nvidia-smi --query-compute-apps`,
   and prints `GPU in use (VRAM 4272MiB)` or a warning. The CUDA context is created when rendering
   starts, so this cannot be determined right after startup.
3. **Manual definitive diagnosis**: `check_gpu_device.py` varies the sample count and looks at the increase in time taken.
   If it runs on the GPU, increasing the samples barely increases the time.

   ```bash
   blender outputs/blender/work_{session}.blend -b \
     -P scripts/05_create_digital_twin/check_gpu_device.py -- \
     --camera Camera_C --res 1000x1000 --frame 3000 --samples 32,256
   ```

   Measured (1000×1000, idle GPU, denoising OFF):

   | device | samples=32 | samples=256 | for the additional 224 samples |
   |---|---|---|---|
   | GPU | 27.4s | 22.1s | almost 0 (the first-image initialization cost is larger) |
   | CPU (256 cores) | 15.4s | 79.9s | +64.5s (288 ms/sample) |

   This difference (GPU is almost flat with respect to samples) is evidence that the GPU is running. During rendering,
   GPU utilization in `nvidia-smi` also reaches 99-100%.

**Note**: At low sample counts such as samples=32, a 256-core EPYC is faster per image than an A100
(15.4s vs 22.1s). Even if the GPU appears idle this is not abnormal; the time per image is
simply determined by CPU-side processing (scene sync, transfer, film, PNG writing) rather than
GPU path tracing. At the production samples=500, the GPU is overwhelmingly faster.

### Performance characteristics (measured / A100, 1000×1000, production blend `work_*.blend`)

| Condition | wall per image | CPU per image | Startup cost (CPU) |
|---|---|---|---|
| samples=500 (blend default) | 42-45s | – | – |
| samples=128 | 13-19s | – | – |
| samples=32 (current default) | 10-12s | 153 core-s | 453 core-s |
| samples=32 + `--pin-cpu` (6 cores) | 43s | 70 core-s | 30 core-s |

By `--threads` (blender `-t`) (500×500, samples=32, 1 worker, per image):

| Setting | wall/image | CPU/image | Mean cores occupied | Startup CPU | Workers that fit in 256 cores | Total throughput |
|---|---|---|---|---|---|---|
| 0 (unlimited) | 15.9s | 130.8 core-s | 8.2 | 163.8s | 31 | 7,100 images/h |
| 64 | 14.6s | 145.7 core-s | 9.9 | 48.2s | 26 | 6,400 images/h |
| **16** | 17.4s | 78.6 core-s | 4.5 | 18.8s | 57 | **11,800 images/h** |
| 4 | 28.9s | 69.4 core-s | 2.4 | 16.4s | 107 | 13,300 images/h |
| 4 + `--pin-cpu` | 24.3s | 67.2 core-s | 2.8 | 15.7s | 91 | 13,500 images/h |
| 16 + `--pin-cpu` | 19.5s | 94.1 core-s | 4.8 | 20.6s | 53 | 9,800 images/h |

- Limiting `-t` **almost halves the CPU per image** (131→70 core-s). The startup cost also drops 164→16 core-s.
  More workers can run accordingly, so the node-wide throughput becomes 1.7-1.9×.
- Too small makes the wall time per image longer (4 is 1.7× of 16). **Around 16 is the best balance**.
- 64 is about the same as unlimited and just wastes CPU.
- The additional effect of `--pin-cpu` is small when combined with `-t` (wall −16%, CPU −3%). It is `-t` that matters.
- The default is "cores ÷ slots × 2, max 16" (because the actually occupied cores are about 1/3 of `-t`).

By resolution (samples=32, 1000×1000 → 500×500 → 320×320):

| Resolution | CPU/image (unlimited) | CPU/image (6-core pin) | wall/image (6-core pin) | PNG per image |
|---|---|---|---|---|
| 1000×1000 | 153 core-s | 70 core-s | 43.0s | approx. 1.2MB |
| 500×500 | 144 core-s | 73 core-s | **25.4s** | approx. 238KB |
| 320×320 | 132 core-s | – | – | approx. 157KB |

How to read this:

- **Going from samples 500→32 gives at most about a 4× speedup**. Each image has a fixed cost independent
  of samples (scene evaluation, output).
- **The CPU is the bottleneck**. With 8 workers/GPU, GPU utilization is only 12%, and one blender
  grabs about 17 cores (775 threads). Even reducing the resolution to 1/9.8 (320×320), CPU only
  drops 153→132 core-s = the main cause is not pixels but **scene evaluation (depsgraph sync of the rig + 430k
  polygons)**.
- Limiting the CPU halves the CPU per image (153→70 core-s, of which sys time drops 302→15 s),
  and wasteful spin-waiting disappears. The wall time of a single worker increases, but total throughput when
  many workers run side by side goes up. **What matters is `--threads` (blender -t); `--pin-cpu`
  adds little even when combined** (see the table above).
- **Lowering the resolution has almost no effect with unlimited threads** (1000²→500² reduces CPU by 6%).
  On the other hand, with `--pin-cpu` it goes from 43s→25.4s per image (1.7×). When cores are limited, CPU-side
  pixel processing (film conversion, PNG compression, denoising) becomes the bottleneck.
  The output size goes from 1.2MB→238KB (1/5), which also helps downstream I/O.
- Before adding workers, determine `--pin-cpu` and `--workers_per_gpu` by measurement.
  The CPU saturates before VRAM (approx. 1.7GB per instance).

Or use the shell scripts:

```bash
# Run Step 2 + Step 3 sequentially (default: NF_Armature)
bash scripts/05_create_digital_twin/run_all.sh

# Run only Step 2 in parallel (already processed .blend files are skipped)
bash scripts/05_create_digital_twin/run_animate_parallel.sh

# To specify the armature name
bash scripts/05_create_digital_twin/run_all.sh --armature MyArmature
bash scripts/05_create_digital_twin/run_animate_parallel.sh --armature MyArmature

# To also specify the blend file
bash scripts/05_create_digital_twin/run_all.sh --blend custom.blend --armature MyArmature
bash scripts/05_create_digital_twin/run_animate_parallel.sh --blend custom.blend --armature MyArmature
```

## Scripts

| Script | Purpose |
|---|---|
| `blender_animation.py` | Apply 3D keypoints to IK rig |
| `allcamrender.py` | Render from all camera positions (Blender script) |
| `render_sessions.py` | Puts all sessions × all cameras into a single queue for GPU-parallel rendering (the implementation behind `run_render.sh`) |
| `render_parallel.py` | Splits the frame range of 1 blend / 1 camera and renders in parallel |
| `render_range.py` | Blender-side worker (PNG). `--serve` for warm resident mode; without it, the previous single run |
| `render_parallel_depth.py` | Depth (Z) version of `render_parallel.py` |
| `render_range_depth.py` | Blender-side worker (Depth EXR). Supports `--serve` |
| `render_worker.py` | Process management for warm workers and per-blend job scheduler (dispatcher side) |
| `render_ipc.py` | Line-JSON protocol definition for warm workers |
| `check_gpu_device.py` | Diagnostic that determines whether Cycles is actually rendering on the GPU (Blender script) |
| `kp3d_to_blender_csv.py` | Convert kp3d pickle to Blender-ready CSV |
| `run_all.sh` | Runs Step 2 + Step 3 sequentially (one CSV at a time) |
| `run_animate_parallel.sh` | Runs only Step 2 in parallel (up to `MAX_JOBS` at once, skips already processed) |
| `run_render.sh` | Runs only Step 3 |

## See also

- [docs/blender_usage.md](../../docs/blender_usage.md)
- [docs/bone_mapping.md](../../docs/bone_mapping.md)
