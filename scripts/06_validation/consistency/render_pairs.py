"""
Render the Fig. 8a pairs (frames.csv) from the Published dataset, one Blender process per Session x model.

I/O ledger: scripts/06_validation/consistency/io.md#render_pairs

Each Session x model is one `blender -b -P render_pairs_blender.py` process; processes are spread over the GPUs
given by --gpu. Renders that already exist are skipped, so an interrupted run resumes with the same command.
Use Blender 5.2.x: other versions render slightly different pixels (README.md, "Tolerance").

Inputs (read only, --dataset-root = extracted Published dataset):
  3D_digital_twin_marmoset_data/rendered_environmental_images/frames.csv
  3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_files/{session}.blend
  3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_simplified_marmoset/{session}.blend   (simplified_fur)
Outputs (--work-dir):
  render/{model}/{session}/Camera.{camera_id}/{frame_3d:07d}.png   [1536, 2048, 3] uint8
  render/background/Camera.{camera_id}.png   empty cage (animals excluded), one per camera, [1536, 2048, 3] uint8
  render_jobs/{model}/{session}.json, render_logs/{model}/{session}.log

Example:
  .venv/bin/python scripts/06_validation/consistency/render_pairs.py \\
      --dataset-root /path/to/3D_digital_twin_marmoset_data --work-dir outputs/validation/consistency_fig8a \\
      --blender /snap/blender/7803/blender --gpu 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import consistency_core as cc

BLENDER_SCRIPT = Path(__file__).resolve().parent / "render_pairs_blender.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--frames-csv", type=Path, default=None, help="default: {dataset-root}/.../frames.csv")
    p.add_argument("--models", nargs="+", default=list(cc.MODELS), choices=cc.MODELS)
    p.add_argument("--session", nargs="*", default=None, help="only these Sessions")
    p.add_argument("--gpu", default="0", help="comma-separated GPU indices (one process per GPU)")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--blender", default="blender")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    pairs = cc.load_pairs(a.dataset_root, a.frames_csv, a.session)
    tasks = []
    for s, g in pairs.groupby("session_id"):
        for m in a.models:
            jobs = [{"camera": f"Camera.{r.camera_id}", "blend_frame": int(r.blend_frame),
                     "out": str(cc.render_path(a.work_dir, m, s, r.camera_id, int(r.frame_3d)))}
                    for r in g.itertuples()]
            n = sum(not os.path.exists(j["out"]) for j in jobs)
            if not n:
                continue
            jf = a.work_dir / "render_jobs" / m / f"{s}.json"
            jf.parent.mkdir(parents=True, exist_ok=True)
            jf.write_text(json.dumps(jobs, indent=1))
            cmd = [a.blender, "-b", "-P", str(BLENDER_SCRIPT), "--", "--model", m, "--jobs", str(jf),
                   "--samples", str(a.samples), "--blend", str(a.dataset_root / cc.DETAILED_BLEND.format(session=s))]
            if m == "simplified_fur":
                cmd += ["--simplified-blend", str(a.dataset_root / cc.SIMPLIFIED_BLEND.format(session=s))]
            tasks.append((m, s, n, cmd, a.work_dir / "render_logs" / m / f"{s}.log"))
    # empty-cage background per camera, rendered from the first Session (the environment is the same in every Session)
    first = pairs.sort_values(["session_id", "frame_3d"]).groupby("camera_id").head(1)
    s0 = sorted(pairs.session_id)[0]
    jobs = [{"camera": f"Camera.{c}", "blend_frame": int(pairs[pairs.session_id == s0].blend_frame.iloc[0]),
             "out": str(cc.background_path(a.work_dir, c))} for c in sorted(first.camera_id.unique())]
    n = sum(not os.path.exists(j["out"]) for j in jobs)
    if n:
        jf = a.work_dir / "render_jobs" / "background.json"
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps(jobs, indent=1))
        tasks.insert(0, ("background", s0, n, [a.blender, "-b", "-P", str(BLENDER_SCRIPT), "--", "--model", "background",
                                              "--jobs", str(jf), "--samples", str(a.samples),
                                              "--blend", str(a.dataset_root / cc.DETAILED_BLEND.format(session=s0))],
                         a.work_dir / "render_logs" / "background.log"))
    print(f"{len(tasks)} tasks, {sum(t[2] for t in tasks)} images")
    if a.dry_run:
        for t in tasks[:3]:
            print(" ".join(t[3]))
        return

    slots = a.gpu.split(",")
    running, failed, done, t0 = {}, [], 0, time.time()
    queue = list(tasks)
    while queue or running:
        for i, g in enumerate(slots):
            if i not in running and queue:
                m, s, n, cmd, log = queue.pop(0)
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
                running[i] = (subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env),
                              (m, s, n, log))
        time.sleep(5)
        for i in list(running):
            pr, (m, s, n, log) = running[i]
            if pr.poll() is None:
                continue
            del running[i]; done += 1
            if pr.returncode != 0:
                failed.append((m, s, str(log)))
            print(f"[{done}/{len(tasks)}] {m} {s} ({n} img) rc={pr.returncode} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"failed: {len(failed)}")
    for f in failed:
        print("  ", *f)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
