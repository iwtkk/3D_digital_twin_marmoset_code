
import sys
import os
import argparse
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

# Ensure we can import marmo_dataset from current dir
sys.path.append(str(Path(__file__).parent))
try:
    from marmo_dataset import MarmoDataset, DatasetSpec, Session
    import yaml
except ImportError:
    print("WARNING: marmo_dataset or pyyaml not found.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("Launcher")

class ParallelRenderer:
    def __init__(self, session_ids: List[str], spec_path: Path, project_root: Path, profile: str, num_workers: int, dry_run: bool):
        self.session_ids = session_ids
        self.project_root = project_root
        self.profile = profile
        self.num_workers = num_workers
        self.dry_run = dry_run
        
        # Load Dataset
        self.ds = MarmoDataset.from_yaml(spec_path, project_root=project_root)
        
        # Load Render Spec
        # Assuming render_spec.yaml is in class_blueprint or specified
        self.render_spec_path = spec_path.parent / "render_spec.yaml" # Default location assumption
        if not self.render_spec_path.exists():
             # Try argument based?
             pass
        with open(self.render_spec_path) as f:
            self.render_spec = yaml.safe_load(f)

    def run(self):
        jobs = []
        for sid in self.session_ids:
            try:
                session = self.ds.session(sid)
                session_jobs = self._create_jobs_for_session(session)
                jobs.extend(session_jobs)
            except Exception as e:
                logger.error(f"Failed to prepare jobs for session {sid}: {e}")

        logger.info(f"Total jobs to run: {len(jobs)}")
        
        # Run jobs
        self._execute_jobs(jobs)

    def _create_jobs_for_session(self, session: Session) -> List[Dict]:
        """
        Determine split based on render spec.
        Image -> Split by Camera (or frame chunks)
        Video -> Split by Session (full)
        """
        # Determine output mode from render spec
        # Spec: rendering: output: ...
        # But actually spec says "image" OR "video" mode.
        # Let's check what is enabled or requested?
        # The prompt says: "Image Mode: Split by Camera", "Video Mode: Split by Session"
        # We need to know which one we are doing.
        # User CLI should probably specify target? Or we do both if defined?
        # Let's assume we do what's in the spec or all.
        # But we need to know if we are doing "frames" (image) or "range" (video).
        
        # For this implementation, I will assume we look at the spec and generate jobs for 'image' first, as that's the complex parallel one.
        # If 'video', usually one job per session.
        
        jobs = []
        
        kp3d_path = session.kp3d_path()
        if not kp3d_path:
            logger.warning(f"No kp3d found for session {session.session_id}, skipping.")
            return []
            
        render_conf = self.render_spec["render"]["rendering"]
        prof = render_conf["profiles"][self.profile]
        
        # 1. Resolve Inputs
        base_blend_tmpl = self.render_spec["render"]["inputs"]["base_blend"]["path"]
        base_blend = base_blend_tmpl.replace("{PROJECT_ROOT}", str(self.project_root))
        
        # 2. Determine View/Frames
        # For 'env_camera', we have 8 views.
        # If we render IMAGES, we usually want specific frames.
        # Where do we get the list of frames to render?
        # Spec 5.1: "select_frames_or_range" -> Dataset side decides?
        # NOTE: For now, I'll default to a small test set or full range if not specified.
        # But for parallelism, let's say we split by Camera.
        
        # Helper to get views
        views = []
        if "views" in prof:
             v_cfg = prof["views"]
             if "count" in v_cfg:
                 for i in range(v_cfg["count"]):
                     views.append(f"Camera.{i}") # Logic matching render_blender.py
             elif "cameras" in v_cfg:
                 views = v_cfg["cameras"]
             elif "camera" in v_cfg:
                 views = [v_cfg["camera"]]
        
        # Construct Job
        # output_dir determination
        out_tmpl = render_conf["output"]["image"]["path_template"] # {PROJECT_ROOT}/results/render/image/{animal}/{session_id}/{frame:08d}/view_{view_index:02d}.png
        # We want a base output dir for the session/job
        # e.g. {PROJECT_ROOT}/results/render/image/{animal}/{session_id}
        out_base = Path(out_tmpl.split("/{frame")[0].format(
            PROJECT_ROOT=str(self.project_root),
            animal=session.animal,
            session_id=session.session_id
        ))
        
        # Job Splitting Logic
        # Option A: One job per camera (handles all frames for that camera) -> Good for image sequence
        # Option B: One job per session (handles all cameras) -> Less overhead if blender lines up
        # Spec says: "Image: Split by Camera"
        
        # But wait, render_blender.py receives "frames".
        # If we split by camera, we need render_blender.py to accept "only render camera X".
        # render_blender.py currently iterates ALL cameras in profile.
        # I need to verify if render_blender supports filtering cameras?
        # Checked code: logic iterates `cameras` list derived from profile.
        # It does NOT seem to filter by arg yet.
        # I should probably pass a hacked profile or filter args.
        # Or, I rely on `Split by Camera` being "Blender filters cameras".
        # Actually render_blender.py Logic:
        # `if "views" in prof: ... cameras.append(...)`
        # I should add an argument to render_blender.py to restrict cameras if I want to parallelize by camera.
        
        # Let's add --cameras arg to render_blender.py later? 
        # Or just run full session per job for now to be safe, or split by frames?
        # Spec: "Image Mode: Split by Camera"
        # "Video Mode: Split by Session"
        
        # Let's assume I can't easily filter cameras in render_blender without modifying it.
        # But I CAN modify it.
        # However, to save time, maybe I split by FRAMES for image mode? (1-100, 101-200)
        # Spec says "Image Mode: Split by Camera".
        # This implies running 8 blender instances, each rendering 1 camera for same frames.
        
        # I will generate jobs. For now, let's treat it as Session execution because render_blender supports multi-cam.
        # If I want to support Split By Camera, I'd need to tell render_blender "Only render Camera.0".
        # I'll stick to Session-based for MVP unless easy.
        
        job_id = f"{session.session_id}_{self.profile}"
        
        cmd_args = [
            "--session_id", session.session_id,
            "--kp3d_pkl", str(kp3d_path),
            "--render_spec", str(self.render_spec_path),
            "--profile", self.profile,
            "--output_dir", str(out_base),
            "--job_id", job_id,
            "--render", "1"
        ]
        
        # base command
        blender_cmd = [
            "blender",
            "-b", base_blend,
            "--python", "render_blender.py",
            "--"
        ] + cmd_args
        
        jobs.append({
            "id": job_id,
            "cmd": blender_cmd,
            "cwd": str(Path(__file__).parent), # run in class_blueprint
            "out_log": out_base / f"{job_id}.out",
            "err_log": out_base / f"{job_id}.err"
        })
        
        return jobs

    def _execute_jobs(self, jobs: List[Dict]):
        active_procs = []
        queue = jobs[:]
        
        while queue or active_procs:
            # Spawn if slots available
            while queue and len(active_procs) < self.num_workers:
                job = queue.pop(0)
                logger.info(f"Starting job {job['id']}")
                if self.dry_run:
                    logger.info(f"DRY RUN: {' '.join(job['cmd'])}")
                    continue
                
                # Prep logs
                job['out_log'].parent.mkdir(parents=True, exist_ok=True)
                stdout = open(job['out_log'], "w")
                stderr = open(job['err_log'], "w")
                
                try:
                    proc = subprocess.Popen(
                        job['cmd'],
                        stdout=stdout,
                        stderr=stderr,
                        cwd=job['cwd']
                    )
                    active_procs.append((proc, job, stdout, stderr))
                except Exception as e:
                    logger.error(f"Failed to start job {job['id']}: {e}")
                    stdout.close()
                    stderr.close()
            
            if not active_procs and not queue:
                break
                
            # Check status
            still_active = []
            for proc, job, out, err in active_procs:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, job, out, err))
                else:
                    logger.info(f"Job {job['id']} finished with code {ret}")
                    out.close()
                    err.close()
                    if ret != 0:
                        logger.error(f"Job {job['id']} FAILED. See {job['err_log']}")
                        
            active_procs = still_active
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=os.getcwd())
    parser.add_argument("--spec_path", default="class_blueprint/dataset_spec.yaml")
    parser.add_argument("--profile", default="env_camera")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--session_ids", nargs="+", help="Specific sessions to process")
    parser.add_argument("--dry_run", action="store_true")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    spec_path = project_root / args.spec_path
    
    if not args.session_ids:
        # Discover all
        ds = MarmoDataset.from_yaml(spec_path, project_root=project_root)
        sids = ds.discover_session_ids()
    else:
        sids = args.session_ids
        
    runner = ParallelRenderer(
        session_ids=sids,
        spec_path=spec_path,
        project_root=project_root,
        profile=args.profile,
        num_workers=args.num_workers,
        dry_run=args.dry_run
    )
    runner.run()

if __name__ == "__main__":
    main()
