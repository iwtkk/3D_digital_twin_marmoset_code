Blender Rendering Automation Walkthrough
I have implemented the Blender rendering automation pipeline consisting of two main scripts:

render_blender.py
: The internal Blender script that handles scene setup, CSV generation, and rendering.
rendering_parallelized.py
: The external launcher that orchestrates parallel execution and integrates with marmo_dataset.
Key Features
Self-Contained Logic: 
render_blender.py
 contains all necessary logic to convert 
kp3d
 pickles to CSV and apply them to the armature, running entirely within Blender's Python environment.
Parallel Execution: 
rendering_parallelized.py
 can launch multiple Blender instances to process sessions in parallel.
Manifest Generation: A manifest.json is strictly generated for every execution to track status (success/failed).
Path Resolution: Fully integrated with marmo_dataset and 
render_spec.yaml
 to resolve project paths automatically.
Files
class_blueprint/render_blender.py
class_blueprint/rendering_parallelized.py
Usage
Parallel Execution (Recommended)
Run the launcher from the project root:

python3 class_blueprint/rendering_parallelized.py --num_workers 4
Arguments:

--project_root: Path to project root (default: current dir).
--spec_path: Path to 
dataset_spec.yaml
.
--profile: Rendering profile (default: env_camera).
--num_workers: Number of parallel Blender processes.
--session_ids: (Optional) List of specific session IDs to process.
--dry_run: Print commands without executing.
Single Session Debugging
You can also run 
rendering_parallelized.py
 for a single session to debug:

python3 class_blueprint/rendering_parallelized.py --session_ids dailylife_cj611_20230312_100000 --num_workers 1
Verification Results
A dry-run verification was performed, confirming that:

All sessions were successfully discovered.
Input/Output paths were correctly resolved using the dataset spec.
Blender commands were constructed with the correct arguments.
Example generated command (Dry Run):

blender -b /path/to/base.blend --python render_blender.py -- \
    --session_id dailylife_cj611_20230312_100000 \
    --kp3d_pkl /path/to/kp3d.pickle \
    --render_spec /path/to/render_spec.yaml \
    --profile env_camera \
    --output_dir /path/to/output \
    --job_id job_123 \
    --render 1