
import json
import pickle
from pathlib import Path
import marmo3d.marmo_dataset as marmo_dataset

def count_frames(sessions: list[marmo_dataset.Session], output_path: Path):
    results = {}
    # sessions = list(ds.iter_sessions(valid_only=True, log_skips=True))
    print(f"Found {len(sessions)} valid sessions.")

    total_frames_all = 0

    for s in sessions:
        kp3d_path = s.kp3d_path()
        if not kp3d_path or not kp3d_path.exists():
            print(f"[{s.session_id}] kp3d path missing despite validation: {kp3d_path}")
            continue
        
        try:
            with open(kp3d_path, "rb") as f:
                data = pickle.load(f)
            
            # data structure verified as dict with 'kp3d' key
            # kp3d shape: (n_animals, n_frames, n_joints, n_coords)
            if "kp3d" in data:
                n_frames = data["kp3d"].shape[1]
                results[s.session_id] = n_frames
                total_frames_all += n_frames
            #     print(f"[{s.session_id}] {n_frames} frames")
            # else:
            #     print(f"[{s.session_id}] 'kp3d' key missing in {kp3d_path}")
                
        except Exception as e:
            print(f"[{s.session_id}] Error reading {kp3d_path}: {e}")
            
    # Output results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Total frames counted: {total_frames_all}")
    print(f"Results saved to {output_path}")

def count_frames_all(sessions: list[marmo_dataset.Session], output_path: Path):
    
    for animal in ['cj611', 'cj425', 'cj357']:
        ds_ = ds.filter(sessions, animal=animal)
        output_json = Path(f"frame_counts_{animal}.json")
        print(f"Counting frames for {animal}...")
        count_frames(ds_, output_json)

if __name__ == "__main__":
    # Assuming this script is run from the project root or class_blueprint dir
    # Adjusted to work if run from project root and file is in class_blueprint/
    
    current_dir = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    
    # Heuristic to find project root and dataset_spec
    # If we are in project root (ez2use), class_blueprint is a subdir
    if (current_dir / "class_blueprint").exists():
        project_root = current_dir
        spec_path = project_root / "class_blueprint" / "dataset_spec.yaml"
    # If we are inside class_blueprint
    elif (current_dir / "dataset_spec.yaml").exists():
        project_root = current_dir.parent
        spec_path = current_dir / "dataset_spec.yaml"
    else:
        # Fallback to script location
        project_root = script_dir.parent
        spec_path = script_dir / "dataset_spec.yaml"

    print(f"Project Root: {project_root}")
    print(f"Spec Path: {spec_path}")
    
    ds = marmo_dataset.MarmoDataset.from_yaml(spec_path, project_root=project_root)
    sessions = list(ds.iter_sessions(valid_only=True, log_skips=True))

    count_frames_all(sessions, current_dir)
