import json
import pickle
from pathlib import Path
import marmo3d.marmo_dataset as marmo_dataset

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
    