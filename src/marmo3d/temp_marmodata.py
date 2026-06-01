from pathlib import Path
import marmo_dataset
import os

def calculate_total_frames(ds: marmo_dataset.MarmoDataset) -> int:
    # 1. auto-discover camids if not set (since allowed_ids is empty in yaml)
    camids = ds.spec.camera_allowed_ids
    if not camids:
        print("Debugging: allowed_ids is empty. Auto-discovering from video_root...")
        discovered = set()
        if ds.video_root.exists():
            for p in ds.video_root.iterdir():
                if p.is_dir():
                    m = ds.spec.video_folder_id_re.match(p.name)
                    if m:
                        discovered.add(m.group("camid"))
        camids = sorted(list(discovered))
        print(f"Debugging: Discovered {len(camids)} cameras: {camids}")
    
    if not camids:
        print("Warning: No cameras found.")
        return 0

    # 2. Iterate sessions and count
    total_frames = 0
    nominal = ds.spec.dataset["layout"]["video_chunks"]["chunk_frames_nominal"]
    
    # valid_only=True ensures we only count "good" sessions (though if allowed_ids was empty, integrity check passed trivially)
    # If we really want to check if *these* cameras exist in the session, we check it here.
    sessions = list(ds.iter_sessions(valid_only=True, log_skips=False))
    print(f"Counting frames for {len(sessions)} sessions...")

    for i, s in enumerate(sessions):
        # Optional: Print progress
        # if i % 10 == 0: print(f"Processing session {i}/{len(sessions)}: {s.session_id}")
        
        for camid in camids:
            cam = s.cam(camid)
            if cam.folder.exists():
                chunks = cam.mp4_chunks(ds.spec)
                count = len(chunks)
                # Approximation: count * 10000
                total_frames += count * nominal
            else:
                # If valid_only=True and allowed_ids was empty, we might have sessions missing some cams (since check was skipped)
                # That's fine, we just don't count missing folders.
                pass
                
    return total_frames

if __name__ == "__main__":
    PROJECT_ROOT = Path('.').resolve()
    # Assuming dataset_spec.yaml is at class_blueprint/dataset_spec.yaml
    spec_path = PROJECT_ROOT / 'class_blueprint/dataset_spec.yaml'
    
    if not spec_path.exists():
        # Fallback for when running from class_blueprint dir?
        # Check current dir
        if Path('dataset_spec.yaml').exists():
            spec_path = Path('dataset_spec.yaml')
            PROJECT_ROOT = PROJECT_ROOT.parent # just a guess, or keep as is if running from root

    print(f"Loading spec from: {spec_path}")
    ds = marmo_dataset.MarmoDataset.from_yaml(spec_path, project_root=PROJECT_ROOT)

    print("video_root:", ds.video_root)
    print("results_subdir:", ds.results_subdir)

    total = calculate_total_frames(ds)
    print(f"Total Frames (approx): {total}")