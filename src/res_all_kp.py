import json
import math
import os
import subprocess
from pathlib import Path

# ==== Settings ====
GT_JSON = Path("data/mmpose0.29/data/tkmarmo/annotations/tkmarmoV0p42.json")  # load GT
DATA_ROOT = Path("data/mmpose0.29mod/data/tkmarmo/images")    # prepended when file_name is a relative path
CONF_THRESHOLD = 0.9
NUM_KP = 20  # number of x,y,conf triplets

def is_valid_xy(x, y):
    # Treat "all output" as not (0,0) and not NaN
    if x is None or y is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    if isinstance(y, float) and math.isnan(y):
        return False
    if x == 0 and y == 0:
        return False
    return True

def collect_qualified_image_ids(det_json_path):
    with det_json_path.open("rb") as f:
        gt = json.load(f)

    image_ids = []

    for ann in gt.get("annotations", []):
        kps = ann.get("keypoints", [])
        if not kps or len(kps) % 3 != 0:
            continue

        xs = kps[0::3]
        ys = kps[1::3]
        vs = kps[2::3]
        K  = len(vs)

        # All present: v>0 and x,y are not 0
        all_exist = all((v > 0) and (x != 0) and (y != 0) for x, y, v in zip(xs, ys, vs))
        if all_exist:
            image_ids.append(ann.get("image_id"))

    return image_ids

def build_image_map(gt_json_path):
    with gt_json_path.open("rb") as f:
        gt = json.load(f)

    images = gt.get("images", [])
    # id -> file_name
    return {img["id"]: img["file_name"] for img in images if "id" in img and "file_name" in img}

def main():
    qualified = collect_qualified_image_ids(GT_JSON)
    id_to_fname = build_image_map(GT_JSON)

    # List matching file_names
    selected_paths = []
    print("Matched image_id (avg_conf):")
    for image_id in qualified:
        fname = id_to_fname.get(image_id)
        if not fname:
            continue
        path = (DATA_ROOT / fname) if not os.path.isabs(fname) else Path(fname)
        if path.exists():
            selected_paths.append(str(path))
            print(f"- {image_id} -> {path}")
        else:
            print(f"- {image_id} -> NOT FOUND: {path}")

    if not selected_paths:
        print("No images matching the conditions were found.")
        return

    # Display in Shotwell (opens all at once)
    try:
        subprocess.run(["shotwell", *selected_paths], check=True)
    except FileNotFoundError:
        print("Shotwell not found. Please install it and run again.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run Shotwell: {e}")

if __name__ == "__main__":
    main()
