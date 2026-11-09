"""
Compute PSNR/SSIM between rendered images and real images

PSNR: PSNR is computed from the MSE over the whole RGB image (all pixels x all channels)
SSIM: SSIM is computed separately for each R, G, B channel and averaged
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Path setup (consistency/ → 06_validation/ → scripts/ → project root)
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import marmo3d.marmo_dataset as marmo_dataset
from marmo3d.validation.utils import (
    load_valid_config,
    expand_path_template,
    discover_camera_ids,
    get_project_root,
)

# Project root
project_root = get_project_root()


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load an image (RGB)"""
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute PSNR
    MSE is computed over the whole RGB image (all pixels x all channels)
    """
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM
    SSIM is computed separately for each R, G, B channel and averaged
    """
    return ssim(img1, img2, channel_axis=2, data_range=255)


def find_image_pairs(
    validation_config: Dict[str, Any],
    project_root: Path,
    session_id: str,
    render_hash: str,
    camera_ids: List[str],
    frames_root_override: Optional[Path] = None,
    render_root_override: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Find pairs of rendered and real images

    Returns:
        [{"session_id": str, "camera_id": str, "frame": int,
          "render_path": Path, "valid_path": Path}, ...]
    """
    pairs = []

    # Use render_root_override if given (overrides consistency.yaml)
    if render_root_override is not None:
        render_root = render_root_override
    else:
        render_root_template = validation_config.get("render", {}).get("root", "{PROJECT_ROOT}/outputs/validation/render")
        render_root = expand_path_template(render_root_template, project_root)

    # Use frames_root_override if given
    if frames_root_override is not None:
        valid_root = frames_root_override
    else:
        valid_root_template = validation_config.get("frame_extraction", {}).get("output_root", "{PROJECT_ROOT}/outputs/validation/frames")
        valid_root = expand_path_template(valid_root_template, project_root)

    # Get path templates
    render_path_template = validation_config.get("render", {}).get(
        "path_template", "{render_root}/{session_id}/{hash}/image/Camera.{camera_id}/{frame_number:07d}.png"
    )
    valid_path_template = validation_config.get("frame_extraction", {}).get(
        "output_path_template", "{output_root}/{session_id}/Camera.{camera_id}/{frame_number:07d}.png"
    )

    for camera_id in camera_ids:
        # Build directory paths from templates
        # Strip the filename part ({frame_number:07d}.png) to get the directory
        render_dir_template = str(Path(render_path_template).parent)
        render_dir = expand_path_template(
            render_dir_template,
            project_root,
            render_root=str(render_root),
            session_id=session_id,
            hash=render_hash,
            camera_id=camera_id,
        )

        valid_dir_template = str(Path(valid_path_template).parent)
        valid_dir = expand_path_template(
            valid_dir_template,
            project_root,
            output_root=str(valid_root),
            session_id=session_id,
            camera_id=camera_id,
        )

        if not render_dir.exists():
            continue
        if not valid_dir.exists():
            continue

        # Use the valid-side files as reference
        for valid_file in valid_dir.glob("*.png"):
            frame_name = valid_file.name
            render_file = render_dir / frame_name

            if render_file.exists():
                # Get frame number (without extension)
                frame_number = int(valid_file.stem)
                pairs.append({
                    "session_id": session_id,
                    "camera_id": camera_id,
                    "frame": frame_number,
                    "render_path": render_file,
                    "valid_path": valid_file,
                })

    return pairs


def calculate_metrics_for_pair(pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute metrics for an image pair"""
    render_img = load_image(pair["render_path"])
    valid_img = load_image(pair["valid_path"])

    if render_img is None or valid_img is None:
        return None

    # Size check
    if render_img.shape != valid_img.shape:
        print(f"  WARN: size mismatch {pair['session_id']} Camera.{pair['camera_id']} frame {pair['frame']}")
        return None

    try:
        psnr_value = calculate_psnr(render_img, valid_img)
        ssim_value = calculate_ssim(render_img, valid_img)

        return {
            "session_id": pair["session_id"],
            "camera_id": pair["camera_id"],
            "frame": pair["frame"],
            "psnr": float(psnr_value),
            "ssim": float(ssim_value),
        }
    except Exception as e:
        print(f"  ERROR: metric computation failed {pair['session_id']} Camera.{pair['camera_id']} frame {pair['frame']}: {e}")
        return None


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics"""
    if not results:
        return {}

    psnr_values = [r["psnr"] for r in results]
    ssim_values = [r["ssim"] for r in results]

    # Per session
    by_session = defaultdict(list)
    for r in results:
        by_session[r["session_id"]].append(r)

    session_stats = {}
    for session_id, session_results in by_session.items():
        session_psnr = [r["psnr"] for r in session_results]
        session_ssim = [r["ssim"] for r in session_results]
        session_stats[session_id] = {
            "count": len(session_results),
            "psnr_mean": float(np.mean(session_psnr)),
            "psnr_std": float(np.std(session_psnr)),
            "ssim_mean": float(np.mean(session_ssim)),
            "ssim_std": float(np.std(session_ssim)),
        }

    # Per camera
    by_camera = defaultdict(list)
    for r in results:
        by_camera[r["camera_id"]].append(r)

    camera_stats = {}
    for camera_id, camera_results in by_camera.items():
        camera_psnr = [r["psnr"] for r in camera_results]
        camera_ssim = [r["ssim"] for r in camera_results]
        camera_stats[camera_id] = {
            "count": len(camera_results),
            "psnr_mean": float(np.mean(camera_psnr)),
            "psnr_std": float(np.std(camera_psnr)),
            "ssim_mean": float(np.mean(camera_ssim)),
            "ssim_std": float(np.std(camera_ssim)),
        }

    return {
        "total": {
            "count": len(results),
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "psnr_min": float(np.min(psnr_values)),
            "psnr_max": float(np.max(psnr_values)),
            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "ssim_min": float(np.min(ssim_values)),
            "ssim_max": float(np.max(ssim_values)),
        },
        "by_session": session_stats,
        "by_camera": camera_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM between rendered and real images"
    )
    parser.add_argument(
        "--render-hash",
        type=str,
        required=True,
        help="Render result hash (e.g. 1228b6de)",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="Path to consistency.yaml (default: config/consistency.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (overrides consistency.yaml)",
    )
    parser.add_argument(
        "--session-ids",
        nargs="+",
        default=None,
        help="Session IDs to compute (all sessions if omitted)",
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        default=None,
        help="Root directory of real image frames (overrides consistency.yaml)",
    )
    parser.add_argument(
        "--render-root",
        type=str,
        default=None,
        help="Root directory of rendered images (overrides consistency.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args()

    # Load consistency.yaml
    if args.valid_config:
        valid_config_path = Path(args.valid_config)
    else:
        valid_config_path = project_root / "config" / "consistency.yaml"

    if not valid_config_path.exists():
        print(f"Error: consistency.yaml not found: {valid_config_path}")
        sys.exit(1)

    valid_config = load_valid_config(valid_config_path)
    validation_config = valid_config.get("validation", {})

    # Get the dataset_spec.yaml path
    dataset_spec_template = validation_config.get("dataset_spec", "{PROJECT_ROOT}/config/dataset_spec.yaml")
    spec_path = expand_path_template(dataset_spec_template, project_root)

    if not spec_path.exists():
        print(f"Error: dataset_spec.yaml not found: {spec_path}")
        sys.exit(1)

    dataset_info = marmo_dataset.MarmoDataset.from_yaml(
        spec_path, project_root=project_root
    )

    # Get camera IDs
    camera_ids = discover_camera_ids(dataset_info)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_template = validation_config.get("metrics", {}).get(
            "output_path", "{PROJECT_ROOT}/valid/metrics.json"
        )
        output_path = expand_path_template(output_template, project_root)

    # Determine frames_root
    if args.frames_root:
        frames_root = Path(args.frames_root)
    else:
        frames_root_template = validation_config.get("frame_extraction", {}).get(
            "output_root", "{PROJECT_ROOT}/outputs/validation/frames"
        )
        frames_root = expand_path_template(frames_root_template, project_root)

    # Get session IDs
    if args.session_ids:
        session_ids = args.session_ids
    else:
        # Get from the frames_root directory
        if not frames_root.exists():
            print(f"Error: frames directory not found: {frames_root}")
            sys.exit(1)
        session_ids = [d.name for d in frames_root.iterdir() if d.is_dir()]

    print(f"Starting PSNR/SSIM computation")
    print(f"  hash: {args.render_hash}")
    print(f"  Frames directory: {frames_root}")
    print(f"  Sessions: {len(session_ids)}")
    print(f"  Cameras: {len(camera_ids)}")
    print(f"  Output: {output_path}")

    # Collect image pairs
    all_pairs = []
    for session_id in session_ids:
        pairs = find_image_pairs(
            validation_config=validation_config,
            project_root=project_root,
            session_id=session_id,
            render_hash=args.render_hash,
            camera_ids=camera_ids,
            frames_root_override=Path(args.frames_root) if args.frames_root else None,
            render_root_override=Path(args.render_root) if args.render_root else None,
        )
        all_pairs.extend(pairs)
        if args.verbose and pairs:
            print(f"  {session_id}: {len(pairs)} pairs")

    print(f"\nTotal image pairs: {len(all_pairs)}")

    if len(all_pairs) == 0:
        print("Error: no image pairs found")
        sys.exit(1)

    # Compute metrics
    results = []
    for i, pair in enumerate(all_pairs, 1):
        if args.verbose and i % 100 == 0:
            print(f"  Progress: {i}/{len(all_pairs)}")

        result = calculate_metrics_for_pair(pair)
        if result:
            results.append(result)

    print(f"Computed: {len(results)}/{len(all_pairs)}")

    # Aggregate
    aggregated = aggregate_metrics(results)

    # Save results
    output_data = {
        "render_hash": args.render_hash,
        "total_pairs": len(all_pairs),
        "successful_calculations": len(results),
        "summary": aggregated,
        "details": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results: {output_path}")

    # Show summary
    if aggregated:
        total = aggregated["total"]
        print(f"\n=== Summary ===")
        print(f"PSNR: {total['psnr_mean']:.2f} ± {total['psnr_std']:.2f} (range: {total['psnr_min']:.2f} - {total['psnr_max']:.2f})")
        print(f"SSIM: {total['ssim_mean']:.4f} ± {total['ssim_std']:.4f} (range: {total['ssim_min']:.4f} - {total['ssim_max']:.4f})")


if __name__ == "__main__":
    main()
