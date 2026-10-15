"""
Outlier detection and visualization

Detect outliers with a statistical method (IQR), write them to CSV, and visualize the images
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Path setup (consistency/ → 06_validation/ → scripts/ → project root)
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from marmo3d.validation.utils import (
    load_valid_config,
    expand_path_template,
    get_project_root,
)

# Project root
project_root = get_project_root()


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load the metrics JSON"""
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_outliers_iqr(
    data: List[float],
    multiplier: float = 1.5,
) -> Tuple[float, float, List[int]]:
    """
    Detect outliers with the IQR method

    Args:
        data: Data array
        multiplier: IQR multiplier (1.5 is typical; 3.0 for extreme outliers only)

    Returns:
        (lower_bound, upper_bound, outlier_indices)
    """
    if not data:
        return 0.0, 0.0, []

    data_array = np.array(data)
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_indices = [
        i for i, val in enumerate(data)
        if val < lower_bound or val > upper_bound
    ]

    return lower_bound, upper_bound, outlier_indices


def detect_all_outliers(
    metrics_data: Dict[str, Any],
    psnr_multiplier: float = 1.5,
    ssim_multiplier: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Detect outliers across all metrics

    Args:
        metrics_data: Metrics data
        psnr_multiplier: IQR multiplier for PSNR
        ssim_multiplier: IQR multiplier for SSIM

    Returns:
        List of outliers
    """
    details = metrics_data.get("details", [])

    if not details:
        return []

    # Extract data
    psnr_values = [d["psnr"] for d in details]
    ssim_values = [d["ssim"] for d in details]

    # Detect outliers with the IQR method
    psnr_lower, psnr_upper, psnr_outlier_indices = detect_outliers_iqr(
        psnr_values, psnr_multiplier
    )
    ssim_lower, ssim_upper, ssim_outlier_indices = detect_outliers_iqr(
        ssim_values, ssim_multiplier
    )

    print(f"PSNR outlier range: < {psnr_lower:.2f} or > {psnr_upper:.2f}")
    print(f"SSIM outlier range: < {ssim_lower:.4f} or > {ssim_upper:.4f}")

    # Merge outlier indices (PSNR or SSIM)
    outlier_indices = set(psnr_outlier_indices) | set(ssim_outlier_indices)

    # Collect outlier data
    outliers = []
    for idx in outlier_indices:
        detail = details[idx]
        outlier_type = []
        if idx in psnr_outlier_indices:
            outlier_type.append("PSNR")
        if idx in ssim_outlier_indices:
            outlier_type.append("SSIM")

        outliers.append({
            **detail,
            "outlier_type": ", ".join(outlier_type),
            "psnr_lower": psnr_lower,
            "psnr_upper": psnr_upper,
            "ssim_lower": ssim_lower,
            "ssim_upper": ssim_upper,
        })

    # Sort by ascending PSNR
    outliers.sort(key=lambda x: x["psnr"])

    return outliers


def save_outliers_csv(outliers: List[Dict[str, Any]], output_path: Path):
    """
    Save outliers to CSV

    Args:
        outliers: List of outliers
        output_path: Output CSV path
    """
    if not outliers:
        print("No outliers found")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "session_id",
        "camera_id",
        "frame",
        "psnr",
        "ssim",
        "outlier_type",
        "psnr_lower",
        "psnr_upper",
        "ssim_lower",
        "ssim_upper",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outliers)

    print(f"Saved outlier CSV: {output_path}")


def load_image(image_path: Path) -> np.ndarray:
    """Load an image (RGB)"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def visualize_outlier_pairs(
    outliers: List[Dict[str, Any]],
    validation_config: Dict[str, Any],
    render_hash: str,
    output_dir: Path,
    max_pairs: int = 20,
    frames_root: Optional[Path] = None,
    render_root_override: Optional[Path] = None,
):
    """
    Visualize image pairs for outliers

    Args:
        outliers: List of outliers
        validation_config: Settings from consistency.yaml
        render_hash: Render hash
        output_dir: Output directory
        max_pairs: Maximum number to display
        frames_root: Root of real image frames (overrides consistency.yaml if given)
        render_root_override: Root of rendered images (overrides consistency.yaml if given)
    """
    if not outliers:
        print("No outliers found")
        return

    if render_root_override is not None:
        render_root = render_root_override
    else:
        render_root_template = validation_config.get("render", {}).get("root", "{PROJECT_ROOT}/outputs/validation/render")
        render_root = expand_path_template(render_root_template, project_root)

    if frames_root is not None:
        valid_root = frames_root
    else:
        valid_root_template = validation_config.get("frame_extraction", {}).get("output_root", "{PROJECT_ROOT}/outputs/validation/frames")
        valid_root = expand_path_template(valid_root_template, project_root)

    render_path_template = validation_config.get("render", {}).get(
        "path_template", "{render_root}/{session_id}/{hash}/image/Camera.{camera_id}/{frame_number:07d}.png"
    )
    valid_path_template = validation_config.get("frame_extraction", {}).get(
        "output_path_template", "{output_root}/{session_id}/Camera.{camera_id}/{frame_number:07d}.png"
    )
    render_dir_base_template = str(Path(render_path_template).parent)
    valid_dir_base_template = str(Path(valid_path_template).parent)

    # Process up to the maximum count
    for i, outlier in enumerate(outliers[:max_pairs], 1):
        session_id = outlier["session_id"]
        camera_id = outlier["camera_id"]
        frame = outlier["frame"]
        psnr = outlier["psnr"]
        ssim = outlier["ssim"]
        outlier_type = outlier["outlier_type"]

        # Build image paths from templates
        render_dir = expand_path_template(
            render_dir_base_template,
            project_root,
            render_root=str(render_root),
            session_id=session_id,
            hash=render_hash,
            camera_id=camera_id,
        )
        render_path = render_dir / f"{frame:07d}.png"

        valid_dir = expand_path_template(
            valid_dir_base_template,
            project_root,
            output_root=str(valid_root),
            session_id=session_id,
            camera_id=camera_id,
        )
        valid_path = valid_dir / f"{frame:07d}.png"

        try:
            # Load images
            render_img = load_image(render_path)
            valid_img = load_image(valid_path)

            # Compute difference image
            diff_img = np.abs(render_img.astype(float) - valid_img.astype(float))
            diff_img = (diff_img / diff_img.max() * 255).astype(np.uint8) if diff_img.max() > 0 else diff_img.astype(np.uint8)

            # Create plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Real image
            axes[0].imshow(valid_img)
            axes[0].set_title("Real Image", fontsize=14, fontweight='bold')
            axes[0].axis('off')

            # Rendered image
            axes[1].imshow(render_img)
            axes[1].set_title("Rendered Image", fontsize=14, fontweight='bold')
            axes[1].axis('off')

            # Difference image
            axes[2].imshow(diff_img)
            axes[2].set_title("Difference", fontsize=14, fontweight='bold')
            axes[2].axis('off')

            # Title
            fig.suptitle(
                f"Outlier #{i}: {session_id} - Camera {camera_id} - Frame {frame}\n"
                f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f} (Type: {outlier_type})",
                fontsize=16,
                fontweight='bold'
            )

            plt.tight_layout()

            # Save
            output_path = output_dir / f"outlier_{i:03d}_{session_id}_cam{camera_id}_frame{frame}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved outlier image: {output_path.name}")

            plt.close()

        except Exception as e:
            print(f"Failed to visualize outlier #{i}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Outlier detection and visualization"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Metrics JSON file path (default: valid/metrics.json)",
    )
    parser.add_argument(
        "--render-hash",
        type=str,
        required=True,
        help="Render result hash (e.g. 1228b6de)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: valid/outliers)",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="Path to consistency.yaml (default: config/consistency.yaml)",
    )
    parser.add_argument(
        "--psnr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for PSNR (default: 1.5)",
    )
    parser.add_argument(
        "--ssim-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for SSIM (default: 1.5)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of outliers to visualize (default: 20)",
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

    args = parser.parse_args()

    # Load consistency.yaml
    if args.valid_config:
        valid_config_path = Path(args.valid_config)
    else:
        valid_config_path = project_root / "config" / "consistency.yaml"

    if valid_config_path.exists():
        valid_config = load_valid_config(valid_config_path)
        validation_config = valid_config.get("validation", {})
    else:
        validation_config = {}

    # Metrics file path
    if args.metrics:
        metrics_path = Path(args.metrics)
    else:
        metrics_template = validation_config.get("metrics", {}).get(
            "output_path", "{PROJECT_ROOT}/valid/metrics.json"
        )
        metrics_path = expand_path_template(metrics_template, project_root)

    if not metrics_path.exists():
        print(f"Error: metrics file not found: {metrics_path}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "outputs" / "validation" / "outliers"

    print(f"Starting outlier detection")
    print(f"  Metrics file: {metrics_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  PSNR IQR multiplier: {args.psnr_multiplier}")
    print(f"  SSIM IQR multiplier: {args.ssim_multiplier}")

    # Load metrics
    metrics_data = load_metrics(metrics_path)

    # Detect outliers
    print(f"\nDetecting outliers...")
    outliers = detect_all_outliers(
        metrics_data,
        psnr_multiplier=args.psnr_multiplier,
        ssim_multiplier=args.ssim_multiplier,
    )

    print(f"\nOutliers detected: {len(outliers)}")

    if not outliers:
        print("No outliers found")
        return

    # Save to CSV
    csv_path = output_dir / "outliers.csv"
    save_outliers_csv(outliers, csv_path)

    # Show outlier statistics
    print(f"\nOutlier statistics:")
    print(f"  Min PSNR: {min(o['psnr'] for o in outliers):.2f} dB")
    print(f"  Max PSNR: {max(o['psnr'] for o in outliers):.2f} dB")
    print(f"  Min SSIM: {min(o['ssim'] for o in outliers):.4f}")
    print(f"  Max SSIM: {max(o['ssim'] for o in outliers):.4f}")

    # Visualize image pairs
    print(f"\nVisualizing outlier image pairs (up to {args.max_images})...")
    visualize_outlier_pairs(
        outliers,
        validation_config,
        args.render_hash,
        output_dir,
        max_pairs=args.max_images,
        frames_root=Path(args.frames_root) if args.frames_root else None,
        render_root_override=Path(args.render_root) if args.render_root else None,
    )

    print(f"\nDone")


if __name__ == "__main__":
    main()
