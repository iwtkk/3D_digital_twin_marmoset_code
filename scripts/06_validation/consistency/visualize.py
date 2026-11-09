"""
Visualize PSNR/SSIM distributions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

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


def plot_distributions(
    metrics_data: Dict[str, Any],
    output_path: Path,
    bins: int = 50,
    figsize: tuple = (14, 6),
):
    """
    Plot PSNR/SSIM distributions

    Args:
        metrics_data: Metrics data
        output_path: Output image path
        bins: Number of histogram bins
        figsize: Figure size
    """
    details = metrics_data.get("details", [])

    if not details:
        print("Error: metrics data is empty")
        return

    # Extract data
    psnr_values = [d["psnr"] for d in details]
    ssim_values = [d["ssim"] for d in details]

    # Statistics
    summary = metrics_data.get("summary", {}).get("total", {})

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # PSNR distribution
    ax1 = axes[0]
    ax1.hist(psnr_values, bins=bins, color='skyblue',
             edgecolor='black', alpha=0.7)
    ax1.axvline(summary.get("psnr_mean", 0), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {summary.get("psnr_mean", 0):.2f}')
    ax1.axvline(summary.get("psnr_mean", 0) - summary.get("psnr_std", 0),
                color='orange', linestyle=':', linewidth=2,
                label=f'±1 std: {summary.get("psnr_std", 0):.2f}')
    ax1.axvline(summary.get("psnr_mean", 0) + summary.get("psnr_std", 0),
                color='orange', linestyle=':', linewidth=2)

    ax1.set_xlabel('PSNR (dB)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'PSNR Distribution (n={len(psnr_values)})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add statistics as text
    stats_text = f'Min: {summary.get("psnr_min", 0):.2f}\nMax: {summary.get("psnr_max", 0):.2f}\nStd: {summary.get("psnr_std", 0):.2f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # SSIM distribution
    ax2 = axes[1]
    ax2.hist(ssim_values, bins=bins, color='lightgreen',
             edgecolor='black', alpha=0.7)
    ax2.axvline(summary.get("ssim_mean", 0), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {summary.get("ssim_mean", 0):.4f}')
    ax2.axvline(summary.get("ssim_mean", 0) - summary.get("ssim_std", 0),
                color='orange', linestyle=':', linewidth=2,
                label=f'±1 std: {summary.get("ssim_std", 0):.4f}')
    ax2.axvline(summary.get("ssim_mean", 0) + summary.get("ssim_std", 0),
                color='orange', linestyle=':', linewidth=2)

    ax2.set_xlabel('SSIM', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'SSIM Distribution (n={len(ssim_values)})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add statistics as text
    stats_text = f'Min: {summary.get("ssim_min", 0):.4f}\nMax: {summary.get("ssim_max", 0):.4f}\nStd: {summary.get("ssim_std", 0):.4f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot: {output_path}")

    plt.close()


def plot_scatter(
    metrics_data: Dict[str, Any],
    output_path: Path,
    figsize: tuple = (8, 6),
):
    """
    Plot a PSNR vs SSIM scatter plot

    Args:
        metrics_data: Metrics data
        output_path: Output image path
        figsize: Figure size
    """
    details = metrics_data.get("details", [])

    if not details:
        print("Error: metrics data is empty")
        return

    # Extract data
    psnr_values = [d["psnr"] for d in details]
    ssim_values = [d["ssim"] for d in details]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(psnr_values, ssim_values, alpha=0.5, c=psnr_values,
                         cmap='viridis', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('PSNR (dB)', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title(f'PSNR vs SSIM (n={len(psnr_values)})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('PSNR (dB)', fontsize=10)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot: {output_path}")

    plt.close()


def plot_by_camera(
    metrics_data: Dict[str, Any],
    output_path: Path,
    camera_mapping: Dict[int, int] = None,
    figsize: tuple = (14, 10),
):
    """
    Plot PSNR/SSIM distributions per camera

    Args:
        metrics_data: Metrics data
        output_path: Output image path
        camera_mapping: Mapping from camera ID to integer (e.g. {23506226: 1, ...})
        figsize: Figure size
    """
    by_camera = metrics_data.get("summary", {}).get("by_camera", {})

    if not by_camera:
        print("Warning: per-camera statistics not found")
        return

    # Sort camera IDs (by mapped integer value if a mapping is given)
    if camera_mapping:
        camera_ids = sorted(by_camera.keys(), key=lambda cid: camera_mapping.get(int(cid), 999))
    else:
        camera_ids = sorted(by_camera.keys())

    psnr_means = [by_camera[cid]["psnr_mean"] for cid in camera_ids]
    psnr_stds = [by_camera[cid]["psnr_std"] for cid in camera_ids]
    ssim_means = [by_camera[cid]["ssim_mean"] for cid in camera_ids]
    ssim_stds = [by_camera[cid]["ssim_std"] for cid in camera_ids]

    # Build camera ID labels
    if camera_mapping:
        camera_labels = [f"Cam {camera_mapping.get(int(cid), cid)}" for cid in camera_ids]
    else:
        camera_labels = camera_ids

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    x = np.arange(len(camera_ids))
    width = 0.6

    # PSNR
    ax1 = axes[0]
    ax1.bar(x, psnr_means, width, yerr=psnr_stds, capsize=5,
            color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR by Camera', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(camera_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # SSIM
    ax2 = axes[1]
    ax2.bar(x, ssim_means, width, yerr=ssim_stds, capsize=5,
            color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM by Camera', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(camera_labels, rotation=45, ha='right')
    ax2.set_xlabel('Camera ID', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-camera distribution plot: {output_path}")

    plt.close()


def extract_animal_from_session_id(session_id: str) -> str:
    """
    Extract the animal ID from a session_id

    e.g.: "cj425_20210426_141115" -> "cj425"
        "dailylife_cj611_20230212_130000" -> "cj611"
    """
    parts = session_id.split('_')
    for part in parts:
        if part.startswith('cj'):
            return part
    return "unknown"


def plot_by_animal(
    metrics_data: Dict[str, Any],
    output_path: Path,
    animal_mapping: Dict[str, str] = None,
    figsize: tuple = (16, 10),
):
    """
    Show PSNR/SSIM distributions per animal as box plots

    Args:
        metrics_data: Metrics data
        output_path: Output image path
        figsize: Figure size
    """
    details = metrics_data.get("details", [])

    if not details:
        print("Error: metrics data is empty")
        return

    # Group data by animal
    from collections import defaultdict
    by_animal = defaultdict(lambda: {"psnr": [], "ssim": []})

    for d in details:
        animal = extract_animal_from_session_id(d["session_id"])
        by_animal[animal]["psnr"].append(d["psnr"])
        by_animal[animal]["ssim"].append(d["ssim"])

    animals = sorted(by_animal.keys())

    if not animals:
        print("Warning: could not extract animal information")
        return

    # Build display names for animal IDs
    animal_labels = [animal_mapping.get(animal, animal) if animal_mapping else animal for animal in animals]

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # PSNR box plot
    ax1 = axes[0]
    psnr_data = [by_animal[animal]["psnr"] for animal in animals]
    bp1 = ax1.boxplot(psnr_data, labels=animal_labels, patch_artist=True,
                       showmeans=True, meanline=True)

    # Set box colors
    for patch in bp1['boxes']:
        patch.set_facecolor('skyblue')
        patch.set_alpha(0.7)

    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR Distribution by Animal', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Animal ID', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add statistics
    for i, animal in enumerate(animals, 1):
        n = len(by_animal[animal]["psnr"])
        mean_val = np.mean(by_animal[animal]["psnr"])
        ax1.text(i, ax1.get_ylim()[1] * 0.95, f'n={n}\nμ={mean_val:.2f}',
                 ha='center', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # SSIM box plot
    ax2 = axes[1]
    ssim_data = [by_animal[animal]["ssim"] for animal in animals]
    bp2 = ax2.boxplot(ssim_data, labels=animal_labels, patch_artist=True,
                       showmeans=True, meanline=True)

    # Set box colors
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)

    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM Distribution by Animal', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Animal ID', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics
    for i, animal in enumerate(animals, 1):
        n = len(by_animal[animal]["ssim"])
        mean_val = np.mean(by_animal[animal]["ssim"])
        ax2.text(i, ax2.get_ylim()[1] * 0.95, f'n={n}\nμ={mean_val:.4f}',
                 ha='center', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-animal distribution plot: {output_path}")

    # Print statistics summary to console
    print(f"\nPer-animal statistics:")
    for animal in animals:
        display_name = animal_mapping.get(animal, animal) if animal_mapping else animal
        psnr_vals = by_animal[animal]["psnr"]
        ssim_vals = by_animal[animal]["ssim"]
        print(f"  {display_name}:")
        print(f"    Samples: {len(psnr_vals)}")
        print(f"    PSNR: {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} (range: {np.min(psnr_vals):.2f} - {np.max(psnr_vals):.2f})")
        print(f"    SSIM: {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f} (range: {np.min(ssim_vals):.4f} - {np.max(ssim_vals):.4f})")

    plt.close()


def plot_by_animal_and_camera(
    metrics_data: Dict[str, Any],
    output_path: Path,
    camera_mapping: Dict[int, int] = None,
    animal_mapping: Dict[str, str] = None,
    figsize_per_animal: tuple = (4, 8),
    width = 0.8,
    labelsize = 12,
    fontsize = 12
):
    """
    Plot per-camera PSNR/SSIM distributions for each animal

    Args:
        metrics_data: Metrics data
        output_path: Output image path
        camera_mapping: Mapping from camera ID to integer (e.g. {23506226: 1, ...})
        figsize_per_animal: Figure size per animal
    """
    details = metrics_data.get("details", [])

    if not details:
        print("Error: metrics data is empty")
        return

    # Group data by animal x camera
    from collections import defaultdict
    by_animal_camera = defaultdict(lambda: defaultdict(lambda: {"psnr": [], "ssim": []}))

    for d in details:
        animal = extract_animal_from_session_id(d["session_id"])
        camera = d["camera_id"]
        by_animal_camera[animal][camera]["psnr"].append(d["psnr"])
        by_animal_camera[animal][camera]["ssim"].append(d["ssim"])

    animals = sorted(by_animal_camera.keys())

    if not animals:
        print("Warning: could not extract animal information")
        return

    # Compute the global value range over all animals and cameras (to share the y-axis scale)
    all_psnr = [v for a in animals for cam in by_animal_camera[a] for v in by_animal_camera[a][cam]["psnr"]]
    all_ssim = [v for a in animals for cam in by_animal_camera[a] for v in by_animal_camera[a][cam]["ssim"]]
    psnr_margin = 0.05 * (max(all_psnr) - min(all_psnr))
    ssim_margin = 0.05 * (max(all_ssim) - min(all_ssim))
    psnr_ylim = (min(all_psnr) - psnr_margin, max(all_psnr) + psnr_margin)
    ssim_ylim = (min(all_ssim) - ssim_margin, max(all_ssim) + ssim_margin)

    # Create plots per animal
    for animal in animals:
        display_name = animal_mapping.get(animal, animal) if animal_mapping else animal
        # Sort camera IDs (by mapped integer value if a mapping is given)
        if camera_mapping:
            cameras = sorted(by_animal_camera[animal].keys(), key=lambda cid: camera_mapping.get(int(cid), 999))
        else:
            cameras = sorted(by_animal_camera[animal].keys())

        if not cameras:
            continue

        # Build camera ID labels
        if camera_mapping:
            camera_labels = [f"Cam {camera_mapping.get(int(cam), cam)}" for cam in cameras]
        else:
            camera_labels = cameras

        fig, axes = plt.subplots(2, 1, figsize=figsize_per_animal)
        
        
        # PSNR box plot
        ax1 = axes[0]
        psnr_data = [by_animal_camera[animal][cam]["psnr"] for cam in cameras]
        bp1 = ax1.boxplot(psnr_data, labels=camera_labels, patch_artist=True,
                           showmeans=True, meanline=True, widths=width)

        # Set box colors
        for patch in bp1['boxes']:
            patch.set_facecolor('skyblue')
            patch.set_alpha(0.7)
            
        ax1.set_xticklabels(camera_labels, fontsize=labelsize , rotation=45, ha='right')
        ax1.tick_params(axis='y', labelsize=labelsize)

        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title(f'PSNR Distribution by Camera - {display_name}', fontsize=fontsize, fontweight='bold')
        ax1.set_xlabel('Camera ID', fontsize=12)
        ax1.set_ylim(psnr_ylim)
        ax1.grid(True, alpha=0.3, axis='y')

        # SSIM box plot
        ax2 = axes[1]
        ssim_data = [by_animal_camera[animal][cam]["ssim"] for cam in cameras]
        bp2 = ax2.boxplot(ssim_data, labels=camera_labels, patch_artist=True,
                           showmeans=True, meanline=True, widths=width)

        # Set box colors
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
            
        ax2.set_xticklabels(camera_labels, fontsize=labelsize, rotation=45, ha='right')
        ax2.tick_params(axis='y', labelsize=labelsize)

        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title(f'SSIM Distribution by Camera - {display_name}', fontsize=fontsize, fontweight='bold')
        ax2.set_xlabel('Camera ID', fontsize=12)
        ax2.set_ylim(ssim_ylim)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save (one file per animal)
        animal_output_path = output_path.parent / f"by_camera_{display_name}.png"
        animal_output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(animal_output_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-animal camera distribution plot: {animal_output_path}")

        # Print statistics summary to console
        print(f"\n  {display_name} - per-camera statistics:")
        for cam in cameras:
            psnr_vals = by_animal_camera[animal][cam]["psnr"]
            ssim_vals = by_animal_camera[animal][cam]["ssim"]
            cam_label = f"Cam {camera_mapping.get(int(cam), cam)}" if camera_mapping else cam
            print(f"    {cam_label}: n={len(psnr_vals)}, PSNR={np.mean(psnr_vals):.2f}±{np.std(psnr_vals):.2f}, SSIM={np.mean(ssim_vals):.4f}±{np.std(ssim_vals):.4f}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PSNR/SSIM distributions"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Metrics JSON file path (default: valid/metrics.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: valid/figures)",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="Path to consistency.yaml (default: config/consistency.yaml)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
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
        output_dir = project_root / "outputs" / "validation" / "figures"

    print(f"Metrics file: {metrics_path}")
    print(f"Output directory: {output_dir}")

    # Get camera ID mapping
    camera_mapping = None
    animal_mapping = None
    if valid_config_path.exists():
        full_config = load_valid_config(valid_config_path)
        camera_mapping = full_config.get("figure", {}).get("mapping_table", None)
        if camera_mapping:
            print(f"  Camera mapping: {len(camera_mapping)} cameras")
        animal_mapping = full_config.get("figure", {}).get("animal_mapping", None)
        if animal_mapping:
            print(f"  Animal mapping: {animal_mapping}")

    # Load metrics
    metrics_data = load_metrics(metrics_path)

    print(f"\nStarting visualization")
    print(f"  Total samples: {metrics_data.get('successful_calculations', 0)}")

    # Create distribution plot
    plot_distributions(
        metrics_data,
        output_dir / "distribution.png",
        bins=args.bins,
    )

    # Create scatter plot
    plot_scatter(
        metrics_data,
        output_dir / "scatter.png",
    )

    # Create per-camera distribution plot
    plot_by_camera(
        metrics_data,
        output_dir / "by_camera.png",
        camera_mapping=camera_mapping,
    )

    # Create per-animal distribution plot
    plot_by_animal(
        metrics_data,
        output_dir / "by_animal.png",
        animal_mapping=animal_mapping,
    )

    # Create per-camera distribution plots for each animal
    plot_by_animal_and_camera(
        metrics_data,
        output_dir / "by_animal_and_camera.png",  # Base path (actually one file per animal)
        camera_mapping=camera_mapping,
        animal_mapping=animal_mapping,
    )

    print(f"\nDone")


if __name__ == "__main__":
    main()
