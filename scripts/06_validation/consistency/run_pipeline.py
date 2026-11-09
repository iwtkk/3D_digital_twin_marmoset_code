"""
Run the validation pipeline

1. random_frame_selector: random frame selection
2. render_from_selection: rendering
3. extract_frames: real image frame extraction
4. calculate_metrics: PSNR/SSIM computation
5. visualize: visualization
6. outlier_detector: outlier detection
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Path setup
current_script = Path(__file__).resolve()
validation_dir = current_script.parent
# consistency/ → 06_validation/ → scripts/ → project root
project_root = validation_dir.parent.parent.parent
marmo3d_dir = project_root / "src" / "marmo3d"


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Basic settings
    seed: int = 42
    frames_per_animal: int = 100
    animals: List[str] = field(default_factory=lambda: ["cj611", "cj357", "cj425"])

    # Rendering settings
    profile: str = "env_camera"
    split: int = 1
    workers: int = 1
    session_process: int = 1

    # Outlier detection settings
    psnr_multiplier: float = 1.5
    ssim_multiplier: float = 1.5
    max_outlier_images: int = 20

    # Path settings
    output_dir: Optional[Path] = None
    valid_config: Optional[Path] = None

    # Execution control
    dry_run: bool = False
    skip_frame_selection: bool = False
    skip_render: bool = False
    skip_extract: bool = False
    skip_metrics: bool = False
    render_hash: Optional[str] = None  # when reusing existing render results


@dataclass
class PipelineStep:
    """Pipeline step"""
    name: str
    command: List[str]
    description: str
    skip: bool = False


def generate_render_hash(config: PipelineConfig) -> str:
    """Generate the hash with the same logic as render_paths.compute_render_hash (including .blend)"""
    import yaml
    render_spec_path = project_root / "config" / "render_spec.yaml"
    with open(render_spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    sys.path.insert(0, str(project_root / "src"))
    from marmo3d.blender.render_paths import compute_render_hash
    return compute_render_hash(spec, project_root)


def run_step(step: PipelineStep, verbose: bool = True) -> bool:
    """
    Run a single step

    Returns:
        True on success
    """
    if step.skip:
        print(f"\n{'='*60}")
        print(f"SKIP: {step.name}")
        print(f"  {step.description}")
        print(f"{'='*60}")
        return True

    print(f"\n{'='*60}")
    print(f"STEP: {step.name}")
    print(f"  {step.description}")
    print(f"{'='*60}")

    if verbose:
        print(f"Command: {' '.join(step.command)}")
        print()

    try:
        result = subprocess.run(
            step.command,
            cwd=project_root,
            check=True,
        )
        print(f"\n[OK] {step.name} done")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {step.name} failed: return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {step.name} failed: {e}")
        return False


def build_pipeline(config: PipelineConfig) -> List[PipelineStep]:
    """Build the pipeline"""

    # Output directory
    if config.output_dir:
        output_dir = config.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "validation" / "consistency" / f"run_{timestamp}"

    output_dir = Path(output_dir)

    # Paths
    selection_json = output_dir / "selected_frames.json"
    frames_cache = output_dir / "frames_cache.json"
    render_output = project_root / "outputs" / "validation" / "consistency" / "render"
    frames_output = output_dir / "frames"
    metrics_json = output_dir / "metrics.json"
    figures_dir = output_dir / "figures"
    outliers_dir = output_dir / "outliers"

    # render_hash
    if config.render_hash:
        render_hash = config.render_hash
    else:
        render_hash = generate_render_hash(config)

    # consistency.yaml path
    valid_config_path = config.valid_config or (project_root / "config" / "consistency.yaml")

    steps = []

    # Step 1: random_frame_selector
    step1_cmd = [
        sys.executable,
        str(marmo3d_dir / "blender" / "random_frame_selector.py"),
        "--animals", *config.animals,
        "--frames-per-animal", str(config.frames_per_animal),
        "--seed", str(config.seed),
        "--output", str(selection_json),
        "--format", "both",
        "--cache", str(frames_cache),
    ]
    steps.append(PipelineStep(
        name="1. Random Frame Selection",
        command=step1_cmd,
        description=f"Randomly select {config.frames_per_animal} frames per animal",
        skip=config.skip_frame_selection,
    ))

    # Step 2: render_from_selection
    step2_cmd = [
        sys.executable,
        str(marmo3d_dir / "blender" / "render_from_selection.py"),
        "--input", str(selection_json),
        "--profile", config.profile,
        "--split", str(config.split),
        "--workers", str(config.workers),
        "--session-process", str(config.session_process),
        "--output-root", str(render_output),
    ]
    if config.dry_run:
        step2_cmd.append("--dry-run")

    steps.append(PipelineStep(
        name="2. Render from Selection",
        command=step2_cmd,
        description=f"Render selected frames with Blender (profile: {config.profile})",
        skip=config.skip_render,
    ))

    # Step 3: extract_frames
    step3_cmd = [
        sys.executable,
        str(validation_dir / "extract_frames.py"),
        "--input", str(selection_json),
        "--output-root", str(frames_output),
        "--valid-config", str(valid_config_path),
        "--verbose",
    ]
    steps.append(PipelineStep(
        name="3. Extract Frames",
        command=step3_cmd,
        description="Extract real image frames from viddata",
        skip=config.skip_extract,
    ))

    # Step 4: calculate_metrics
    step4_cmd = [
        sys.executable,
        str(validation_dir / "calculate_metrics.py"),
        "--render-hash", render_hash,
        "--frames-root", str(frames_output),
        "--render-root", str(render_output),
        "--output", str(metrics_json),
        "--valid-config", str(valid_config_path),
        "--verbose",
    ]
    steps.append(PipelineStep(
        name="4. Calculate Metrics",
        command=step4_cmd,
        description="Compute PSNR/SSIM",
        skip=config.skip_metrics,
    ))

    # Step 5: visualize
    step5_cmd = [
        sys.executable,
        str(validation_dir / "visualize.py"),
        "--metrics", str(metrics_json),
        "--output-dir", str(figures_dir),
        "--valid-config", str(valid_config_path),
    ]
    steps.append(PipelineStep(
        name="5. Visualize",
        command=step5_cmd,
        description="Generate distribution and scatter plots",
    ))

    # Step 6: outlier_detector
    step6_cmd = [
        sys.executable,
        str(validation_dir / "outlier_detector.py"),
        "--metrics", str(metrics_json),
        "--render-hash", render_hash,
        "--output-dir", str(outliers_dir),
        "--valid-config", str(valid_config_path),
        "--psnr-multiplier", str(config.psnr_multiplier),
        "--ssim-multiplier", str(config.ssim_multiplier),
        "--max-images", str(config.max_outlier_images),
        "--frames-root", str(frames_output),
        "--render-root", str(render_output),
    ]
    steps.append(PipelineStep(
        name="6. Outlier Detection",
        command=step6_cmd,
        description="Detect and visualize outliers",
    ))

    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Run the validation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run (new)
  python run_pipeline.py

  # Run only Steps 4-6 (reuse existing results)
  python run_pipeline.py \\
    --skip-frame-selection \\
    --skip-render \\
    --skip-extract \\
    --render-hash 1228b6de \\
    --output-dir valid/run_20260108_144800

  # Run only Steps 5-6 (metrics already computed)
  python run_pipeline.py \\
    --skip-frame-selection \\
    --skip-render \\
    --skip-extract \\
    --skip-metrics \\
    --render-hash 1228b6de \\
    --output-dir valid/run_20260108_144800

  # Specify frame count and seed
  python run_pipeline.py --frames-per-animal 50 --seed 123

  # Specific animals only
  python run_pipeline.py --animals cj611 cj357

  # dry-run (only print render commands)
  python run_pipeline.py --dry-run
        """
    )

    # Basic settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--frames-per-animal",
        type=int,
        default=100,
        help="Number of frames to select per animal (default: 100)",
    )
    parser.add_argument(
        "--animals",
        nargs="+",
        default=["cj611", "cj357", "cj425"],
        help="Target animal IDs (default: cj611 cj357 cj425)",
    )

    # Rendering settings
    parser.add_argument(
        "--profile",
        type=str,
        default="env_camera",
        help="Render profile (default: env_camera)",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="Number of splits (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--session-process",
        type=int,
        default=1,
        help="Number of parallel sessions (default: 1)",
    )

    # Outlier detection settings
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
        "--max-outlier-images",
        type=int,
        default=20,
        help="Maximum number of outliers to visualize (default: 20)",
    )

    # Path settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: valid/run_{timestamp})",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="Path to consistency.yaml (default: config/consistency.yaml)",
    )

    # Execution control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without rendering",
    )
    parser.add_argument(
        "--skip-frame-selection",
        action="store_true",
        help="Skip frame selection (use existing results)",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip rendering (use existing results)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip frame extraction (use existing results)",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip metric computation (use existing results)",
    )
    parser.add_argument(
        "--render-hash",
        type=str,
        default=None,
        help="Hash of existing render results (required with --skip-render)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args()

    # Build configuration
    config = PipelineConfig(
        seed=args.seed,
        frames_per_animal=args.frames_per_animal,
        animals=args.animals,
        profile=args.profile,
        split=args.split,
        workers=args.workers,
        session_process=args.session_process,
        psnr_multiplier=args.psnr_multiplier,
        ssim_multiplier=args.ssim_multiplier,
        max_outlier_images=args.max_outlier_images,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        valid_config=Path(args.valid_config) if args.valid_config else None,
        dry_run=args.dry_run,
        skip_frame_selection=args.skip_frame_selection,
        skip_render=args.skip_render,
        skip_extract=args.skip_extract,
        skip_metrics=args.skip_metrics,
        render_hash=args.render_hash,
    )

    # --output-dir is required when using skip options
    has_skip = config.skip_frame_selection or config.skip_render or config.skip_extract or config.skip_metrics
    if has_skip and not config.output_dir:
        print("Error: --output-dir is required when using --skip-* options")
        print("  Specify an existing output directory (e.g. valid/run_20260108_144800)")
        sys.exit(1)

    # --render-hash is required with --skip-render
    if config.skip_render and not config.render_hash:
        print("Error: --render-hash is required when using --skip-render")
        sys.exit(1)

    # Build pipeline
    steps = build_pipeline(config)

    print("=" * 60)
    print("Validation pipeline")
    print("=" * 60)
    print(f"Animals: {', '.join(config.animals)}")
    print(f"Frames per animal: {config.frames_per_animal}")
    print(f"Seed: {config.seed}")
    print(f"Profile: {config.profile}")
    if config.render_hash:
        print(f"render_hash: {config.render_hash}")
    print(f"dry-run: {config.dry_run}")
    print(f"skip-frame-selection: {config.skip_frame_selection}")
    print(f"skip-render: {config.skip_render}")
    print(f"skip-extract: {config.skip_extract}")
    print(f"skip-metrics: {config.skip_metrics}")

    # Run pipeline
    success_count = 0
    fail_count = 0

    for step in steps:
        success = run_step(step, verbose=args.verbose)
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"\n[ABORT] Aborting pipeline")
            break

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline finished")
    print("=" * 60)
    print(f"Succeeded: {success_count} / {len(steps)}")
    if fail_count > 0:
        print(f"Failed: {fail_count}")
        sys.exit(1)

    print("\nDone")


if __name__ == "__main__":
    main()
