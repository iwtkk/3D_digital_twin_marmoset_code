#!/usr/bin/env python3

import argparse
import glob
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_BASE = (
    PROJECT_ROOT
    / "3D_digital_twin_marmoset_data"
    / "marmoset_keypoint_data"
    / "3D_temporal_keypoint_data"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_distribution.png"


def load_kp3d_err_values(paths, key: str) -> tuple[np.ndarray, list[tuple[str, int, float, float, float]]]:
    aggregated = []
    per_file_stats = []

    for path in sorted(paths):
        with open(path, "rb") as f:
            data = pickle.load(f)

        if key not in data:
            raise KeyError(f"{key} not found in {path}")

        arr = np.asarray(data[key], dtype=float)
        flat = arr.reshape(-1)
        valid = flat[np.isfinite(flat)]
        if valid.size == 0:
            continue

        aggregated.append(valid)
        per_file_stats.append(
            (
                path,
                valid.size,
                float(np.mean(valid)),
                float(np.median(valid)),
                float(np.percentile(valid, 95)),
            )
        )

    if not aggregated:
        raise RuntimeError("No finite values collected from input pickles.")

    return np.concatenate(aggregated), per_file_stats


def plot_histogram(
    values: np.ndarray,
    bins: int,
    output: Path,
    title: str,
    xmin: float | None = None,
    xmax: float | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    dpi: int = 200,
    vlines: tuple[tuple[float, str], ...] = (),
) -> int:
    if bins <= 0:
        raise ValueError("bins must be positive.")

    hist_vals = values
    range_min = xmin if xmin is not None else float(np.min(values))
    range_max = xmax if xmax is not None else float(np.max(values))

    if xmin is not None or xmax is not None:
        if range_min >= range_max:
            raise ValueError(f"Invalid histogram range [{range_min}, {range_max}]")
        mask = (values >= range_min) & (values <= range_max)
        hist_vals = values[mask]
        if hist_vals.size == 0:
            raise RuntimeError(f"No values within specified range [{range_min}, {range_max}]")
    else:
        range_min, range_max = float(np.min(values)), float(np.max(values))

    if np.isclose(range_min, range_max):
        span = max(1e-6, abs(range_min) * 0.05)
        if span == 0.0:
            span = 1e-3
        range_min -= span
        range_max += span

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(hist_vals, bins=bins, range=(range_min, range_max), color="#1f77b4", alpha=0.85)
    # ax.set_xlabel("Reprojection error [px]", fontsize=25)
    # ax.set_ylabel("Number of keypoints across all frames (×10$^6$)", fontsize=20)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    # ax.set_title(title)
    ax.set_xlim(range_min, range_max)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    for x, label, color in vlines:
        if x is None:
            continue
        display_x = x
        clamped = False
        if display_x < range_min:
            display_x = range_min
            clamped = True
        if display_x > range_max:
            display_x = range_max
            clamped = True
        lbl = f"{label} (clamped)" if clamped else label
        ax.axvline(display_x, color=color, linestyle="--", linewidth=1.4, label=lbl)
    if vlines:
        ax.legend(fontsize=20, loc="upper right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return int(hist_vals.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot distribution of kp3d_err from pickle files.")
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Root directory containing session folders with kp3d_fxdJointLen.pickle.",
    )
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=None,
        help="One or more glob patterns for kp3d pickle files. Overrides --base.",
    )
    parser.add_argument("--key", default="kp3d_err", help="Key to extract from each pickle.")
    parser.add_argument("--bins", type=int, default=200, help="Number of histogram bins.")
    parser.add_argument("--fig-width", type=float, default=10.0, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=6.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI used when saving figures.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save the histogram image.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Lower limit for the x-axis when plotting the main histogram.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=40.0,
        help="Upper limit for the x-axis when plotting the main histogram.",
    )
    parser.add_argument(
        "--focus-mean-p99",
        action="store_true",
        help="Additionally save a histogram zoomed to [mean, p99] using same bin count.",
    )
    parser.add_argument(
        "--focus-output",
        type=Path,
        default=None,
        help="Optional filename for the mean-to-p99 focused histogram. Defaults to "
        "<output stem>_mean_p99.png when --focus-mean-p99 is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[str] = []
    patterns = args.pattern or [str(args.base / "*" / "kp3d_fxdJointLen.pickle")]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(matches)
        else:
            print(f"Warning: pattern '{pattern}' matched no files.")

    if paths:
        paths = sorted(set(paths))
    else:
        raise FileNotFoundError(f"No files matched any of: {patterns}")

    values, per_file_stats = load_kp3d_err_values(paths, args.key)

    print(f"Loaded {values.size} kp3d_err values from {len(per_file_stats)} files.")
    print("Per-file stats (path, count, mean, median, p95):")
    for path, count, mean, median, p95 in per_file_stats:
        print(f"- {path}: n={count}, mean={mean:.3f}, median={median:.3f}, p95={p95:.3f}")

    overall_mean = float(np.mean(values))
    overall_median = float(np.median(values))
    overall_p95 = float(np.percentile(values, 95))
    overall_p99 = float(np.percentile(values, 99))
    overall_max = float(np.max(values))
    overall_min = float(np.min(values))

    print("\nAggregated stats:")
    print(f"mean={overall_mean:.3f}, median={overall_median:.3f}, p95={overall_p95:.3f}, p99={overall_p99:.3f}")
    print(f"min={overall_min:.3f}, max={overall_max:.3f}")

    title = f"kp3d_err distribution (n={values.size})"
    figsize = (args.fig_width, args.fig_height)
    vlines = (
        (overall_mean, f"mean={overall_mean:.2f}", '#BF0000'),
        (overall_p99, f"p99={overall_p99:.2f}", '#00BF00'),
    )
    used_main = plot_histogram(
        values,
        args.bins,
        args.output,
        title,
        args.clip_min,
        args.clip_max,
        figsize,
        args.dpi,
        vlines,
    )
    print(f"Histogram saved to {args.output} (values within plotted range: {used_main})")

    if args.focus_mean_p99:
        focus_min = overall_mean
        focus_max = overall_p99
        focus_title = f"kp3d_err mean→p99 (n={values.size})"
        focus_output = args.focus_output
        if focus_output is None:
            focus_output = args.output.with_name(f"{args.output.stem}_mean_p99{args.output.suffix}")
        used_focus = plot_histogram(
            values,
            args.bins,
            focus_output,
            focus_title,
            focus_min,
            focus_max,
            figsize,
            args.dpi,
            vlines,
        )
        print(
            f"Mean→p99 histogram saved to {focus_output} "
            f"(range: [{focus_min:.3f}, {focus_max:.3f}], values within range: {used_focus})"
        )


if __name__ == "__main__":
    main()
