"""Sweep Plotter: Validate and plot results from a sweep folder.

Reads prediction CSVs from a sweep output folder, runs validation on each,
and generates line plots with error bars (std dev from multiple runs).

Input folder structure:
    outputs/sweeps/top_k_vs_rerank/
    ├── metadata.json
    ├── rerank_strategy=frequency_top_k=4_run0_preds.csv
    ├── rerank_strategy=frequency_top_k=4_run1_preds.csv
    └── ...

Output:
    outputs/sweeps/top_k_vs_rerank/
    ├── sweep_results.csv        # Validation scores for all runs
    ├── plot_final_score.png     # Line plot with error bars
    ├── plot_value_score.png
    └── plot_ref_score.png

Usage:
    python workflows/sweeps/sweep_plot.py outputs/sweeps/top_k_vs_rerank
    python workflows/sweeps/sweep_plot.py outputs/sweeps/top_k_vs_rerank --metric value_score
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kohakuengine import Config, Script


def parse_filename(
    filename: str, line_param: str, x_param: str
) -> tuple[Any, Any, int] | None:
    """Extract line value, x value, and run index from prediction filename.

    Expected format: {line_param}={line_val}_{x_param}={x_val}_run{idx}_preds.csv
    Also handles old format without run index: {line_param}={line_val}_{x_param}={x_val}_preds.csv

    Both line values and x values can contain underscores (e.g., jina_v3_img, openai_GPT-5-mini).
    """
    # Try new format with run index
    # Match x_val as everything up to _run{digits}_preds.csv
    pattern = (
        rf"{re.escape(line_param)}=(.+)_{re.escape(x_param)}=(.+)_run(\d+)_preds\.csv"
    )
    match = re.match(pattern, filename)
    if match:
        line_str, x_str, run_str = match.groups()
        run_idx = int(run_str)
    else:
        # Try old format without run index
        pattern = rf"{re.escape(line_param)}=(.+)_{re.escape(x_param)}=(.+)_preds\.csv"
        match = re.match(pattern, filename)
        if not match:
            return None
        line_str, x_str = match.groups()
        run_idx = 0

    # Parse line value
    line_val: Any
    if line_str.lower() == "none":
        line_val = None
    else:
        line_val = line_str

    # Parse x value (try int, then float, then string)
    x_val: Any
    try:
        x_val = int(x_str)
    except ValueError:
        try:
            x_val = float(x_str)
        except ValueError:
            x_val = x_str

    return line_val, x_val, run_idx


def run_validation(pred_path: Path, truth_path: str) -> dict[str, float]:
    """Run validation and extract scores."""
    import io

    validate_config = Config(
        globals_dict={
            "truth": truth_path,
            "pred": str(pred_path),
            "show_errors": 0,
            "verbose": False,
        }
    )
    validate_script = Script("scripts/wattbot_validate.py", config=validate_config)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        validate_script.run()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    return parse_validation_output(output)


def parse_validation_output(output: str) -> dict[str, float]:
    """Parse validation script output to extract scores."""
    scores: dict[str, float] = {}
    for line in output.split("\n"):
        if "Component scores:" in line:
            parts = line.split("Component scores:")[1].strip()
            for part in parts.split(","):
                key, val = part.strip().split("=")
                scores[f"{key.strip()}_score"] = float(val.strip())
        if "FINAL WATTBOT SCORE:" in line:
            score_str = line.split("FINAL WATTBOT SCORE:")[1].strip()
            scores["final_score"] = float(score_str)
    return scores


def save_results(results: list[dict[str, Any]], path: Path) -> None:
    """Save results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def aggregate_runs(
    results: list[dict[str, Any]],
    line_param: str,
    x_param: str,
    metric: str,
) -> dict[Any, dict[Any, tuple[float, float, list[float]]]]:
    """Aggregate multiple runs into mean and std.

    Returns:
        {line_val: {x_val: (mean, std, [individual_scores])}}
    """
    # Group scores by (line_val, x_val)
    grouped: dict[tuple[Any, Any], list[float]] = defaultdict(list)
    for row in results:
        line_val = row[line_param]
        x_val = row[x_param]
        score = row.get(metric, 0)
        grouped[(line_val, x_val)].append(score)

    # Compute mean and std for each group
    aggregated: dict[Any, dict[Any, tuple[float, float, list[float]]]] = defaultdict(
        dict
    )
    for (line_val, x_val), scores in grouped.items():
        mean = np.mean(scores)
        std = np.std(scores) if len(scores) > 1 else 0.0
        aggregated[line_val][x_val] = (mean, std, scores)

    return dict(aggregated)


def plot_metric(
    results: list[dict[str, Any]],
    line_param: str,
    x_param: str,
    metric: str,
    output_path: Path,
) -> None:
    """Generate line plot with shaded ±1 std dev range and max value dotted line."""
    aggregated = aggregate_runs(results, line_param, x_param, metric)

    _, ax = plt.subplots(figsize=(10, 6))

    # Track global max for star marker
    global_max_val = float("-inf")
    global_max_x_idx = None
    global_max_line = None
    global_max_color = None

    # Get all unique x values across all lines for consistent ordering
    all_x_vals: set[Any] = set()
    for data in aggregated.values():
        all_x_vals.update(data.keys())

    # Sort x values - try numeric first, then string
    try:
        sorted_x_vals = sorted(all_x_vals, key=lambda x: float(x))
    except (ValueError, TypeError):
        sorted_x_vals = sorted(all_x_vals, key=str)

    # Check if x-axis is categorical (strings)
    is_categorical = any(isinstance(x, str) for x in sorted_x_vals)

    # Create x positions for categorical axis
    if is_categorical:
        x_positions = list(range(len(sorted_x_vals)))
        x_to_pos = {x: i for i, x in enumerate(sorted_x_vals)}
    else:
        x_positions = sorted_x_vals
        x_to_pos = {x: x for x in sorted_x_vals}

    for line_val in sorted(aggregated.keys(), key=lambda x: str(x)):
        data = aggregated[line_val]
        x_vals = sorted(data.keys(), key=lambda x: x_to_pos.get(x, 0))
        x_plot = [x_to_pos[x] for x in x_vals]
        means = np.array([data[x][0] for x in x_vals])
        stds = np.array([data[x][1] for x in x_vals])
        maxs = np.array([max(data[x][2]) for x in x_vals])  # Max from individual scores

        label = str(line_val) if line_val is not None else "None"

        # Plot mean line with markers
        (line,) = ax.plot(
            x_plot,
            means,
            marker="o",
            linewidth=2,
            markersize=8,
            label=f"{line_param}={label}",
        )

        # Add shaded ±1 std dev range
        ax.fill_between(
            x_plot,
            means - stds,
            means + stds,
            alpha=0.2,
            color=line.get_color(),
        )

        # Add max value dotted line
        ax.plot(
            x_plot,
            maxs,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            color=line.get_color(),
        )

        # Track global max
        for i, x in enumerate(x_vals):
            if maxs[i] > global_max_val:
                global_max_val = maxs[i]
                global_max_x_idx = x_to_pos[x]
                global_max_line = line_val
                global_max_color = line.get_color()

    # Add star marker at global max
    if global_max_x_idx is not None:
        line_label = str(global_max_line) if global_max_line is not None else "None"
        ax.plot(
            global_max_x_idx,
            global_max_val,
            marker="*",
            markersize=20,
            color=global_max_color,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=10,
        )
        ax.annotate(
            f"{global_max_val:.4f} ({line_label})",
            (global_max_x_idx, global_max_val),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
            fontweight="bold",
        )

    # Set x-axis ticks for categorical values
    if is_categorical:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(x) for x in sorted_x_vals], rotation=45, ha="right")

    ax.set_xlabel(x_param)
    ax.set_ylabel(metric)
    ax.set_title(
        f"{metric} vs {x_param} by {line_param}\n(solid=mean, dashed=max, star=global max, shaded=±1 std dev)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


def print_summary_table(
    results: list[dict[str, Any]],
    line_param: str,
    x_param: str,
    metric: str,
) -> None:
    """Print summary table with mean ± std."""
    aggregated = aggregate_runs(results, line_param, x_param, metric)

    # Get all x values
    all_x = sorted({row[x_param] for row in results})

    # Print header
    print(f"\n{'=' * 80}")
    print(f"Summary: {metric} (mean ± std)")
    print("=" * 80)

    header = f"{line_param:<20}"
    for x_val in all_x:
        header += f" {x_val:>14}"
    print(header)
    print("-" * len(header))

    # Print each line
    for line_val in sorted(aggregated.keys(), key=lambda x: str(x)):
        data = aggregated[line_val]
        label = str(line_val) if line_val is not None else "None"
        row_str = f"{label:<20}"
        for x_val in all_x:
            if x_val in data:
                mean, std, _ = data[x_val]
                row_str += f" {mean:.4f}±{std:.4f}"
            else:
                row_str += f" {'N/A':>14}"
        print(row_str)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results with std dev")
    parser.add_argument("sweep_dir", type=Path, help="Sweep output directory")
    parser.add_argument(
        "--metric", default=None, help="Specific metric to plot (default: all)"
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    # Load metadata
    metadata_path = sweep_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {sweep_dir}")
        sys.exit(1)

    with metadata_path.open() as f:
        metadata = json.load(f)

    line_param = metadata["line_param"]
    x_param = metadata["x_param"]
    questions = metadata.get("questions", "data/train_QA.csv")
    num_runs = metadata.get("num_runs", 1)

    print("=" * 70)
    print(f"Processing sweep: {sweep_dir}")
    print("=" * 70)
    print(f"Line parameter: {line_param}")
    print(f"X parameter: {x_param}")
    print(f"Runs per config: {num_runs}")
    print(f"Truth file: {questions}")
    print("=" * 70)

    # Find all prediction files
    pred_files = list(sweep_dir.glob("*_preds.csv"))
    if not pred_files:
        print("Error: No prediction files found")
        sys.exit(1)

    print(f"Found {len(pred_files)} prediction files")

    # Validate each and collect results
    results: list[dict[str, Any]] = []
    for pred_path in sorted(pred_files):
        parsed = parse_filename(pred_path.name, line_param, x_param)
        if parsed is None:
            print(f"Skipping: {pred_path.name} (doesn't match expected pattern)")
            continue

        line_val, x_val, run_idx = parsed
        print(
            f"Validating: {line_param}={line_val}, {x_param}={x_val}, run={run_idx}...",
            end=" ",
        )

        scores = run_validation(pred_path, questions)
        print(f"final_score={scores.get('final_score', 0):.4f}")

        results.append(
            {
                line_param: line_val,
                x_param: x_val,
                "run": run_idx,
                **scores,
            }
        )

    # Save results CSV
    results_path = sweep_dir / "sweep_results.csv"
    save_results(results, results_path)
    print(f"\nResults saved: {results_path}")

    # Determine which metrics to plot
    if args.metric:
        metrics = [args.metric]
    else:
        metrics = ["final_score", "value_score", "ref_score"]

    # Generate plots
    for metric in metrics:
        if not results or metric not in results[0]:
            print(f"Warning: metric '{metric}' not found in results, skipping")
            continue

        plot_path = sweep_dir / f"plot_{metric}.png"
        plot_metric(results, line_param, x_param, metric, plot_path)
        print_summary_table(results, line_param, x_param, metric)

    print("\n" + "=" * 70)
    print("Plotting complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
