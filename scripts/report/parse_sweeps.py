#!/usr/bin/env python3
"""Parse sweep results and generate summary tables for the report.

Usage:
    python scripts/report/parse_sweeps.py
    python scripts/report/parse_sweeps.py --latex  # Output LaTeX tables
    python scripts/report/parse_sweeps.py --sweep llm_model_vs_top_k  # Single sweep
    python scripts/report/parse_sweeps.py --significance  # Run statistical significance tests
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SWEEPS_DIR = Path("outputs/sweeps")

_LATEX_EMPTY_CELL = " & --"


def parse_sweep(sweep_path: Path) -> dict | None:
    """Parse a single sweep directory and return aggregated results."""
    results_path = sweep_path / "sweep_results.csv"
    metadata_path = sweep_path / "metadata.json"

    if not results_path.exists() or not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    with open(results_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    line_param = metadata["line_param"]
    x_param = metadata["x_param"]

    # Aggregate by (line_val, x_val)
    grouped = defaultdict(list)
    for row in rows:
        line_val = row[line_param]
        x_val = row[x_param]
        scores = {
            "final_score": float(row.get("final_score", 0)),
            "value_score": float(row.get("value_score", 0)),
            "ref_score": float(row.get("ref_score", 0)),
        }
        grouped[(line_val, x_val)].append(scores)

    # Compute statistics for each group
    results = {}
    for (line_val, x_val), score_list in grouped.items():
        final_scores = [s["final_score"] for s in score_list]
        value_scores = [s["value_score"] for s in score_list]
        ref_scores = [s["ref_score"] for s in score_list]

        results[(line_val, x_val)] = {
            "final": {
                "mean": np.mean(final_scores),
                "std": np.std(final_scores),
                "max": np.max(final_scores),
                "scores": final_scores,  # Keep raw scores for significance tests
            },
            "value": {
                "mean": np.mean(value_scores),
                "std": np.std(value_scores),
                "max": np.max(value_scores),
                "scores": value_scores,
            },
            "ref": {
                "mean": np.mean(ref_scores),
                "std": np.std(ref_scores),
                "max": np.max(ref_scores),
                "scores": ref_scores,
            },
            "n": len(score_list),
        }

    return {
        "name": sweep_path.name,
        "metadata": metadata,
        "results": results,
    }


def get_sorted_values(results: dict, key_idx: int) -> list:
    """Get sorted unique values from result keys."""
    vals = sorted({k[key_idx] for k in results.keys()}, key=str)
    # Try numeric sort if possible
    try:
        return sorted(vals, key=lambda x: float(x))
    except (ValueError, TypeError):
        return vals


def compute_average_ranks(
    results: dict, line_vals: list, x_vals: list, metric: str = "final"
) -> dict[str, float]:
    """Compute average rank across x_vals for each line_val.

    For each x value (column), rank all line values by their score.
    Then compute the average rank for each line value across all columns.
    Lower rank = better (rank 1 is best).
    """
    # For each x_val, compute ranks of line_vals
    ranks_per_line = defaultdict(list)

    for x in x_vals:
        # Collect (line_val, score) pairs for this x
        scores_for_x = []
        for line in line_vals:
            if (line, x) in results:
                scores_for_x.append((line, results[(line, x)][metric]["mean"]))

        if not scores_for_x:
            continue

        # Sort by score descending (higher is better)
        scores_for_x.sort(key=lambda item: item[1], reverse=True)

        # Assign ranks (1-indexed)
        for rank, (line, _) in enumerate(scores_for_x, start=1):
            ranks_per_line[line].append(rank)

    # Compute average rank for each line_val
    avg_ranks = {}
    for line in line_vals:
        if line in ranks_per_line and ranks_per_line[line]:
            avg_ranks[line] = np.mean(ranks_per_line[line])
        else:
            avg_ranks[line] = float("inf")

    return avg_ranks


def print_sweep_summary(data: dict, metric: str = "final") -> None:
    """Print summary table for a sweep."""
    meta = data["metadata"]
    results = data["results"]
    line_param = meta["line_param"]
    x_param = meta["x_param"]

    print(f"\n{'=' * 70}")
    print(f"SWEEP: {data['name']}")
    print(f"Line: {line_param}, X: {x_param}")
    print(f"{'=' * 70}")

    # Find best result
    best_key = max(results.keys(), key=lambda k: results[k][metric]["max"])
    best = results[best_key][metric]
    print(f"Best ({metric}): {line_param}={best_key[0]}, {x_param}={best_key[1]}")
    print(f"  Max: {best['max']:.4f}, Mean: {best['mean']:.4f} +/- {best['std']:.4f}")

    # Print table
    line_vals = get_sorted_values(results, 0)
    x_vals = get_sorted_values(results, 1)

    # Compute average ranks
    avg_ranks = compute_average_ranks(results, line_vals, x_vals, metric)
    print(f"\nAverage ranks across {x_param}:")
    for line in sorted(line_vals, key=lambda l: avg_ranks.get(l, float("inf"))):
        print(f"  {line}: {avg_ranks.get(line, 'N/A'):.2f}")

    print(f"\nMean {metric} scores:")
    col_width = max(10, max(len(str(x)) for x in x_vals) + 2)
    header = f"{line_param[:20]:<22}"
    for x in x_vals:
        header += f" {str(x):>{col_width}}"
    print(header)
    print("-" * len(header))

    for line in line_vals:
        row = f"{str(line)[:20]:<22}"
        for x in x_vals:
            if (line, x) in results:
                r = results[(line, x)][metric]
                row += f" {r['mean']:.4f}"
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)


def generate_latex_table(
    data: dict,
    metric: str = "final",
    mark_second_best: bool = True,
    show_avg_rank: bool = True,
) -> str:
    """Generate LaTeX table for a sweep.

    Args:
        data: Parsed sweep data
        metric: Which metric to display (final, value, ref)
        mark_second_best: If True, underline second best in each column
        show_avg_rank: If True, add average rank column when >=3 x values
    """
    meta = data["metadata"]
    results = data["results"]
    line_param = meta["line_param"]

    line_vals = get_sorted_values(results, 0)
    x_vals = get_sorted_values(results, 1)

    # Compute average ranks if we have enough x values
    avg_ranks = None
    if show_avg_rank and len(x_vals) >= 3:
        avg_ranks = compute_average_ranks(results, line_vals, x_vals, metric)

    # Find best and second best per column
    best_per_col = {}
    second_best_per_col = {}
    global_best = -1

    for x in x_vals:
        col_scores = []
        for line in line_vals:
            if (line, x) in results:
                val = results[(line, x)][metric]["mean"]
                col_scores.append((line, val))
                if val > global_best:
                    global_best = val

        # Sort by score descending
        col_scores.sort(key=lambda item: item[1], reverse=True)
        if len(col_scores) >= 1:
            best_per_col[x] = col_scores[0][0]
        if len(col_scores) >= 2:
            second_best_per_col[x] = col_scores[1][0]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + f"{data['name'].replace('_', ' ').title()}" + "}")
    lines.append("\\label{tab:" + data["name"] + "}")

    # Column spec
    n_cols = len(x_vals)
    if avg_ranks:
        n_cols += 1  # Add avg rank column
    col_spec = "l" + "c" * n_cols
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")

    # Header
    header = f"\\textbf{{{line_param}}}"
    for x in x_vals:
        header += f" & \\textbf{{{x}}}"
    if avg_ranks:
        header += " & \\textbf{Avg Rank}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for line in line_vals:
        row = str(line).replace("_", "\\_")
        for x in x_vals:
            if (line, x) in results:
                r = results[(line, x)][metric]
                val_str = f"{r['mean']:.3f}"
                # Bold if best in column
                if best_per_col.get(x) == line:
                    val_str = f"\\textbf{{{val_str}}}"
                # Underline if second best in column (and mark_second_best enabled)
                elif mark_second_best and second_best_per_col.get(x) == line:
                    val_str = f"\\underline{{{val_str}}}"
                row += f" & {val_str}"
            else:
                row += _LATEX_EMPTY_CELL

        # Add average rank if available
        if avg_ranks:
            rank = avg_ranks.get(line, float("inf"))
            if rank != float("inf"):
                row += f" & {rank:.2f}"
            else:
                row += _LATEX_EMPTY_CELL

        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def run_significance_tests(data: dict, metric: str = "final") -> None:
    """Run paired t-tests between configurations within each sweep.

    For each x value (e.g., top_k), compares the best configuration against others.
    Reports p-values and effect sizes (Cohen's d).
    """
    meta = data["metadata"]
    results = data["results"]
    line_param = meta["line_param"]
    x_param = meta["x_param"]

    print(f"\n{'=' * 70}")
    print(f"STATISTICAL SIGNIFICANCE TESTS: {data['name']}")
    print(f"Metric: {metric}, Comparing: {line_param}")
    print(f"{'=' * 70}")

    line_vals = get_sorted_values(results, 0)
    x_vals = get_sorted_values(results, 1)

    for x in x_vals:
        print(f"\n--- {x_param} = {x} ---")

        # Get scores for each line value at this x
        configs = []
        for line in line_vals:
            if (line, x) in results:
                scores = results[(line, x)][metric].get("scores", [])
                if len(scores) >= 2:  # Need at least 2 samples for t-test
                    configs.append((line, scores, np.mean(scores)))

        if len(configs) < 2:
            print("  Insufficient data for comparison")
            continue

        # Sort by mean score descending
        configs.sort(key=lambda c: c[2], reverse=True)
        best_name, best_scores, best_mean = configs[0]

        print(f"  Best: {best_name} (mean={best_mean:.4f}, n={len(best_scores)})")
        print()

        for other_name, other_scores, other_mean in configs[1:]:
            # Welch's t-test (does not assume equal variances)
            _, p_value = stats.ttest_ind(
                best_scores, other_scores, equal_var=False
            )

            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(best_scores) + np.var(other_scores)) / 2)
            if pooled_std > 0:
                cohens_d = (best_mean - other_mean) / pooled_std
            else:
                cohens_d = 0.0

            # Significance markers
            # Note: marginal (^) for p in [0.05, 0.10) due to small sample sizes
            # and inherent LLM sampling randomness
            sig = ""
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            elif p_value < 0.10:
                sig = "^"  # marginal significance

            print(
                f"  vs {other_name}: d={best_mean - other_mean:+.4f}, "
                f"p={p_value:.4f}{sig}, Cohen's d={cohens_d:.2f}"
            )


def generate_significance_latex(data: dict, metric: str = "final") -> str:
    """Generate LaTeX table with significance annotations."""
    meta = data["metadata"]
    results = data["results"]
    line_param = meta["line_param"]

    line_vals = get_sorted_values(results, 0)
    x_vals = get_sorted_values(results, 1)

    # For each x value, compute pairwise significance vs best
    significance_markers = {}  # (line, x) -> marker string

    for x in x_vals:
        configs = []
        for line in line_vals:
            if (line, x) in results:
                scores = results[(line, x)][metric].get("scores", [])
                if len(scores) >= 2:
                    configs.append((line, scores, np.mean(scores)))

        if len(configs) < 2:
            continue

        configs.sort(key=lambda c: c[2], reverse=True)
        _, best_scores, _ = configs[0]

        for other_name, other_scores, _ in configs[1:]:
            _, p_value = stats.ttest_ind(best_scores, other_scores, equal_var=False)

            # Note: marginal (†) for p in [0.05, 0.10) due to small sample sizes
            # and inherent LLM sampling randomness
            marker = ""
            if p_value < 0.001:
                marker = "$^{***}$"
            elif p_value < 0.01:
                marker = "$^{**}$"
            elif p_value < 0.05:
                marker = "$^{*}$"
            elif p_value < 0.10:
                marker = "$^{\\dagger}$"  # marginal significance

            significance_markers[(other_name, x)] = marker

    # Build table with significance markers
    lines = []
    lines.append(
        "% Significance: † p<0.10 (marginal), * p<0.05, ** p<0.01, *** p<0.001 (vs best in column)"
    )
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{"
        + f"{data['name'].replace('_', ' ').title()} with significance markers"
        + "}"
    )
    lines.append("\\label{tab:" + data["name"] + "_sig}")

    col_spec = "l" + "c" * len(x_vals)
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")

    header = f"\\textbf{{{line_param}}}"
    for x in x_vals:
        header += f" & \\textbf{{{x}}}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Find best per column
    best_per_col = {}
    for x in x_vals:
        best_score = -1
        best_line = None
        for line in line_vals:
            if (line, x) in results:
                score = results[(line, x)][metric]["mean"]
                if score > best_score:
                    best_score = score
                    best_line = line
        best_per_col[x] = best_line

    for line in line_vals:
        row = str(line).replace("_", "\\_")
        for x in x_vals:
            if (line, x) in results:
                r = results[(line, x)][metric]
                val_str = f"{r['mean']:.3f}"
                marker = significance_markers.get((line, x), "")

                if best_per_col.get(x) == line:
                    val_str = f"\\textbf{{{val_str}}}"

                row += f" & {val_str}{marker}"
            else:
                row += _LATEX_EMPTY_CELL
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse sweep results")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables")
    parser.add_argument("--sweep", type=str, help="Specific sweep to parse")
    parser.add_argument(
        "--metric",
        default="final",
        choices=["final", "value", "ref"],
        help="Metric to display",
    )
    parser.add_argument("--output", type=str, help="Output file for LaTeX tables")
    parser.add_argument(
        "--no-second-best",
        action="store_true",
        help="Don't mark second best with underline",
    )
    parser.add_argument(
        "--no-avg-rank",
        action="store_true",
        help="Don't show average rank column",
    )
    parser.add_argument(
        "--significance",
        action="store_true",
        help="Run statistical significance tests (paired t-tests)",
    )
    parser.add_argument(
        "--sig-latex",
        action="store_true",
        help="Output LaTeX tables with significance markers",
    )
    args = parser.parse_args()

    # Find sweeps to parse
    if args.sweep:
        sweep_dirs = [SWEEPS_DIR / args.sweep]
    else:
        sweep_dirs = sorted(SWEEPS_DIR.iterdir())

    # Parse all sweeps
    all_sweeps = []
    for sweep_dir in sweep_dirs:
        if sweep_dir.is_dir():
            data = parse_sweep(sweep_dir)
            if data:
                all_sweeps.append(data)

    if not all_sweeps:
        print("No sweep results found")
        return

    # Output
    if args.significance:
        for data in all_sweeps:
            run_significance_tests(data, args.metric)
    elif args.sig_latex:
        latex_output = []
        for data in all_sweeps:
            latex_output.append(generate_significance_latex(data, args.metric))
            latex_output.append("")
        output_text = "\n".join(latex_output)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_text)
            print(f"LaTeX tables with significance written to {args.output}")
        else:
            print(output_text)
    elif args.latex:
        latex_output = []
        for data in all_sweeps:
            latex_output.append(
                generate_latex_table(
                    data,
                    args.metric,
                    mark_second_best=not args.no_second_best,
                    show_avg_rank=not args.no_avg_rank,
                )
            )
            latex_output.append("")

        output_text = "\n".join(latex_output)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_text)
            print(f"LaTeX tables written to {args.output}")
        else:
            print(output_text)
    else:
        for data in all_sweeps:
            print_sweep_summary(data, args.metric)


if __name__ == "__main__":
    main()
