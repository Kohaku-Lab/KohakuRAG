"""Sweep: Ensemble Size vs Ignore Blank Mode

Aggregates raw inference runs with optional filtering of "is_blank" answers
before majority voting.

Line parameter (categorical): ignore_blank_mode
X-axis parameter (numerical): ensemble_size

Ignore blank modes:
- "off": Include all answers in voting (default behavior)
- "on": Filter out "is_blank" answers before voting (if any non-blank exists)

When ignore_blank="on":
- For answer voting: only vote among non-blank answers (if any exist)
- For ref_priority mode: only vote among non-blank refs (if any exist)
- Falls back to "is_blank" only when ALL answers/refs are blank

Usage:
    # First run inference (only once)
    python workflows/sweeps/ensemble_inference.py

    # Then run aggregation sweep
    python workflows/sweeps/ensemble_vs_ignore_blank.py
"""

import argparse
import ast
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

from kohakuengine import Config, Script

# ============================================================================
# SWEEP PARAMETERS
# ============================================================================

# Ensemble sizes to test (odd numbers)
ENSEMBLE_SIZES = [1, 3, 5, 7, 9, 11, 13, 15]

# Max combinations per ensemble size (evenly sampled if exceeded)
DEFAULT_MAX_COMBINATIONS = 32

# Line parameter: ignore blank modes
LINE_PARAM = "ignore_blank"
LINE_VALUES = ["off", "on"]

# X-axis parameter: ensemble size
X_PARAM = "ensemble_size"
X_VALUES = ENSEMBLE_SIZES

# Ref vote mode to use (answer_priority works well with ignore_blank)
REF_MODE = "union"

# ============================================================================
# PATHS
# ============================================================================

INPUT_DIR = Path("outputs/sweeps/ensemble/raw_runs")
OUTPUT_DIR = Path("outputs/sweeps/ensemble_vs_ignore_blank")
QUESTIONS = "data/train_QA.csv"


# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    """Load predictions CSV into dict keyed by question id."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader}


def majority_vote(values: list[str], ignore_blank: bool = False) -> str:
    """Return most common value (first occurrence breaks ties).

    If ignore_blank=True, filter out "is_blank" values before voting
    (only if there are non-blank values).
    """
    if not values:
        return "is_blank"

    if ignore_blank:
        non_blank = [v for v in values if v != "is_blank"]
        if non_blank:
            values = non_blank

    counter = Counter(values)
    max_count = counter.most_common(1)[0][1]
    tied = [v for v, c in counter.items() if c == max_count]

    if len(tied) == 1:
        return tied[0]

    # First occurrence tiebreak
    for v in values:
        if v in tied:
            return v
    return tied[0]


def parse_ref_ids(ref_str: str) -> set[str]:
    """Parse ref_id string into set of IDs."""
    if not ref_str or ref_str == "is_blank":
        return set()
    try:
        parsed = ast.literal_eval(ref_str)
        if isinstance(parsed, (list, tuple, set)):
            return {str(x).strip().lower() for x in parsed if x}
    except (ValueError, SyntaxError):
        pass
    # Fallback: split by comma
    return {x.strip().lower() for x in ref_str.strip("[]").split(",") if x.strip()}


def format_ref_ids(ref_set: set[str]) -> str:
    """Format set of ref_ids back to string."""
    if not ref_set:
        return "is_blank"
    return str(sorted(ref_set))


def aggregate_independent(
    rows: list[dict[str, str]], ignore_blank: bool = False
) -> tuple[str, str]:
    """Vote ref_id and answer_value independently."""
    answer_values = [r.get("answer_value", "is_blank") or "is_blank" for r in rows]
    ref_ids = [r.get("ref_id", "is_blank") or "is_blank" for r in rows]

    return majority_vote(answer_values, ignore_blank), majority_vote(
        ref_ids, ignore_blank
    )


def aggregate_ref_priority(
    rows: list[dict[str, str]], ignore_blank: bool = False
) -> tuple[str, str]:
    """First vote on ref_id, then vote answer_value among rows with winning ref."""
    if not rows:
        return "is_blank", "is_blank"

    # Vote on ref_id (with optional blank filtering)
    ref_ids = [r.get("ref_id", "is_blank") or "is_blank" for r in rows]
    best_ref = majority_vote(ref_ids, ignore_blank)

    # Filter to rows with winning ref
    matching_rows = [
        r for r in rows if (r.get("ref_id", "is_blank") or "is_blank") == best_ref
    ]

    # Vote on answer_value among matching rows
    answer_values = [
        r.get("answer_value", "is_blank") or "is_blank" for r in matching_rows
    ]
    best_val = majority_vote(answer_values, ignore_blank)

    return best_val, best_ref


def aggregate_answer_priority(
    rows: list[dict[str, str]], ignore_blank: bool = False
) -> tuple[str, str]:
    """First vote on answer_value, then vote ref_id among rows with winning answer."""
    if not rows:
        return "is_blank", "is_blank"

    # Vote on answer_value (with optional blank filtering)
    answer_values = [r.get("answer_value", "is_blank") or "is_blank" for r in rows]
    best_val = majority_vote(answer_values, ignore_blank)

    # Filter to rows with winning answer
    matching_rows = [
        r for r in rows if (r.get("answer_value", "is_blank") or "is_blank") == best_val
    ]

    # Vote on ref_id among matching rows
    ref_ids = [r.get("ref_id", "is_blank") or "is_blank" for r in matching_rows]
    best_ref = majority_vote(ref_ids, ignore_blank)

    return best_val, best_ref


def aggregate_union(
    rows: list[dict[str, str]], ignore_blank: bool = False
) -> tuple[str, str]:
    """Vote on answer_value, then union all ref_ids from matching rows."""
    if not rows:
        return "is_blank", "is_blank"

    # Vote on answer_value (with optional blank filtering)
    answer_values = [r.get("answer_value", "is_blank") or "is_blank" for r in rows]
    best_val = majority_vote(answer_values, ignore_blank)

    # Filter to rows with winning answer
    matching_rows = [
        r for r in rows if (r.get("answer_value", "is_blank") or "is_blank") == best_val
    ]

    # Union all ref_ids from matching rows
    combined_refs = set()
    for row in matching_rows:
        combined_refs.update(parse_ref_ids(row.get("ref_id", "")))

    return best_val, format_ref_ids(combined_refs)


def aggregate_intersection(
    rows: list[dict[str, str]], ignore_blank: bool = False
) -> tuple[str, str]:
    """Vote on answer_value, then intersect ref_ids from matching rows."""
    if not rows:
        return "is_blank", "is_blank"

    # Vote on answer_value (with optional blank filtering)
    answer_values = [r.get("answer_value", "is_blank") or "is_blank" for r in rows]
    best_val = majority_vote(answer_values, ignore_blank)

    # Filter to rows with winning answer
    matching_rows = [
        r for r in rows if (r.get("answer_value", "is_blank") or "is_blank") == best_val
    ]

    # Intersect ref_ids from matching rows
    ref_sets = [parse_ref_ids(row.get("ref_id", "")) for row in matching_rows]
    if not ref_sets:
        return best_val, "is_blank"

    combined_refs = ref_sets[0]
    for ref_set in ref_sets[1:]:
        combined_refs = combined_refs.intersection(ref_set)

    return best_val, format_ref_ids(combined_refs)


def aggregate_ensemble(
    pred_paths: list[Path],
    ignore_blank: bool,
    ref_mode: str = "union",
) -> dict[str, dict[str, str]]:
    """Aggregate multiple prediction files using specified mode."""
    all_preds = [load_predictions(p) for p in pred_paths]

    all_ids = set()
    for preds in all_preds:
        all_ids.update(preds.keys())

    result = {}
    for qid in all_ids:
        rows = [p.get(qid, {}) for p in all_preds if qid in p]

        if ref_mode == "independent":
            best_val, best_ref = aggregate_independent(rows, ignore_blank)
        elif ref_mode == "ref_priority":
            best_val, best_ref = aggregate_ref_priority(rows, ignore_blank)
        elif ref_mode == "answer_priority":
            best_val, best_ref = aggregate_answer_priority(rows, ignore_blank)
        elif ref_mode == "union":
            best_val, best_ref = aggregate_union(rows, ignore_blank)
        elif ref_mode == "intersection":
            best_val, best_ref = aggregate_intersection(rows, ignore_blank)
        else:
            raise ValueError(f"Unknown ref_mode: {ref_mode}")

        result[qid] = {
            "id": qid,
            "answer_value": best_val,
            "ref_id": best_ref,
        }

    return result


def save_predictions(preds: dict[str, dict[str, str]], path: Path) -> None:
    """Save predictions to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "answer_value", "ref_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid in sorted(preds.keys()):
            writer.writerow(preds[qid])


def get_total_runs() -> int:
    """Detect total runs from raw_runs directory."""
    if not INPUT_DIR.exists():
        return 0
    return len(list(INPUT_DIR.glob("run*_preds.csv")))


def sample_combinations_evenly(all_combos: list, max_count: int) -> list:
    """Sample combinations evenly spread across the list.

    If len(all_combos) <= max_count, return all.
    Otherwise, pick max_count items with maximum spread.
    """
    n = len(all_combos)
    if n <= max_count:
        return all_combos

    # Evenly spaced indices
    indices = [int(i * (n - 1) / (max_count - 1)) for i in range(max_count)]
    return [all_combos[i] for i in indices]


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep: Ensemble Size vs Ignore Blank Mode"
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=DEFAULT_MAX_COMBINATIONS,
        help=f"Max combinations per ensemble size (default: {DEFAULT_MAX_COMBINATIONS})",
    )
    parser.add_argument(
        "--ref-mode",
        type=str,
        default=REF_MODE,
        choices=[
            "independent",
            "ref_priority",
            "answer_priority",
            "union",
            "intersection",
        ],
        help=f"Ref aggregation mode to use (default: {REF_MODE})",
    )
    args = parser.parse_args()

    max_combinations = args.max_combinations
    ref_mode = args.ref_mode
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = get_total_runs()
    if total_runs == 0:
        print("Error: No raw runs found. Run ensemble_inference.py first.")
        exit(1)

    print("=" * 70)
    print("Sweep: Ensemble Size vs Ignore Blank Mode")
    print("=" * 70)
    print(f"Raw runs available: {total_runs}")
    print(f"Ensemble sizes: {ENSEMBLE_SIZES}")
    print(f"Ignore blank modes: {LINE_VALUES}")
    print(f"Ref aggregation mode: {ref_mode}")
    print(f"Max combinations per size: {max_combinations}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Check max ensemble size
    max_size = max(ENSEMBLE_SIZES)
    if max_size > total_runs:
        print(f"Warning: max ensemble size ({max_size}) > total runs ({total_runs})")
        ENSEMBLE_SIZES[:] = [s for s in ENSEMBLE_SIZES if s <= total_runs]
        print(f"Adjusted ensemble sizes: {ENSEMBLE_SIZES}")

    for ensemble_size in ENSEMBLE_SIZES:
        all_combos = list(itertools.combinations(range(total_runs), ensemble_size))
        combos = sample_combinations_evenly(all_combos, max_combinations)
        print(
            f"\nEnsemble size {ensemble_size}: C({total_runs},{ensemble_size}) = {len(all_combos)}, using {len(combos)}"
        )

        for mode in LINE_VALUES:
            ignore_blank = mode == "on"

            for combo_idx, combo in enumerate(combos):
                pred_paths = [INPUT_DIR / f"run{i}_preds.csv" for i in combo]

                missing = [p for p in pred_paths if not p.exists()]
                if missing:
                    print(f"  Skipping combo {combo_idx}: missing files")
                    continue

                aggregated = aggregate_ensemble(pred_paths, ignore_blank, ref_mode)

                out_filename = f"{LINE_PARAM}={mode}_{X_PARAM}={ensemble_size}_run{combo_idx}_preds.csv"
                out_path = OUTPUT_DIR / out_filename
                save_predictions(aggregated, out_path)

            print(f"  Created {len(combos)} files for mode={mode}")

    # Save metadata
    sweep_metadata = {
        "line_param": LINE_PARAM,
        "line_values": LINE_VALUES,
        "x_param": X_PARAM,
        "x_values": ENSEMBLE_SIZES,
        "num_runs": max_combinations,
        "questions": QUESTIONS,
        "ref_mode": ref_mode,
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as f:
        json.dump(sweep_metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("Aggregation complete!")
    print("=" * 70)
    print(f"Run 'python workflows/sweeps/sweep_plot.py {OUTPUT_DIR}' to plot results")
