#!/usr/bin/env python3
"""
Aggregate multiple result CSVs using majority voting.

For each question, selects the most frequent answer_value as the final answer.
Reference IDs can be aggregated using different strategies.

Ref modes:
- independent: Vote ref_id and answer_value separately (simple majority)
- ref_priority: First vote on ref_id, then vote answer among rows with winning ref
- answer_priority: First vote on answer, then vote ref among rows with winning answer
- union: Vote on answer, then union all ref_ids from matching rows
- intersection: Vote on answer, then intersect ref_ids from matching rows

Options:
- ignore_blank: If True, filter out "is_blank" answers before voting
  (only if non-blank answers exist). Useful for ensemble voting where
  some runs may fail to produce an answer.

Usage (CLI):
    python scripts/wattbot_aggregate.py artifacts/results/*.csv -o aggregated.csv

Usage (KohakuEngine):
    kogine run scripts/wattbot_aggregate.py --config configs/aggregate_config.py
"""

import csv
import ast
import sys
import glob
from pathlib import Path
from collections import Counter
from typing import Literal

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

inputs: list[str] = []  # Input CSV files (required)
output = "artifacts/aggregated_preds.csv"
ref_mode: Literal[
    "independent", "ref_priority", "answer_priority", "union", "intersection"
] = "union"
tiebreak: Literal["blank", "first"] = "first"
ignore_blank: bool = (
    False  # Filter out "is_blank" before voting if non-blank answers exist
)

# Column names matching existing scripts
COLUMNS = [
    "id",
    "question",
    "answer",
    "answer_value",
    "answer_unit",
    "ref_id",
    "ref_url",
    "supporting_materials",
    "explanation",
]


def parse_ref_ids(ref_str: str) -> set[str]:
    """Parse ref_id string to a set of document IDs."""
    if not ref_str or ref_str == "is_blank":
        return set()

    # Try parsing as Python list
    try:
        parsed = ast.literal_eval(ref_str)
        if isinstance(parsed, list):
            return {str(x).strip() for x in parsed if x}
    except (ValueError, SyntaxError):
        pass

    # Try comma-separated
    return {x.strip() for x in ref_str.split(",") if x.strip()}


def format_ref_ids(ref_set: set[str]) -> str:
    """Format ref_id set back to CSV format."""
    if not ref_set:
        return "is_blank"
    return str(sorted(ref_set))


def normalize_value(value: str) -> str:
    """Normalize answer_value for comparison."""
    if not value:
        return "is_blank"
    value = str(value).strip()
    if not value:
        return "is_blank"
    return value


def load_csv(path: Path) -> dict[str, dict]:
    """Load a result CSV and return dict keyed by question id."""
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("id", "")
            if qid:
                rows[qid] = row
    return rows


def majority_vote(values: list[str], filter_blank: bool = False) -> str:
    """Return most common value (first occurrence breaks ties).

    If filter_blank=True, filter out "is_blank" values before voting
    (only if there are non-blank values).
    """
    if not values:
        return "is_blank"

    if filter_blank:
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


def aggregate_question_independent(
    answers: list[dict], filter_blank: bool = False
) -> tuple[str, str]:
    """Vote ref_id and answer_value separately."""
    answer_values = [normalize_value(r.get("answer_value", "")) for r in answers]
    ref_ids = [r.get("ref_id", "is_blank") or "is_blank" for r in answers]
    return majority_vote(answer_values, filter_blank), majority_vote(
        ref_ids, filter_blank
    )


def aggregate_question_ref_priority(
    answers: list[dict], filter_blank: bool = False
) -> tuple[str, str]:
    """First vote on ref_id, then vote answer among rows with winning ref."""
    if not answers:
        return "is_blank", "is_blank"

    ref_ids = [r.get("ref_id", "is_blank") or "is_blank" for r in answers]
    best_ref = majority_vote(ref_ids, filter_blank)

    matching_rows = [
        r for r in answers if (r.get("ref_id", "is_blank") or "is_blank") == best_ref
    ]
    answer_values = [normalize_value(r.get("answer_value", "")) for r in matching_rows]
    best_val = majority_vote(answer_values, filter_blank)

    return best_val, best_ref


def aggregate_question_answer_priority(
    answers: list[dict], filter_blank: bool = False
) -> tuple[str, str]:
    """First vote on answer, then vote ref among rows with winning answer."""
    if not answers:
        return "is_blank", "is_blank"

    answer_values = [normalize_value(r.get("answer_value", "")) for r in answers]
    best_val = majority_vote(answer_values, filter_blank)

    matching_rows = [
        r for r in answers if normalize_value(r.get("answer_value", "")) == best_val
    ]
    ref_ids = [r.get("ref_id", "is_blank") or "is_blank" for r in matching_rows]
    best_ref = majority_vote(ref_ids, filter_blank)

    return best_val, best_ref


def aggregate_question_union(
    answers: list[dict], filter_blank: bool = False
) -> tuple[str, str]:
    """Vote on answer, then union all ref_ids from matching rows."""
    if not answers:
        return "is_blank", "is_blank"

    answer_values = [normalize_value(r.get("answer_value", "")) for r in answers]
    best_val = majority_vote(answer_values, filter_blank)

    matching_rows = [
        r for r in answers if normalize_value(r.get("answer_value", "")) == best_val
    ]

    combined_refs = set()
    for row in matching_rows:
        combined_refs.update(parse_ref_ids(row.get("ref_id", "")))

    return best_val, format_ref_ids(combined_refs)


def aggregate_question_intersection(
    answers: list[dict], filter_blank: bool = False
) -> tuple[str, str]:
    """Vote on answer, then intersect ref_ids from matching rows."""
    if not answers:
        return "is_blank", "is_blank"

    answer_values = [normalize_value(r.get("answer_value", "")) for r in answers]
    best_val = majority_vote(answer_values, filter_blank)

    matching_rows = [
        r for r in answers if normalize_value(r.get("answer_value", "")) == best_val
    ]

    ref_sets = [parse_ref_ids(row.get("ref_id", "")) for row in matching_rows]
    if not ref_sets:
        return best_val, "is_blank"

    combined_refs = ref_sets[0]
    for ref_set in ref_sets[1:]:
        combined_refs = combined_refs.intersection(ref_set)

    return best_val, format_ref_ids(combined_refs)


_AGGREGATORS_BY_MODE = {
    "independent": aggregate_question_independent,
    "ref_priority": aggregate_question_ref_priority,
    "answer_priority": aggregate_question_answer_priority,
    "union": aggregate_question_union,
    "intersection": aggregate_question_intersection,
}


def aggregate_results(
    csv_paths: list[Path],
    ref_mode: Literal[
        "independent", "ref_priority", "answer_priority", "union", "intersection"
    ] = "union",
    tiebreak_mode: Literal["blank", "first"] = "first",
    filter_blank: bool = False,
) -> list[dict]:
    """
    Aggregate multiple result CSVs using majority voting.

    Args:
        csv_paths: List of paths to result CSVs
        ref_mode: How to aggregate - "independent", "ref_priority", "answer_priority", "union", "intersection"
        tiebreak_mode: What to do when all answers differ - "blank" or "first"
        filter_blank: If True, filter out "is_blank" answers before voting (if non-blank exist)

    Returns:
        List of aggregated result rows
    """
    # Load all CSVs
    all_data = []
    for path in csv_paths:
        data = load_csv(path)
        all_data.append(data)
        print(f"Loaded {len(data)} questions from {path.name}")

    if not all_data:
        return []

    # Get all question IDs
    all_qids = set()
    for data in all_data:
        all_qids.update(data.keys())

    results = []

    for qid in sorted(all_qids):
        # Collect all answers for this question
        answers = []
        for data in all_data:
            if qid in data:
                answers.append(data[qid])

        if not answers:
            continue

        # Check if all answers are different (for tiebreak)
        answer_values = [normalize_value(r.get("answer_value", "")) for r in answers]
        if len(set(answer_values)) == len(answers) and len(answers) > 1:
            if tiebreak_mode == "blank":
                result_row = answers[0].copy()
                result_row["answer"] = "is_blank"
                result_row["answer_value"] = "is_blank"
                result_row["ref_id"] = "is_blank"
                result_row["ref_url"] = "is_blank"
                result_row["supporting_materials"] = "is_blank"
                result_row["explanation"] = "is_blank"
                results.append(result_row)
                continue

        # Aggregate based on mode
        aggregator = _AGGREGATORS_BY_MODE.get(ref_mode)
        if aggregator is None:
            raise ValueError(f"Unknown ref_mode: {ref_mode}")
        best_val, best_ref = aggregator(answers, filter_blank)

        # Find a matching row to use as base
        matching_rows = [
            r for r in answers if normalize_value(r.get("answer_value", "")) == best_val
        ]
        result_row = (matching_rows[0] if matching_rows else answers[0]).copy()

        result_row["answer_value"] = best_val
        result_row["ref_id"] = best_ref

        results.append(result_row)

    return results


def main():
    if not inputs:
        raise ValueError("inputs must be set in config")

    # Expand glob patterns (needed for Windows)
    expanded_inputs = []
    for pattern in inputs:
        matches = glob.glob(str(pattern))
        if matches:
            expanded_inputs.extend(Path(m) for m in matches)
        else:
            expanded_inputs.append(Path(pattern))

    # Validate inputs exist
    for path in expanded_inputs:
        if not path.exists():
            print(f"Error: Input file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if not expanded_inputs:
        print("Error: No input files found", file=sys.stderr)
        sys.exit(1)

    print(f"Aggregating {len(expanded_inputs)} result files...")
    print(f"  ref_id mode: {ref_mode}")
    print(f"  tiebreak mode: {tiebreak}")
    print(f"  ignore_blank: {ignore_blank}")
    print()

    # Aggregate results
    results = aggregate_results(
        expanded_inputs,
        ref_mode=ref_mode,
        tiebreak_mode=tiebreak,
        filter_blank=ignore_blank,
    )

    # Write output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in results:
            # Ensure all columns exist
            out_row = {col: row.get(col, "is_blank") for col in COLUMNS}
            writer.writerow(out_row)

    print(f"\nAggregated {len(results)} questions to {output_path}")


if __name__ == "__main__":
    main()
