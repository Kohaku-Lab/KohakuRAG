"""Ensemble runner: parallel model comparison + aggregation.

Runs multiple models in parallel, then aggregates results using voting.

Usage:
    python configs/workflows/ensemble_runner.py
"""

from kohakuengine import Config, Script, Flow

# Shared settings
DB = "artifacts/wattbot_with_images.db"
TABLE_PREFIX = "wattbot_img"
QUESTIONS = "data/train_QA.csv"
METADATA = "data/metadata.csv"

_DEFAULT_MODEL = "openai/GPT-5-mini"

# Models to run in parallel
MODELS = [
    {
        "model": _DEFAULT_MODEL,
        "output": "outputs/train-result-gpt-mini/single_preds1.csv",
    },
    {
        "model": _DEFAULT_MODEL,
        "output": "outputs/train-result-gpt-mini/single_preds2.csv",
    },
    {
        "model": _DEFAULT_MODEL,
        "output": "outputs/train-result-gpt-mini/single_preds3.csv",
    },
    {
        "model": _DEFAULT_MODEL,
        "output": "outputs/train-result-gpt-mini/single_preds4.csv",
    },
    {
        "model": _DEFAULT_MODEL,
        "output": "outputs/train-result-gpt-mini/single_preds5.csv",
    },
]

# Aggregation settings
AGGREGATED_OUTPUT = "outputs/train-result-gpt-mini/ensemble_preds.csv"
REF_MODE = "union"
TIEBREAK = "first"


def create_answer_config(model_name: str, output_path: str) -> Config:
    """Create answer config for a specific model."""
    return Config(
        globals_dict={
            "db": DB,
            "table_prefix": TABLE_PREFIX,
            "questions": QUESTIONS,
            "output": output_path,
            "metadata": METADATA,
            "model": model_name,
            "planner_model": None,
            "top_k": 16,
            "planner_max_queries": 3,
            "max_retries": 3,
            "max_concurrent": -1,
            "with_images": True,
            "top_k_images": 4,
            "single_run_debug": False,
            "question_id": None,
        }
    )


if __name__ == "__main__":
    import os

    # Ensure output directory exists
    os.makedirs("outputs/train-result-gpt-mini", exist_ok=True)

    # Create answer scripts for each model
    answer_scripts = [
        Script(
            "scripts/wattbot_answer.py",
            config=create_answer_config(cfg["model"], cfg["output"]),
        )
        for cfg in MODELS
    ]

    # Run answer scripts in parallel
    print("=" * 60)
    print(f"Running {len(MODELS)} models in parallel...")
    print("=" * 60)

    answer_flow = Flow(answer_scripts, mode="parallel", max_workers=len(MODELS))
    answer_flow.run()

    # Aggregate results
    print("\n" + "=" * 60)
    print("Aggregating results...")
    print("=" * 60)

    aggregate_config = Config(
        globals_dict={
            "inputs": [cfg["output"] for cfg in MODELS],
            "output": AGGREGATED_OUTPUT,
            "ref_mode": REF_MODE,
            "tiebreak": TIEBREAK,
        }
    )

    aggregate_script = Script("scripts/wattbot_aggregate.py", config=aggregate_config)
    aggregate_script.run()

    # Validate aggregated results
    print("\n" + "=" * 60)
    print("Validating ensemble results...")
    print("=" * 60)

    validate_config = Config(
        globals_dict={
            "truth": "data/train_QA.csv",
            "pred": AGGREGATED_OUTPUT,
            "show_errors": 0,
            "verbose": True,
        }
    )

    validate_script = Script("scripts/wattbot_validate.py", config=validate_config)
    validate_script.run()

    print("\n" + "=" * 60)
    print("Ensemble complete!")
    print(f"Aggregated results: {AGGREGATED_OUTPUT}")
    print("=" * 60)
