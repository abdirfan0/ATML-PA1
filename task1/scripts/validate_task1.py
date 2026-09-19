
import sys

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(REPO_ROOT),
)

from task1.analysis.evaluate_bias import (
    build_cue_conflict_summary,
)


def report_check(
    condition,
    description,
    failures,
):
    if condition:
        print(f"[PASS] {description}")
    else:
        print(f"[FAIL] {description}")
        failures.append(description)


def main():
    failures = []

    task1_root = REPO_ROOT / "task1"

    required_python_files = [
        "data/transforms.py",
        "data/datasets.py",
        "data/make_subset.py",
        "data/cue_conflicts.py",
        "data/make_cue_conflicts.py",
        "models/backbones.py",
        "models/linear_probe.py",
        "models/feature_extraction.py",
        "analysis/evaluate_bias.py",
        "analysis/feature_similarity.py",
        "analysis/representation.py",
    ]

    for relative_path in required_python_files:
        report_check(
            (
                task1_root
                / relative_path
            ).exists(),
            f"Python module exists: "
            f"{relative_path}",
            failures,
        )

    expected_notebooks = {
        "01_clean_baseline.ipynb",
        "02_color_translation_patch.ipynb",
        "03_cue_conflicts.ipynb",
        "04_representation_analysis.ipynb",
    }

    notebook_directory = (
        task1_root / "notebooks"
    )

    notebook_names = {
        path.name
        for path in notebook_directory.glob(
            "*.ipynb"
        )
    }

    report_check(
        expected_notebooks.issubset(
            notebook_names
        ),
        "All four experiment notebooks exist",
        failures,
    )

    expected_split_files = {
        "stl10_train_val_seed6304.json",
        "stl10_test_subset_500_seed6304.json",
        "representation_visualization_200_seed6304.json",
        "patch_permutations_4x4_seed6304.json",
    }

    split_directory = (
        task1_root / "splits"
    )

    split_names = {
        path.name
        for path in split_directory.glob(
            "*.json"
        )
    }

    report_check(
        expected_split_files.issubset(
            split_names
        ),
        "All fixed split files exist",
        failures,
    )

    result_directory = (
        task1_root / "results"
    )

    result_files = list(
        result_directory.glob("*.csv")
    )

    report_check(
        len(result_files) == 14,
        "Exactly 14 result CSV files exist",
        failures,
    )

    figure_directory = (
        task1_root / "figures"
    )

    figure_files = list(
        figure_directory.glob("*.png")
    )

    report_check(
        len(figure_files) == 10,
        "Exactly 10 final figures exist",
        failures,
    )

    candidate_directory = (
        task1_root
        / "data/cue_conflicts/candidates"
    )

    report_check(
        not candidate_directory.exists(),
        "Generated candidate images are "
        "excluded from the repository",
        failures,
    )

    cue_metadata_directory = (
        task1_root
        / "data/cue_conflicts"
    )

    candidate_metadata = pd.read_csv(
        cue_metadata_directory
        / "candidate_metadata.csv"
    )

    selected_metadata = pd.read_csv(
        cue_metadata_directory
        / "selected_conflicts_200.csv"
    )

    report_check(
        len(candidate_metadata) == 300,
        "Candidate metadata contains "
        "300 images",
        failures,
    )

    report_check(
        len(selected_metadata) == 200,
        "Selected metadata contains "
        "200 images",
        failures,
    )

    direction_counts = (
        selected_metadata["direction"]
        .value_counts()
    )

    report_check(
        (
            len(direction_counts) == 10
            and direction_counts.eq(20).all()
        ),
        "Selected cue conflicts contain "
        "20 images in each of 10 directions",
        failures,
    )

    report_check(
        (
            selected_metadata["decision"]
            .astype(str)
            .str.lower()
            .eq("accepted")
            .all()
        ),
        "Every selected cue conflict "
        "was manually accepted",
        failures,
    )

    prediction_table = pd.read_csv(
        result_directory
        / "cue_conflict_predictions.csv"
    )

    saved_summary = pd.read_csv(
        result_directory
        / "cue_conflict_summary.csv"
    )

    calculated_summary = (
        build_cue_conflict_summary(
            prediction_table
        )
    )

    report_check(
        len(prediction_table) == 800,
        "Cue-conflict predictions contain "
        "800 rows",
        failures,
    )

    group_sizes = (
        prediction_table.groupby(
            ["model", "evaluation"]
        )
        .size()
    )

    report_check(
        (
            len(group_sizes) == 4
            and group_sizes.eq(200).all()
        ),
        "Each of the four evaluations "
        "contains 200 predictions",
        failures,
    )

    comparison = calculated_summary.merge(
        saved_summary,
        on=["model", "evaluation"],
        suffixes=(
            "_calculated",
            "_saved",
        ),
    )

    count_columns = [
        "n_shape",
        "n_texture",
        "n_other",
        "n_total",
    ]

    count_values_match = True

    for column in count_columns:
        count_values_match = (
            count_values_match
            and np.array_equal(
                comparison[
                    f"{column}_calculated"
                ],
                comparison[
                    f"{column}_saved"
                ],
            )
        )

    report_check(
        count_values_match,
        "Saved cue-conflict counts match "
        "recalculated counts",
        failures,
    )

    metric_columns = [
        "shape_bias_percent",
        "coverage_percent",
    ]

    metric_values_match = True

    for column in metric_columns:
        metric_values_match = (
            metric_values_match
            and np.allclose(
                comparison[
                    f"{column}_calculated"
                ],
                comparison[
                    f"{column}_saved"
                ],
            )
        )

    report_check(
        metric_values_match,
        "Saved cue-conflict metrics match "
        "recalculated metrics",
        failures,
    )

    print()

    if failures:
        print(
            "Task 1 validation failed with "
            f"{len(failures)} problem(s)."
        )

        raise SystemExit(1)

    print(
        "Task 1 repository validation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
