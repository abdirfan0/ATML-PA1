from pathlib import Path
import sys

import pandas as pd
import yaml


REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2]
)

TASK2_ROOT = (
    REPOSITORY_ROOT / "task2"
)


def require_file(relative_path):
    path = REPOSITORY_ROOT / relative_path

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {relative_path}"
        )

    return path


required_files = [
    "task2/README.md",
    "task2/train.py",
    "task2/configs/base.yaml",
    "task2/configs/source_only.yaml",
    "task2/configs/dan.yaml",
    "task2/configs/dann.yaml",
    "task2/configs/cdan.yaml",
    "task2/notebooks/01_source_only.ipynb",
    "task2/notebooks/02_dan.ipynb",
    "task2/notebooks/03_dann.ipynb",
    "task2/notebooks/04_cdan.ipynb",
    "task2/notebooks/05_controlled_study.ipynb",
    "task2/notebooks/06_final_evaluation.ipynb",
    "task2/results/target_sketch_summary.csv",
    "task2/results/target_sketch_per_class.csv",
    "task2/results/domain_separability.csv",
    "task2/results/dan_controlled_study_summary.csv",
    "task2/figures/target_sketch_performance.png",
    "task2/figures/target_sketch_confusion_matrices.png",
    "task2/figures/domain_separability.png",
    "task2/figures/dan_controlled_study.png",
]

for required_file in required_files:
    require_file(required_file)


with require_file(
    "task2/configs/base.yaml"
).open(
    "r",
    encoding="utf-8",
) as config_file:
    base_config = yaml.safe_load(
        config_file
    )

assert (
    base_config["optimization"][
        "gradient_clip_norm"
    ]
    == 1.0
)

assert (
    base_config["selection"][
        "use_target_labels"
    ]
    is False
)


target_summary = pd.read_csv(
    require_file(
        "task2/results/"
        "target_sketch_summary.csv"
    )
)

expected_methods = {
    "source_only",
    "dan",
    "dann",
    "cdan",
}

assert set(
    target_summary["method"]
) == expected_methods

assert (
    target_summary[
        "number_of_target_samples"
    ]
    == 3929
).all()

for metric_column in [
    "target_accuracy",
    "target_macro_f1",
]:
    assert target_summary[
        metric_column
    ].between(
        0.0,
        1.0,
    ).all()


per_class_results = pd.read_csv(
    require_file(
        "task2/results/"
        "target_sketch_per_class.csv"
    )
)

assert set(
    per_class_results["method"]
) == expected_methods

assert len(per_class_results) == 28


controlled_summary = pd.read_csv(
    require_file(
        "task2/results/"
        "dan_controlled_study_summary.csv"
    )
)

assert set(
    controlled_summary["mmd_weight"]
) == {
    0.1,
    1.0,
    10.0,
}


domain_separability = pd.read_csv(
    require_file(
        "task2/results/"
        "domain_separability.csv"
    )
)

assert set(
    domain_separability["method"]
) == expected_methods

assert domain_separability[
    "domain_balanced_accuracy"
].between(
    0.0,
    1.0,
).all()


checkpoint_files = list(
    TASK2_ROOT.rglob("*.pt")
) + list(
    TASK2_ROOT.rglob("*.pth")
) + list(
    TASK2_ROOT.rglob("*.ckpt")
)

assert not checkpoint_files, (
    "Checkpoint files should not be "
    "stored inside the Git repository."
)


print("Task 2 validation passed.")
print(
    "Methods:",
    sorted(expected_methods),
)
print(
    "Target samples:",
    int(
        target_summary[
            "number_of_target_samples"
        ].iloc[0]
    ),
)
print(
    "Tracked checkpoints inside task2/:",
    len(checkpoint_files),
)

sys.exit(0)
