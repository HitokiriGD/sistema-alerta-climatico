import json

import pandas as pd

from scripts.evaluate_ml_models import main as evaluate_script_main
from src.ml.evaluate import build_robust_evaluation_output_paths
from src.ml.evaluate import build_robust_evaluation_report
from src.ml.evaluate import build_metrics_summary
from src.ml.evaluate import extract_metrics_by_class
from src.ml.evaluate import save_confusion_matrix_csv
from src.ml.evaluate import select_best_model
from src.ml.models import get_candidate_models
from src.ml.train import split_dataset_temporal


LABELS = ["baixo", "moderado", "alto", "critico"]


def robust_dataframe() -> pd.DataFrame:
    rows = []
    for year in [2023, 2024, 2025, 2026]:
        rows.extend(
            [
                _row(year, 24, 24, 80, 0, 5, 900, 1, 0, 1, "baixo"),
                _row(year, 28, 29, 65, 0, 8, 901, 2, 6, 32, "moderado"),
                _row(year, 36, 40, 18, 0, 12, 902, 3, 12, 60, "alto"),
                _row(year, 41, 46, 10, 50, 80, 903, 4, 18, 91, "critico"),
            ]
        )
    return pd.DataFrame(rows)


def _row(
    year,
    temperature,
    feels_like,
    humidity,
    precipitation,
    wind_speed,
    pressure,
    month,
    hour,
    day_of_year,
    risk_level,
) -> dict[str, object]:
    return {
        "year": year,
        "temperature": temperature,
        "feels_like": feels_like,
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "month": month,
        "hour": hour,
        "day_of_year": day_of_year,
        "risk_level": risk_level,
    }


def test_split_dataset_temporal_separates_train_and_test() -> None:
    X_train, X_test, y_train, y_test, split_info = split_dataset_temporal(
        robust_dataframe(),
        train_end_year=2024,
        test_start_year=2025,
    )

    assert len(X_train) == 8
    assert len(X_test) == 8
    assert set(y_train) == set(LABELS)
    assert set(y_test) == set(LABELS)
    assert split_info["train_period"] == {"start_year": 2023, "end_year": 2024}
    assert split_info["test_period"] == {"start_year": 2025, "end_year": 2026}


def test_split_dataset_temporal_raises_friendly_error_without_test_records() -> None:
    dataframe = robust_dataframe()
    dataframe = dataframe[dataframe["year"] <= 2024]

    try:
        split_dataset_temporal(
            dataframe,
            train_end_year=2024,
            test_start_year=2025,
        )
    except ValueError as error:
        assert "Split temporal sem registros suficientes" in str(error)
    else:
        raise AssertionError("split_dataset_temporal deveria falhar")


def test_baseline_most_frequent_is_included() -> None:
    models = get_candidate_models()

    assert "baseline_most_frequent" in models


def test_metrics_by_class_are_extracted() -> None:
    metrics = build_metrics_summary(
        ["alto", "critico", "critico"],
        ["alto", "alto", "critico"],
        labels=["alto", "critico"],
    )
    by_class = extract_metrics_by_class(metrics["classification_report"], LABELS)

    assert by_class["alto"]["precision"] == 0.5
    assert by_class["alto"]["recall"] == 1.0
    assert by_class["critico"]["f1-score"] > 0
    assert by_class["baixo"]["support"] == 0


def test_robust_report_contains_distributions_and_note() -> None:
    baseline_metrics = build_metrics_summary(["baixo", "alto"], ["baixo", "baixo"], LABELS)
    model_metrics = build_metrics_summary(["baixo", "alto"], ["baixo", "alto"], LABELS)

    report = build_robust_evaluation_report(
        split_type="temporal",
        train_period={"start_year": 2023, "end_year": 2024},
        test_period={"start_year": 2025, "end_year": 2026},
        total_records=4,
        train_records=2,
        test_records=2,
        train_class_distribution={"baixo": 1, "alto": 1},
        test_class_distribution={"baixo": 1, "alto": 1},
        features_used=["temperature", "humidity"],
        labels=LABELS,
        metrics_by_model={
            "baseline_most_frequent": baseline_metrics,
            "random_forest": model_metrics,
        },
        best_model_name="random_forest",
        selection_metric="f1_macro",
    )

    assert report["train_class_distribution"] == {"baixo": 1, "alto": 1}
    assert report["test_class_distribution"] == {"baixo": 1, "alto": 1}
    assert "RiskClassifier" in report["methodological_note"]
    assert report["metrics_by_class"]["random_forest"]["alto"]["support"] == 1
    assert "alto" in report["priority_class_metrics"]


def test_evaluate_ml_models_script_runs_with_small_fake_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "ml_dataset.csv"
    report_dir = tmp_path / "reports"
    robust_dataframe().to_csv(dataset_path, index=False)

    exit_code = evaluate_script_main(
        [
            "--dataset",
            str(dataset_path),
            "--report-dir",
            str(report_dir),
            "--split",
            "temporal",
            "--train-end-year",
            "2024",
            "--test-start-year",
            "2025",
        ]
    )

    report_path = (
        report_dir / "risk_level_robust_evaluation_temporal_report.json"
    )
    matrix_path = (
        report_dir / "risk_level_robust_confusion_matrix_temporal.csv"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report_path.exists()
    assert matrix_path.exists()
    assert report["split_type"] == "temporal"
    assert "baseline_most_frequent" in report["metrics_by_model"]


def test_evaluate_ml_models_script_runs_random_split(tmp_path) -> None:
    dataset_path = tmp_path / "ml_dataset.csv"
    report_dir = tmp_path / "reports"
    robust_dataframe().to_csv(dataset_path, index=False)

    exit_code = evaluate_script_main(
        [
            "--dataset",
            str(dataset_path),
            "--report-dir",
            str(report_dir),
            "--split",
            "random",
            "--test-size",
            "0.5",
        ]
    )

    report = json.loads(
        (
            report_dir / "risk_level_robust_evaluation_random_report.json"
        ).read_text(encoding="utf-8")
    )
    matrix_path = report_dir / "risk_level_robust_confusion_matrix_random.csv"

    assert exit_code == 0
    assert matrix_path.exists()
    assert report["split_type"] == "random"
    assert report["train_records"] == 8
    assert report["test_records"] == 8


def test_robust_evaluation_output_paths_include_split_type(tmp_path) -> None:
    temporal_report, temporal_matrix = build_robust_evaluation_output_paths(
        tmp_path,
        "temporal",
    )
    random_report, random_matrix = build_robust_evaluation_output_paths(
        tmp_path,
        "random",
    )

    assert temporal_report.name == (
        "risk_level_robust_evaluation_temporal_report.json"
    )
    assert temporal_matrix.name == (
        "risk_level_robust_confusion_matrix_temporal.csv"
    )
    assert random_report.name == (
        "risk_level_robust_evaluation_random_report.json"
    )
    assert random_matrix.name == (
        "risk_level_robust_confusion_matrix_random.csv"
    )
    assert temporal_report != random_report
    assert temporal_matrix != random_matrix


def test_missing_rare_class_in_test_does_not_break_metrics() -> None:
    metrics = build_metrics_summary(
        ["baixo", "alto", "alto"],
        ["baixo", "baixo", "alto"],
        labels=LABELS,
    )

    assert metrics["metrics_by_class"]["critico"]["support"] == 0
    assert len(metrics["confusion_matrix"]) == len(LABELS)


def test_confusion_matrix_csv_preserves_expected_labels(tmp_path) -> None:
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    csv_path = tmp_path / "confusion_matrix.csv"

    save_confusion_matrix_csv(csv_path, matrix, LABELS)
    saved = pd.read_csv(csv_path, index_col="actual")

    assert list(saved.index) == LABELS
    assert list(saved.columns) == LABELS


def test_select_best_model_still_chooses_highest_metric() -> None:
    best_model = select_best_model(
        {
            "baseline_most_frequent": {"f1_macro": 0.20},
            "logistic_regression": {"f1_macro": 0.50},
            "random_forest": {"f1_macro": 0.70},
        },
        metric="f1_macro",
    )

    assert best_model == "random_forest"
