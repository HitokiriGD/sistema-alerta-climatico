import json

import pandas as pd
from sklearn.dummy import DummyClassifier

from scripts.train_ml_models import main as train_script_main
from src.ml.evaluate import build_metrics_summary
from src.ml.evaluate import evaluate_classifier
from src.ml.evaluate import select_best_model
from src.ml.models import get_candidate_models
from src.ml.train import prepare_features_and_target
from src.ml.train import split_dataset
from src.ml.train import train_and_evaluate_models


def training_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(25, 25, 80, 0, 5, 900, 1, 0, 1, "baixo"),
            _row(26, 26, 75, 0, 8, 901, 2, 6, 32, "baixo"),
            _row(33, 35, 28, 0, 10, 902, 3, 12, 60, "moderado"),
            _row(10, 10, 85, 0, 12, 903, 4, 18, 91, "moderado"),
            _row(36, 40, 18, 0, 16, 904, 5, 9, 121, "alto"),
            _row(20, 20, 70, 30, 10, 905, 6, 15, 152, "alto"),
            _row(41, 46, 10, 0, 30, 906, 7, 11, 182, "critico"),
            _row(22, 22, 75, 60, 80, 907, 8, 23, 213, "critico"),
        ]
    )


def _row(
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
        "event_type": "mantido_para_analise_futura",
    }


def test_prepare_features_and_target_selects_expected_columns() -> None:
    X, y, features = prepare_features_and_target(training_dataframe())

    assert features == [
        "temperature",
        "feels_like",
        "humidity",
        "precipitation",
        "wind_speed",
        "pressure",
        "month",
        "hour",
        "day_of_year",
    ]
    assert list(X.columns) == features
    assert y.name == "risk_level"


def test_prepare_features_and_target_removes_rows_without_target() -> None:
    dataframe = training_dataframe()
    dataframe.loc[0, "risk_level"] = None

    X, y, _ = prepare_features_and_target(dataframe)

    assert len(X) == 7
    assert len(y) == 7
    assert y.isna().sum() == 0


def test_prepare_features_and_target_ignores_missing_optional_feature() -> None:
    dataframe = training_dataframe().drop(columns=["pressure"])

    X, _, features = prepare_features_and_target(dataframe)

    assert "pressure" not in features
    assert "pressure" not in X.columns
    assert "temperature" in X.columns


def test_split_dataset_preserves_classes_with_stratify_when_possible() -> None:
    X, y, _ = prepare_features_and_target(training_dataframe())

    _, _, y_train, y_test = split_dataset(X, y, test_size=0.5, random_state=42)

    assert set(y_train) == {"baixo", "moderado", "alto", "critico"}
    assert set(y_test) == {"baixo", "moderado", "alto", "critico"}


def test_evaluate_classifier_returns_required_metrics() -> None:
    X, y, _ = prepare_features_and_target(training_dataframe())
    model = DummyClassifier(strategy="most_frequent").fit(X, y)

    metrics = evaluate_classifier(
        model,
        X,
        y,
        labels=["baixo", "moderado", "alto", "critico"],
    )

    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics


def test_build_metrics_summary_is_json_serializable() -> None:
    metrics = build_metrics_summary(
        ["baixo", "alto"],
        ["baixo", "baixo"],
        labels=["baixo", "alto"],
    )

    json.dumps(metrics)


def test_select_best_model_chooses_highest_metric() -> None:
    best = select_best_model(
        {
            "logistic_regression": {"f1_macro": 0.45},
            "random_forest": {"f1_macro": 0.72},
            "xgboost": {"f1_macro": 0.61},
        }
    )

    assert best == "random_forest"


def test_get_candidate_models_returns_expected_models() -> None:
    models = get_candidate_models()

    assert set(models) == {
        "baseline_most_frequent",
        "logistic_regression",
        "random_forest",
        "xgboost",
    }


def test_train_and_evaluate_models_saves_metadata_and_report(tmp_path) -> None:
    dataset_path = tmp_path / "training_dataset.csv"
    training_dataframe().to_csv(dataset_path, index=False)

    result = train_and_evaluate_models(
        dataset_path=dataset_path,
        output_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        test_size=0.5,
        candidate_models={
            "dummy_frequent": DummyClassifier(strategy="most_frequent"),
            "dummy_stratified": DummyClassifier(
                strategy="stratified",
                random_state=42,
            ),
        },
    )

    assert (tmp_path / "models" / "risk_level_model.joblib").exists()
    assert (tmp_path / "models" / "risk_level_model_metadata.json").exists()
    assert (tmp_path / "reports" / "risk_level_training_report.json").exists()
    assert (tmp_path / "reports" / "risk_level_confusion_matrix.csv").exists()
    assert result["selected_model_name"] in {"dummy_frequent", "dummy_stratified"}


def test_train_and_evaluate_models_preserves_labels_in_metadata(tmp_path) -> None:
    dataset_path = tmp_path / "training_dataset.csv"
    training_dataframe().to_csv(dataset_path, index=False)

    result = train_and_evaluate_models(
        dataset_path=dataset_path,
        output_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        test_size=0.5,
        candidate_models={"dummy": DummyClassifier(strategy="most_frequent")},
    )
    metadata = json.loads(
        (tmp_path / "models" / "risk_level_model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["labels"] == ["baixo", "moderado", "alto", "critico"]
    assert metadata["labels"] == ["baixo", "moderado", "alto", "critico"]


def test_train_script_returns_error_when_dataset_is_missing(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    missing_dataset = tmp_path / "missing.parquet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ml_models.py",
            "--dataset",
            str(missing_dataset),
            "--output-dir",
            str(tmp_path / "models"),
            "--report-dir",
            str(tmp_path / "reports"),
        ],
    )

    exit_code = train_script_main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Dataset rotulado nao encontrado" in output
