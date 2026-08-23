from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.evaluate import evaluate_classifier
from src.ml.evaluate import select_best_model
from src.ml.models import create_model
from src.ml.models import get_candidate_models


ML_FEATURE_COLUMNS = [
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
RISK_LEVEL_LABELS = ["baixo", "moderado", "alto", "critico"]


def load_training_dataset(path: str | Path) -> pd.DataFrame:
    """Carrega dataset rotulado em Parquet ou CSV."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_path}")

    if dataset_path.suffix.lower() == ".csv":
        return pd.read_csv(dataset_path)
    return pd.read_parquet(dataset_path)


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = "risk_level",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Seleciona features numericas disponiveis e o alvo supervisionado."""
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo nao encontrada: {target_column}")

    available_features = [
        column for column in ML_FEATURE_COLUMNS if column in df.columns
    ]
    if not available_features:
        raise ValueError("Nenhuma feature numerica esperada foi encontrada.")

    data = df[available_features + [target_column]].copy()
    data[target_column] = data[target_column].astype("string")
    data = data.dropna(subset=[target_column])
    data = data[data[target_column].str.strip() != ""]

    for column in available_features:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=available_features, how="all")
    for column in available_features:
        median = data[column].median()
        fill_value = 0.0 if pd.isna(median) else float(median)
        data[column] = data[column].fillna(fill_value)

    X = data[available_features].reset_index(drop=True)
    y = data[target_column].astype(str).reset_index(drop=True)
    return X, y, available_features


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide treino/teste com estratificacao quando a amostra permite."""
    stratify = y if _can_use_stratify(y, test_size) else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def train_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Treina os modelos candidatos para classificacao de risk_level."""
    models = candidate_models or get_candidate_models()
    return {
        model_name: model.fit(X_train, y_train)
        for model_name, model in models.items()
    }


def train_and_evaluate_models(
    dataset_path: str | Path,
    output_dir: str | Path = "data/models",
    report_dir: str | Path = "data/reports",
    test_size: float = 0.2,
    random_state: int = 42,
    metric: str = "f1_macro",
    candidate_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Treina candidatos, seleciona o melhor e salva modelo e relatorios."""
    dataset = load_training_dataset(dataset_path)
    X, y, feature_columns = prepare_features_and_target(dataset)
    labels = _ordered_labels(y)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    trained_models = train_candidate_models(X_train, y_train, candidate_models)
    metrics_by_model = {
        model_name: evaluate_classifier(model, X_test, y_test, labels)
        for model_name, model in trained_models.items()
    }
    selected_model_name = select_best_model(metrics_by_model, metric)
    selected_model = trained_models[selected_model_name]

    output_path = Path(output_dir)
    report_path = Path(report_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / "risk_level_model.joblib"
    metadata_file = output_path / "risk_level_model_metadata.json"
    report_file = report_path / "risk_level_training_report.json"
    confusion_matrix_file = report_path / "risk_level_confusion_matrix.csv"

    metadata = {
        "target": "risk_level",
        "selected_model_name": selected_model_name,
        "metric_used": metric,
        "features": feature_columns,
        "labels": labels,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "total_records": int(len(X)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "metrics_by_model": metrics_by_model,
    }
    report = {
        "metadata": metadata,
        "class_distribution": _class_distribution(y),
        "metrics_by_model": metrics_by_model,
    }

    joblib.dump(selected_model, model_file)
    _write_json(metadata_file, metadata)
    _write_json(report_file, report)
    _write_confusion_matrix_csv(
        confusion_matrix_file,
        metrics_by_model[selected_model_name]["confusion_matrix"],
        labels,
    )

    return {
        "model_path": str(model_file),
        "metadata_path": str(metadata_file),
        "report_path": str(report_file),
        "confusion_matrix_path": str(confusion_matrix_file),
        "selected_model_name": selected_model_name,
        "metric_used": metric,
        "features": feature_columns,
        "labels": labels,
        "class_distribution": _class_distribution(y),
        "metrics_by_model": metrics_by_model,
        "total_records": int(len(X)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
    }


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = "logistic_regression",
) -> Any:
    """Treina um modelo supervisionado simples."""
    model = create_model(model_name)
    return model.fit(features, target)


def _can_use_stratify(y: pd.Series, test_size: float) -> bool:
    if y.nunique() < 2:
        return False

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        return False

    test_records = int(round(len(y) * test_size))
    train_records = len(y) - test_records
    class_count = int(y.nunique())
    return test_records >= class_count and train_records >= class_count


def _ordered_labels(y: pd.Series) -> list[str]:
    observed = [str(label) for label in y.dropna().unique()]
    ordered = [label for label in RISK_LEVEL_LABELS if label in observed]
    ordered.extend(sorted(set(observed) - set(ordered)))
    return ordered


def _class_distribution(y: pd.Series) -> dict[str, int]:
    return {str(label): int(count) for label, count in y.value_counts().items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_confusion_matrix_csv(
    path: Path,
    matrix: list[list[int]],
    labels: list[str],
) -> None:
    dataframe = pd.DataFrame(matrix, index=labels, columns=labels)
    dataframe.index.name = "actual"
    dataframe.to_csv(path)
