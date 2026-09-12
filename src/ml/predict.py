from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import json

import joblib
import pandas as pd

from src.ml.train import ML_FEATURE_COLUMNS
from src.ml.train import RISK_LEVEL_LABELS


DEFAULT_MODEL_PATH = Path("data/models/risk_level_model.joblib")
DEFAULT_METADATA_PATH = Path("data/models/risk_level_model_metadata.json")
METHODOLOGICAL_NOTE = (
    "O modelo foi treinado com rótulos derivados do classificador por regras. "
    "A previsão representa reprodução aprendida dessa classificação técnica, "
    "não validação contra eventos reais oficiais."
)
MODEL_NOT_FOUND_MESSAGE = (
    "Modelo ML ainda nao encontrado localmente. Gere o dataset com "
    "python scripts/build_ml_dataset.py e treine o modelo com "
    "python scripts/train_ml_models.py. Em uma etapa futura, esses artefatos "
    "tambem podem ser publicados em uma GitHub Release."
)


def model_files_available(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> bool:
    """Verifica se modelo e metadados existem localmente."""
    return Path(model_path).exists() and Path(metadata_path).exists()


def load_model_bundle(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    loader: Callable[[str | Path], Any] = joblib.load,
) -> dict[str, Any]:
    """Carrega modelo treinado e metadados, sem quebrar quando ausentes."""
    if not model_files_available(model_path, metadata_path):
        return _unavailable_result(MODEL_NOT_FOUND_MESSAGE)

    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        model = loader(model_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _unavailable_result(
            f"Nao foi possivel carregar o modelo ML local: {error}"
        )

    return {
        "available": True,
        "model": model,
        "metadata": metadata,
        "model_name": metadata.get("selected_model_name"),
        "selection_metric": metadata.get("metric_used"),
        "features_used": list(metadata.get("features") or ML_FEATURE_COLUMNS),
        "labels": list(metadata.get("labels") or RISK_LEVEL_LABELS),
        "missing_features": [],
        "methodological_note": METHODOLOGICAL_NOTE,
        "error_message": "",
    }


def build_prediction_features(
    weather_data: dict[str, object],
    feature_names: list[str],
    reference_datetime: datetime | None = None,
) -> dict[str, Any]:
    """Monta uma linha de features para predicao de risk_level."""
    values: dict[str, float] = {}
    missing_features: list[str] = []
    observations: list[str] = []
    weather_datetime = _resolve_weather_datetime(weather_data, reference_datetime)

    for feature_name in feature_names:
        value = _feature_value(
            feature_name,
            weather_data,
            weather_datetime,
            observations,
        )
        numeric_value = _to_float(value)
        if numeric_value is None:
            missing_features.append(feature_name)
            continue
        values[feature_name] = numeric_value

    if missing_features:
        return {
            "available": False,
            "features": None,
            "features_used": feature_names,
            "missing_features": missing_features,
            "observations": observations,
            "error_message": (
                "Nao foi possivel montar todas as features obrigatorias para "
                "a previsao ML: "
                + ", ".join(missing_features)
                + "."
            ),
        }

    return {
        "available": True,
        "features": pd.DataFrame([values], columns=feature_names),
        "features_used": feature_names,
        "missing_features": [],
        "observations": observations,
        "error_message": "",
    }


def predict_risk_level(
    weather_data: dict[str, object],
    model_bundle: dict[str, Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    reference_datetime: datetime | None = None,
) -> dict[str, Any]:
    """Executa predicao ML de risk_level com retorno seguro para o dashboard."""
    bundle = model_bundle or load_model_bundle(model_path, metadata_path)
    base_result = {
        "available": bool(bundle.get("available")),
        "prediction": None,
        "model_name": bundle.get("model_name"),
        "selection_metric": bundle.get("selection_metric"),
        "features_used": list(bundle.get("features_used") or ML_FEATURE_COLUMNS),
        "labels": list(bundle.get("labels") or RISK_LEVEL_LABELS),
        "missing_features": list(bundle.get("missing_features") or []),
        "observations": [],
        "probabilities": {},
        "methodological_note": METHODOLOGICAL_NOTE,
        "error_message": str(bundle.get("error_message") or ""),
    }
    if not bundle.get("available"):
        return base_result

    feature_result = build_prediction_features(
        weather_data,
        base_result["features_used"],
        reference_datetime=reference_datetime,
    )
    base_result["missing_features"] = feature_result["missing_features"]
    base_result["observations"] = feature_result["observations"]
    if not feature_result["available"]:
        base_result["error_message"] = feature_result["error_message"]
        return base_result

    try:
        prediction = bundle["model"].predict(feature_result["features"])[0]
    except (AttributeError, ValueError, TypeError) as error:
        base_result["error_message"] = f"Nao foi possivel executar a previsao ML: {error}"
        return base_result

    base_result["prediction"] = str(prediction)
    base_result["probabilities"] = _predict_probabilities(
        bundle["model"],
        feature_result["features"],
        base_result["labels"],
    )
    base_result["error_message"] = ""
    return base_result


def _predict_probabilities(
    model: Any,
    features: pd.DataFrame,
    labels: list[str],
) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {}

    try:
        probabilities = model.predict_proba(features)[0]
    except (AttributeError, ValueError, TypeError):
        return {}

    model_labels = [str(label) for label in getattr(model, "classes_", labels)]
    probability_by_label = {
        label: float(probabilities[index])
        for index, label in enumerate(model_labels)
        if index < len(probabilities)
    }
    return {
        label: probability_by_label[label]
        for label in labels
        if label in probability_by_label
    }


def _feature_value(
    feature_name: str,
    weather_data: dict[str, object],
    weather_datetime: datetime,
    observations: list[str],
) -> object:
    if feature_name == "feels_like" and weather_data.get("feels_like") is None:
        observations.append("feels_like ausente; usando temperature como fallback.")
        return weather_data.get("temperature")

    if feature_name == "pressure" and weather_data.get("pressure") is None:
        if weather_data.get("pressure_station_hpa") is not None:
            return weather_data.get("pressure_station_hpa")
        if weather_data.get("pressure_sea_level_hpa") is not None:
            observations.append(
                "pressure ausente; usando pressure_sea_level_hpa como fallback."
            )
            return weather_data.get("pressure_sea_level_hpa")

    if feature_name == "month":
        return weather_datetime.month
    if feature_name == "hour":
        return weather_datetime.hour
    if feature_name == "day_of_year":
        return weather_datetime.timetuple().tm_yday

    return weather_data.get(feature_name)


def _resolve_weather_datetime(
    weather_data: dict[str, object],
    reference_datetime: datetime | None,
) -> datetime:
    for key in ["weather_datetime", "datetime"]:
        value = weather_data.get(key)
        if value is None:
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return parsed.to_pydatetime()
    return reference_datetime or datetime.now()


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _unavailable_result(error_message: str) -> dict[str, Any]:
    return {
        "available": False,
        "model": None,
        "metadata": {},
        "model_name": None,
        "selection_metric": None,
        "features_used": ML_FEATURE_COLUMNS,
        "labels": RISK_LEVEL_LABELS,
        "missing_features": [],
        "methodological_note": METHODOLOGICAL_NOTE,
        "error_message": error_message,
    }
