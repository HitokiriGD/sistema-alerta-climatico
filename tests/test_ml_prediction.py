import json
from datetime import datetime

import pandas as pd

from app.streamlit_app import get_ml_prediction_for_dashboard
from src.ml.predict import METHODOLOGICAL_NOTE
from src.ml.predict import build_prediction_features
from src.ml.predict import load_model_bundle
from src.ml.predict import model_files_available
from src.ml.predict import predict_risk_level


FEATURES = [
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


class FakeModel:
    def predict(self, features: pd.DataFrame) -> list[str]:
        assert list(features.columns) == FEATURES
        return ["alto"]


def weather_data() -> dict[str, object]:
    return {
        "temperature": 32.0,
        "feels_like": 34.0,
        "humidity": 45.0,
        "precipitation": 0.0,
        "wind_speed": 12.0,
        "pressure": 1008.0,
        "weather_datetime": "2026-09-05 14:30:00",
    }


def model_bundle(model=None) -> dict[str, object]:
    return {
        "available": True,
        "model": model or FakeModel(),
        "metadata": {
            "selected_model_name": "random_forest",
            "metric_used": "f1_macro",
            "features": FEATURES,
        },
        "model_name": "random_forest",
        "selection_metric": "f1_macro",
        "features_used": FEATURES,
        "missing_features": [],
        "methodological_note": METHODOLOGICAL_NOTE,
        "error_message": "",
    }


def test_model_files_available_identifies_missing_model(tmp_path) -> None:
    assert model_files_available(
        tmp_path / "missing.joblib",
        tmp_path / "missing.json",
    ) is False


def test_load_model_bundle_loads_fake_model(tmp_path) -> None:
    model_path = tmp_path / "risk_level_model.joblib"
    metadata_path = tmp_path / "risk_level_model_metadata.json"
    model_path.write_text("fake", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "selected_model_name": "random_forest",
                "metric_used": "f1_macro",
                "features": FEATURES,
            }
        ),
        encoding="utf-8",
    )

    bundle = load_model_bundle(
        model_path,
        metadata_path,
        loader=lambda path: FakeModel(),
    )

    assert bundle["available"] is True
    assert isinstance(bundle["model"], FakeModel)
    assert bundle["model_name"] == "random_forest"
    assert bundle["selection_metric"] == "f1_macro"
    assert bundle["features_used"] == FEATURES


def test_build_prediction_features_from_weather_data() -> None:
    result = build_prediction_features(weather_data(), FEATURES)
    features = result["features"]

    assert result["available"] is True
    assert features.loc[0, "temperature"] == 32.0
    assert features.loc[0, "month"] == 9.0
    assert features.loc[0, "hour"] == 14.0
    assert features.loc[0, "day_of_year"] == 248.0


def test_build_prediction_features_fallback_feels_like_temperature() -> None:
    data = weather_data()
    data.pop("feels_like")

    result = build_prediction_features(data, FEATURES)

    assert result["features"].loc[0, "feels_like"] == 32.0
    assert "feels_like ausente" in result["observations"][0]


def test_build_prediction_features_fallback_pressure_station() -> None:
    data = weather_data()
    data.pop("pressure")
    data["pressure_station_hpa"] = 930.0

    result = build_prediction_features(data, FEATURES)

    assert result["features"].loc[0, "pressure"] == 930.0
    assert result["missing_features"] == []


def test_build_prediction_features_returns_friendly_error_when_missing() -> None:
    data = weather_data()
    data.pop("humidity")

    result = build_prediction_features(data, FEATURES)

    assert result["available"] is False
    assert result["missing_features"] == ["humidity"]
    assert "Nao foi possivel montar" in result["error_message"]


def test_prediction_result_contains_methodological_note() -> None:
    result = predict_risk_level(weather_data(), model_bundle=model_bundle())

    assert result["methodological_note"] == METHODOLOGICAL_NOTE


def test_predict_risk_level_returns_expected_risk_level() -> None:
    result = predict_risk_level(weather_data(), model_bundle=model_bundle())

    assert result["available"] is True
    assert result["prediction"] == "alto"
    assert result["model_name"] == "random_forest"
    assert result["selection_metric"] == "f1_macro"


def test_dashboard_helper_does_not_break_without_model() -> None:
    result = get_ml_prediction_for_dashboard(
        weather_data(),
        model_bundle={
            "available": False,
            "error_message": "Modelo ML ainda nao encontrado localmente.",
        },
    )

    assert result["available"] is False
    assert "Modelo ML" in result["error_message"]
