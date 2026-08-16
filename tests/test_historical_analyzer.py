import numpy as np
import pandas as pd
import pytest
from src.alerts.historical_analyzer import HistoricalAnalyzer
from src.alerts.historical_analyzer import normalize_pressure_to_station_level
from src.alerts.historical_analyzer import (
    resolve_current_pressure_for_historical_comparison,
)


@pytest.fixture
def mock_historical_df():
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "datetime": dates,
        "temperature": np.linspace(15.0, 35.0, n),
        "humidity": np.linspace(20.0, 90.0, n),
        "precipitation": np.random.exponential(scale=2.0, size=n),
        "wind_speed": np.linspace(0.5, 12.0, n),
        "pressure": np.linspace(1000.0, 1025.0, n),
    })


def test_temperature_above_p95(mock_historical_df):
    analyzer = HistoricalAnalyzer()
    current_data = {"temperature": 36.0}
    result = analyzer.analyze(current_data, mock_historical_df)

    assert result["has_historical_data"] is True
    anomalies = [a for a in result["anomalies"] if a["anomaly_type"] == "HIGH_TEMPERATURE"]
    assert len(anomalies) == 1


def test_humidity_below_p10(mock_historical_df):
    analyzer = HistoricalAnalyzer()
    current_data = {"humidity": 15.0}
    result = analyzer.analyze(current_data, mock_historical_df)

    anomalies = [a for a in result["anomalies"] if a["anomaly_type"] == "LOW_HUMIDITY"]
    assert len(anomalies) == 1


def test_empty_historical_data():
    analyzer = HistoricalAnalyzer()
    result = analyzer.analyze({"temperature": 25.0}, pd.DataFrame())

    assert result["has_historical_data"] is False
    assert len(result["anomalies"]) == 0


def test_pressure_without_grnd_level_is_estimated_by_altitude():
    pressure = normalize_pressure_to_station_level(
        pressure_sea_level_hpa=1010.0,
        altitude_m=1160.0,
    )

    assert pressure == pytest.approx(878.63, abs=0.5)


def test_resolve_pressure_uses_openweather_grnd_level_first():
    current_data = {
        "pressure_sea_level_hpa": 1010.0,
        "pressure_station_hpa": 890.0,
    }

    pressure = resolve_current_pressure_for_historical_comparison(
        current_data,
        {"altitude_m": 1160.0},
    )

    assert pressure["evaluated"] is True
    assert pressure["pressure_station_hpa"] == 890.0
    assert pressure["reference_used"] == "pressure_station_hpa"
    assert pressure["is_estimated"] is False


def test_brasilia_sea_level_pressure_is_estimated_without_absurd_false_positive():
    analyzer = HistoricalAnalyzer(min_samples_seasonal=999)
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    historical_df = pd.DataFrame(
        {
            "datetime": dates,
            "pressure": np.linspace(870.0, 905.0, 200),
            "altitude_m": 1160.0,
        }
    )
    current_data = {"pressure_sea_level_hpa": 1010.0}

    result = analyzer.analyze(
        current_data,
        historical_df,
        current_date=pd.Timestamp("2020-01-15").to_pydatetime(),
    )

    pressure_comparison = result["pressure_comparison"]
    assert pressure_comparison["evaluated"] is True
    assert pressure_comparison["is_estimated"] is True
    assert pressure_comparison["pressure_station_hpa"] == pytest.approx(
        878.63,
        abs=0.5,
    )
    assert result["statistics"]["pressure"]["p95"] < 1010.0
    assert result["statistics"]["pressure"]["p5"] < 878.63
    assert not [
        anomaly
        for anomaly in result["anomalies"]
        if anomaly["anomaly_type"] == "PRESSURE_ANOMALY"
    ]


def test_historical_analyzer_compares_station_pressure_not_raw_sea_level():
    analyzer = HistoricalAnalyzer(min_samples_seasonal=999)
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    historical_df = pd.DataFrame(
        {
            "datetime": dates,
            "pressure": np.linspace(870.0, 905.0, 200),
        }
    )
    current_data = {
        "pressure": 1010.0,
        "pressure_sea_level_hpa": 1010.0,
        "pressure_station_hpa": 890.0,
    }

    result = analyzer.analyze(
        current_data,
        historical_df,
        current_date=pd.Timestamp("2020-01-15").to_pydatetime(),
    )

    assert result["pressure_comparison"]["pressure_station_hpa"] == 890.0
    assert result["pressure_comparison"]["reference_used"] == "pressure_station_hpa"
    assert result["statistics"]["pressure"]["p95"] < 1010.0
    assert result["statistics"]["pressure"]["p5"] < 890.0
    assert result["statistics"]["pressure"]["p95"] > 890.0
    assert not [
        anomaly
        for anomaly in result["anomalies"]
        if anomaly["anomaly_type"] == "PRESSURE_ANOMALY"
    ]


def test_pressure_is_not_evaluated_without_altitude_or_station_pressure():
    analyzer = HistoricalAnalyzer()
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    historical_df = pd.DataFrame(
        {
            "datetime": dates,
            "pressure": np.linspace(880.0, 905.0, 200),
        }
    )
    current_data = {"pressure_sea_level_hpa": 1010.0}

    result = analyzer.analyze(
        current_data,
        historical_df,
        current_date=pd.Timestamp("2020-01-15").to_pydatetime(),
    )

    assert result["pressure_comparison"]["evaluated"] is False
    assert "altitude" in result["pressure_comparison"]["message"]
    assert not [
        anomaly
        for anomaly in result["anomalies"]
        if anomaly["anomaly_type"] == "PRESSURE_ANOMALY"
    ]


def test_wind_anomaly_message_uses_kmh(mock_historical_df):
    analyzer = HistoricalAnalyzer()
    current_data = {"wind_speed": 20.0}

    result = analyzer.analyze(
        current_data,
        mock_historical_df,
        current_date=pd.Timestamp("2020-01-15").to_pydatetime(),
    )

    wind_anomaly = next(
        anomaly
        for anomaly in result["anomalies"]
        if anomaly["anomaly_type"] == "HIGH_WIND"
    )
    assert "km/h" in wind_anomaly["reason"]
    assert "m/s" not in wind_anomaly["reason"]
