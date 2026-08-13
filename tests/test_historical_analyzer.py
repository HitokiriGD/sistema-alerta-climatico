import numpy as np
import pandas as pd
import pytest
from src.alerts.historical_analyzer import HistoricalAnalyzer


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