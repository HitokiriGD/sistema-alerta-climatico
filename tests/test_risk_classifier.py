from src.alerts.risk_classifier import classify_weather_risk


def test_classify_weather_risk_for_heavy_rain() -> None:
    weather_data = {
        "temperature": 24.0,
        "humidity": 90.0,
        "precipitation": 55.0,
        "wind_speed": 10.0,
    }

    risk = classify_weather_risk(weather_data)

    assert risk.level == "alto"
    assert risk.event == "chuva intensa"


def test_classify_weather_risk_for_normal_condition() -> None:
    weather_data = {
        "temperature": 24.0,
        "humidity": 60.0,
        "precipitation": 0.0,
        "wind_speed": 8.0,
    }

    risk = classify_weather_risk(weather_data)

    assert risk.level == "baixo"
