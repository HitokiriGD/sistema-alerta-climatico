from src.data.weather_client import normalize_weather_payload


def test_normalize_weather_payload_returns_standard_fields() -> None:
    payload = {
        "main": {
            "temp": 26.5,
            "feels_like": 27.0,
            "humidity": 72,
            "pressure": 1012,
        },
        "rain": {"1h": 4.2},
        "wind": {"speed": 10.0},
    }

    weather_data = normalize_weather_payload(payload)

    assert weather_data == {
        "temperature": 26.5,
        "feels_like": 27.0,
        "humidity": 72.0,
        "precipitation": 4.2,
        "wind_speed": 36.0,
        "pressure": 1012.0,
    }
