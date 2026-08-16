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
        "pressure_sea_level_hpa": 1012.0,
    }


def test_normalize_weather_payload_preserves_grnd_level_pressure() -> None:
    payload = {
        "main": {
            "temp": 26.5,
            "feels_like": 27.0,
            "humidity": 72,
            "pressure": 1012,
            "grnd_level": 890,
        },
        "wind": {"speed": 3.0},
    }

    weather_data = normalize_weather_payload(payload)

    assert weather_data["pressure"] == 1012.0
    assert weather_data["pressure_sea_level_hpa"] == 1012.0
    assert weather_data["pressure_station_hpa"] == 890.0
    assert weather_data["pressure_reference"] == "openweather_grnd_level"


def test_normalize_weather_payload_preserves_coordinates() -> None:
    payload = {
        "name": "Manaus",
        "coord": {"lat": -3.1, "lon": -60.0},
        "main": {
            "temp": 30.0,
            "feels_like": 34.0,
            "humidity": 70,
            "pressure": 1010,
        },
        "wind": {"speed": 2.0},
    }

    weather_data = normalize_weather_payload(payload)

    assert weather_data["city"] == "Manaus"
    assert weather_data["latitude"] == -3.1
    assert weather_data["longitude"] == -60.0
