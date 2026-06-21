import pandas as pd
import pytest

from src.processing.preprocessing import WEATHER_COLUMNS, clean_weather_data


def test_clean_weather_data_returns_expected_columns() -> None:
    data = pd.DataFrame(
        [
            {
                "temperature": "28",
                "feels_like": 29,
                "humidity": 70,
                "precipitation": None,
                "wind_speed": 12,
                "pressure": 1012,
                "extra_column": "ignored",
            },
            {
                "temperature": 30,
                "feels_like": 31,
                "humidity": 65,
                "precipitation": 2,
                "wind_speed": 10,
                "pressure": 1010,
                "extra_column": "ignored",
            },
        ]
    )

    cleaned_data = clean_weather_data(data)

    assert list(cleaned_data.columns) == WEATHER_COLUMNS
    assert cleaned_data.isna().sum().sum() == 0


def test_clean_weather_data_requires_weather_columns() -> None:
    data = pd.DataFrame([{"temperature": 28}])

    with pytest.raises(ValueError):
        clean_weather_data(data)
