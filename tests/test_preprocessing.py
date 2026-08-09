import pandas as pd
import pytest

from src.processing.preprocessing import QUALITY_COMPLETE
from src.processing.preprocessing import QUALITY_INCOMPLETE
from src.processing.preprocessing import WEATHER_COLUMNS
from src.processing.preprocessing import clean_weather_data
from src.processing.preprocessing import convert_decimal_comma_to_float
from src.processing.preprocessing import preprocess_weather_dataframe
from src.processing.preprocessing import remove_weather_duplicates
from src.processing.preprocessing import sort_weather_data
from src.processing.preprocessing import standardize_weather_columns
from src.processing.preprocessing import validate_plausible_limits
from src.processing.preprocessing import validate_required_columns


def make_weather_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "station_code": "A001",
                "datetime": "2026-01-01 02:00:00",
                "temperature": "21,5",
                "feels_like": "22,0",
                "humidity": "75",
                "precipitation": "0,0",
                "wind_speed": "10,8",
                "pressure": "1010,5",
                "source": "test",
            },
            {
                "station_code": "A001",
                "datetime": "2026-01-01 01:00:00",
                "temperature": "20,0",
                "feels_like": None,
                "humidity": "80",
                "precipitation": "1,2",
                "wind_speed": "8",
                "pressure": "1009",
                "source": "test",
            },
        ]
    )


def test_standardize_weather_columns_maps_known_names() -> None:
    data = pd.DataFrame(
        [
            {
                "Temperatura": 28,
                "Sensacao Termica": 29,
                "Umidade": 70,
                "Precipitacao": 0,
                "Velocidade Vento": 12,
                "Pressao": 1012,
                "Data Hora": "2026-01-01 00:00:00",
                "Origem": "manual",
            }
        ]
    )

    standardized_data = standardize_weather_columns(data)

    assert "temperature" in standardized_data.columns
    assert "feels_like" in standardized_data.columns
    assert "datetime" in standardized_data.columns
    assert "source" in standardized_data.columns


def test_convert_decimal_comma_to_float() -> None:
    data = pd.DataFrame(
        [
            {
                "temperature": "21,5",
                "pressure": "1.010,5",
            }
        ]
    )

    converted_data = convert_decimal_comma_to_float(
        data,
        columns=["temperature", "pressure"],
    )

    assert converted_data.loc[0, "temperature"] == 21.5
    assert converted_data.loc[0, "pressure"] == 1010.5


def test_validate_required_columns_raises_for_missing_columns() -> None:
    data = pd.DataFrame([{"temperature": 28}])

    with pytest.raises(ValueError, match="Colunas obrigatorias ausentes"):
        validate_required_columns(data)


def test_preprocess_removes_records_without_datetime() -> None:
    data = make_weather_frame()
    data.loc[0, "datetime"] = None

    processed_data = preprocess_weather_dataframe(data)

    assert len(processed_data) == 1
    assert processed_data.loc[0, "datetime"] == pd.Timestamp("2026-01-01 01:00:00")


def test_preprocess_removes_records_without_temperature() -> None:
    data = make_weather_frame()
    data.loc[0, "temperature"] = None

    processed_data = preprocess_weather_dataframe(data)

    assert len(processed_data) == 1
    assert processed_data.loc[0, "temperature"] == 20.0


def test_validate_plausible_limits_sets_invalid_values_to_missing() -> None:
    data = make_weather_frame()
    data.loc[0, "temperature"] = "80"
    data.loc[0, "humidity"] = "120"
    data = convert_decimal_comma_to_float(data)

    validated_data = validate_plausible_limits(data)

    assert pd.isna(validated_data.loc[0, "temperature"])
    assert pd.isna(validated_data.loc[0, "humidity"])


def test_preprocess_creates_quality_flag() -> None:
    processed_data = preprocess_weather_dataframe(make_weather_frame())

    assert processed_data.loc[0, "quality_flag"] == QUALITY_INCOMPLETE
    assert processed_data.loc[1, "quality_flag"] == QUALITY_COMPLETE


def test_sort_weather_data_orders_by_datetime() -> None:
    data = make_weather_frame()
    data["datetime"] = pd.to_datetime(data["datetime"])

    sorted_data = sort_weather_data(data)

    assert sorted_data.loc[0, "datetime"] == pd.Timestamp("2026-01-01 01:00:00")


def test_remove_weather_duplicates_by_station_and_datetime() -> None:
    data = pd.DataFrame(
        [
            {"station_code": "A001", "datetime": "2026-01-01 00:00:00", "value": 1},
            {"station_code": "A001", "datetime": "2026-01-01 00:00:00", "value": 2},
            {"station_code": "A002", "datetime": "2026-01-01 00:00:00", "value": 3},
        ]
    )

    deduplicated_data = remove_weather_duplicates(data)

    assert len(deduplicated_data) == 2
    assert deduplicated_data.loc[0, "value"] == 1
    assert deduplicated_data.loc[1, "value"] == 3


def test_preprocess_weather_dataframe_runs_full_pipeline() -> None:
    data = make_weather_frame()
    duplicate = data.iloc[[0]].copy()
    data = pd.concat([data, duplicate], ignore_index=True)

    processed_data = preprocess_weather_dataframe(data)

    assert len(processed_data) == 2
    assert processed_data.loc[0, "datetime"] == pd.Timestamp("2026-01-01 01:00:00")
    assert processed_data.loc[1, "temperature"] == 21.5
    assert "quality_flag" in processed_data.columns


def test_clean_weather_data_keeps_existing_feature_matrix_behavior() -> None:
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
