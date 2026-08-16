import pandas as pd

from app.streamlit_app import historical_period_text
from app.streamlit_app import load_station_catalog_for_app
from app.streamlit_app import load_station_history_for_app
from app.streamlit_app import build_historical_statistics_rows
from app.streamlit_app import build_pressure_reference_details
from app.streamlit_app import find_nearest_station_by_coordinates
from app.streamlit_app import find_station_by_city_name
from app.streamlit_app import format_historical_anomalies
from app.streamlit_app import format_safe_table_rows
from app.streamlit_app import historical_analysis_status
from app.streamlit_app import normalize_historical_datetime
from app.streamlit_app import resolve_station_for_weather_data


def make_station_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "station_code": "A001",
                "station_name": "BRASILIA",
                "station_name_normalized": "BRASILIA",
                "city": "BRASILIA",
                "city_normalized": "BRASILIA",
                "state": "DF",
                "latitude": -15.7894,
                "longitude": -47.9258,
                "altitude_m": 1160.0,
                "station_label": "BRASILIA - DF | A001",
            },
            {
                "station_code": "A101",
                "station_name": "MANAUS",
                "station_name_normalized": "MANAUS",
                "city": "MANAUS",
                "city_normalized": "MANAUS",
                "state": "AM",
                "latitude": -3.103,
                "longitude": -60.016,
                "altitude_m": 61.0,
                "station_label": "MANAUS - AM | A101",
            },
        ]
    )


def test_find_nearest_station_by_coordinates_returns_manaus() -> None:
    station = find_nearest_station_by_coordinates(
        -3.1,
        -60.0,
        make_station_catalog(),
    )

    assert station is not None
    assert station["station_code"] == "A101"
    assert station["station_resolution_method"] == "coordinates"
    assert float(station["distance_km"]) < 5.0


def test_find_station_by_city_name_fallback_returns_manaus() -> None:
    station = find_station_by_city_name("Manaus", make_station_catalog())

    assert station is not None
    assert station["station_code"] == "A101"
    assert station["station_resolution_method"] == "city"


def test_resolve_station_for_weather_data_falls_back_to_city() -> None:
    station, message = resolve_station_for_weather_data(
        {"city": "Manaus"},
        make_station_catalog(),
    )

    assert station is not None
    assert station["station_code"] == "A101"
    assert station["station_resolution_method"] == "city"
    assert "nome da cidade" in message


def test_resolve_station_for_weather_data_reports_manual_fallback() -> None:
    station, message = resolve_station_for_weather_data(
        {"city": "Cidade Inexistente"},
        make_station_catalog(),
    )

    assert station is None
    assert "Nao foi possivel" in message


def test_format_safe_table_rows_converts_mixed_values_to_strings() -> None:
    rows = format_safe_table_rows(
        [
            {"Campo": "Registros", "Valor": 123},
            {"Campo": "Pressao", "Valor": {"p5": 870.0, "p95": 905.0}},
            {"Campo": "Distancia", "Valor": None},
        ]
    )

    assert rows == [
        {"Campo": "Registros", "Valor": "123"},
        {"Campo": "Pressao", "Valor": "p5: 870.0, p95: 905.0"},
        {"Campo": "Distancia", "Valor": "-"},
    ]


def test_normalize_historical_datetime_removes_invalid_mixed_dates() -> None:
    history = pd.DataFrame(
        {
            "datetime": [
                "2000-01-01 00:00:00",
                1.5,
                None,
                pd.Timestamp("2026-12-31 23:00:00"),
            ],
            "temperature": [20.0, 21.0, 22.0, 23.0],
        }
    )

    normalized = normalize_historical_datetime(history)

    assert len(normalized) == 2
    assert normalized["datetime"].min() == pd.Timestamp("2000-01-01 00:00:00")
    assert normalized["datetime"].max() == pd.Timestamp("2026-12-31 23:00:00")


def test_historical_period_text_handles_2000_to_2026() -> None:
    history = pd.DataFrame(
        {
            "datetime": [
                "2000-01-01",
                None,
                123.4,
                pd.Timestamp("2026-12-31"),
            ]
        }
    )

    assert historical_period_text(history) == "2000-01-01 a 2026-12-31"


def test_historical_period_text_without_valid_date() -> None:
    history = pd.DataFrame({"datetime": [None, 123.4]})

    assert historical_period_text(history) == "Nao identificado"


def test_load_station_catalog_for_app_uses_zip_when_database_missing(tmp_path) -> None:
    class FakeSettings:
        inmet_database_path = str(tmp_path / "missing.duckdb")

    class FakeClient:
        def build_station_catalog(self):
            return make_station_catalog()

    catalog, source = load_station_catalog_for_app(FakeSettings(), FakeClient())

    assert source == "zip"
    assert list(catalog["station_code"]) == ["A001", "A101"]


def test_load_station_history_for_app_uses_zip_fallback(tmp_path) -> None:
    class FakeSettings:
        inmet_database_path = str(tmp_path / "missing.duckdb")

    class FakeClient:
        def load_station_history(self, station_code, start_year, end_year):
            assert station_code == "A101"
            assert start_year == 2000
            assert end_year == 2026
            return pd.DataFrame(
                {
                    "station_code": ["A101", "A101"],
                    "datetime": ["2000-01-01", 123.4],
                    "temperature": [30.0, 31.0],
                }
            )

    history = load_station_history_for_app(
        FakeSettings(),
        FakeClient(),
        "A101",
        2000,
        2026,
        "zip",
    )

    assert len(history) == 1
    assert history.loc[0, "station_code"] == "A101"


def test_historical_analysis_status_requires_current_data() -> None:
    can_analyze, message = historical_analysis_status(None, None)

    assert can_analyze is False
    assert "dados meteorologicos atuais" in message


def test_historical_analysis_status_requires_history() -> None:
    can_analyze, message = historical_analysis_status({"temperature": 30.0}, None)

    assert can_analyze is False
    assert "historico INMET" in message


def test_format_historical_anomalies_uses_wind_kmh() -> None:
    rows = format_historical_anomalies(
        [
            {
                "anomaly_type": "HIGH_WIND",
                "severity": "HIGH",
                "current_value": 55.0,
                "historical_reference": "Percentil 95",
                "historical_value": 40.0,
                "variables_used": ["wind_speed"],
                "reason": "Vento acima do padrao historico.",
            }
        ]
    )

    assert rows == [
        {
            "Tipo": "Vento forte",
            "Severidade": "HIGH",
            "Valor atual": "55.0 km/h",
            "Referencia historica": "Percentil 95: 40.0 km/h",
            "Justificativa": "Vento acima do padrao historico.",
        }
    ]


def test_build_pressure_reference_details_for_estimated_pressure() -> None:
    message = build_pressure_reference_details(
        {"pressure_sea_level_hpa": 1010.0},
        {
            "evaluated": True,
            "pressure_station_hpa": 878.63,
            "is_estimated": True,
            "altitude_m": 1160.0,
        },
    )

    assert "Pressao estimada ao nivel da estacao" in message
    assert "nivel do mar" in message
    assert "1010.0 hPa" in message
    assert "878.6 hPa" in message
    assert "1160.0 m" in message
    assert "nivel da estacao INMET" in message


def test_build_pressure_reference_details_for_grnd_level() -> None:
    message = build_pressure_reference_details(
        {
            "pressure_reference": "openweather_grnd_level",
            "pressure_sea_level_hpa": 1012.0,
        },
        {
            "evaluated": True,
            "pressure_station_hpa": 890.0,
            "is_estimated": False,
        },
    )

    assert "grnd_level da OpenWeather" in message
    assert "1012.0 hPa" in message
    assert "890.0 hPa" in message


def test_build_historical_statistics_rows_marks_pressure_reference() -> None:
    rows = build_historical_statistics_rows(
        {
            "temperature": 30.0,
            "humidity": 40.0,
            "precipitation": 0.0,
            "wind_speed": 20.0,
            "pressure_sea_level_hpa": 1010.0,
        },
        {
            "has_historical_data": True,
            "anomalies": [],
            "pressure_comparison": {
                "evaluated": True,
                "pressure_station_hpa": 878.63,
                "is_estimated": True,
            },
            "statistics": {
                "pressure": {"mean": 890.0, "p5": 870.0, "p95": 905.0},
                "wind_speed": {"mean": 12.0, "p95": 35.0},
            },
        },
    )

    pressure_row = next(row for row in rows if row["Variavel"] == "Pressao")
    wind_row = next(row for row in rows if row["Variavel"] == "Vento")

    assert pressure_row["Valor atual"] == "878.6 hPa"
    assert pressure_row["Percentil usado"] == "P5 a P95: 870.0 hPa a 905.0 hPa"
    assert wind_row["Valor atual"] == "20.0 km/h"
