from pathlib import Path

import duckdb
import pandas as pd

from src.data.inmet_database import INMET_HOURLY_COLUMNS
from src.data.inmet_database import INMET_HOURLY_TABLE
from src.data.inmet_database import database_exists
from src.data.inmet_database import get_available_stations_from_database
from src.data.inmet_database import get_database_metadata
from src.data.inmet_database import load_station_history_from_database


def write_test_database(db_path: Path) -> None:
    data = pd.DataFrame(
        [
            {
                "station_code": "A001",
                "station_name": "BRASILIA",
                "state": "DF",
                "latitude": -15.7,
                "longitude": -47.9,
                "altitude_m": 1160.0,
                "date": "2020/01/01",
                "hour": "0000 UTC",
                "datetime": pd.Timestamp("2020-01-01 00:00:00"),
                "temperature": 20.0,
                "feels_like": None,
                "humidity": 80.0,
                "precipitation": 0.0,
                "wind_speed": 10.0,
                "pressure": 890.0,
                "source": "INMET Historico",
                "quality_flag": "complete",
            },
            {
                "station_code": "A001",
                "station_name": "BRASILIA",
                "state": "DF",
                "latitude": -15.7,
                "longitude": -47.9,
                "altitude_m": 1160.0,
                "date": "2022/01/01",
                "hour": "0000 UTC",
                "datetime": pd.Timestamp("2022-01-01 00:00:00"),
                "temperature": 22.0,
                "feels_like": None,
                "humidity": 70.0,
                "precipitation": 0.0,
                "wind_speed": 12.0,
                "pressure": 892.0,
                "source": "INMET Historico",
                "quality_flag": "complete",
            },
            {
                "station_code": "A101",
                "station_name": "MANAUS",
                "state": "AM",
                "latitude": -3.1,
                "longitude": -60.0,
                "altitude_m": 61.0,
                "date": "2020/01/01",
                "hour": "0000 UTC",
                "datetime": pd.Timestamp("2020-01-01 00:00:00"),
                "temperature": 30.0,
                "feels_like": None,
                "humidity": 85.0,
                "precipitation": 1.0,
                "wind_speed": 5.0,
                "pressure": 1005.0,
                "source": "INMET Historico",
                "quality_flag": "complete",
            },
        ],
        columns=INMET_HOURLY_COLUMNS,
    )
    with duckdb.connect(str(db_path)) as connection:
        connection.register("data", data)
        connection.execute(
            f"CREATE TABLE {INMET_HOURLY_TABLE} AS "
            f"SELECT * FROM data"
        )


def test_database_exists_false_for_missing_file(tmp_path) -> None:
    assert database_exists(tmp_path / "missing.duckdb") is False


def test_load_station_history_from_database_filters_station_and_year(tmp_path) -> None:
    db_path = tmp_path / "inmet.duckdb"
    write_test_database(db_path)

    history = load_station_history_from_database("A001", 2020, 2020, db_path)

    assert list(history["station_code"]) == ["A001"]
    assert len(history) == 1
    assert history.loc[0, "datetime"].year == 2020
    assert history.loc[0, "temperature"] == 20.0


def test_load_station_history_from_database_missing_file_returns_empty(
    tmp_path,
) -> None:
    history = load_station_history_from_database(
        "A001",
        2020,
        2020,
        tmp_path / "x.duckdb",
    )

    assert history.empty


def test_get_available_stations_from_database_returns_metadata(tmp_path) -> None:
    db_path = tmp_path / "inmet.duckdb"
    write_test_database(db_path)

    stations = get_available_stations_from_database(db_path)

    assert set(stations["station_code"]) == {"A001", "A101"}
    assert "record_count" in stations.columns


def test_get_database_metadata(tmp_path) -> None:
    db_path = tmp_path / "inmet.duckdb"
    write_test_database(db_path)

    metadata = get_database_metadata(db_path)

    assert metadata["exists"] is True
    assert metadata["station_count"] == 2
    assert metadata["record_count"] == 3
