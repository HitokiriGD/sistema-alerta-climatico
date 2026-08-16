from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


INMET_HOURLY_TABLE = "inmet_hourly"
INMET_HOURLY_COLUMNS = [
    "station_code",
    "station_name",
    "state",
    "latitude",
    "longitude",
    "altitude_m",
    "date",
    "hour",
    "datetime",
    "temperature",
    "feels_like",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
    "source",
    "quality_flag",
]


def database_exists(path: str | Path) -> bool:
    """Verifica se o arquivo DuckDB historico existe."""
    return Path(path).exists() and Path(path).is_file()


def load_station_history_from_database(
    station_code: str,
    start_year: int,
    end_year: int,
    db_path: str | Path,
) -> pd.DataFrame:
    """Consulta apenas uma estacao e intervalo anual no DuckDB historico."""
    if not database_exists(db_path):
        return pd.DataFrame()

    start_datetime = f"{int(start_year)}-01-01"
    end_datetime = f"{int(end_year) + 1}-01-01"
    query = f"""
        SELECT {", ".join(INMET_HOURLY_COLUMNS)}
        FROM {INMET_HOURLY_TABLE}
        WHERE station_code = ?
          AND datetime >= ?
          AND datetime < ?
        ORDER BY datetime
    """
    with duckdb.connect(str(db_path), read_only=True) as connection:
        data = connection.execute(
            query,
            [station_code.strip().upper(), start_datetime, end_datetime],
        ).fetchdf()

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    return data.dropna(subset=["datetime"]).reset_index(drop=True)


def get_available_stations_from_database(db_path: str | Path) -> pd.DataFrame:
    """Lista estacoes disponiveis no DuckDB sem carregar todo o historico."""
    if not database_exists(db_path):
        return pd.DataFrame()

    query = f"""
        SELECT
            station_code,
            any_value(station_name) AS station_name,
            any_value(state) AS state,
            any_value(latitude) AS latitude,
            any_value(longitude) AS longitude,
            any_value(altitude_m) AS altitude_m,
            min(datetime) AS first_datetime,
            max(datetime) AS last_datetime,
            count(*) AS record_count
        FROM {INMET_HOURLY_TABLE}
        GROUP BY station_code
        ORDER BY state, station_name, station_code
    """
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute(query).fetchdf()


def get_database_metadata(db_path: str | Path) -> dict[str, Any]:
    """Retorna resumo simples da base DuckDB historica."""
    if not database_exists(db_path):
        return {
            "exists": False,
            "station_count": 0,
            "record_count": 0,
            "first_datetime": None,
            "last_datetime": None,
        }

    query = f"""
        SELECT
            count(DISTINCT station_code) AS station_count,
            count(*) AS record_count,
            min(datetime) AS first_datetime,
            max(datetime) AS last_datetime
        FROM {INMET_HOURLY_TABLE}
    """
    with duckdb.connect(str(db_path), read_only=True) as connection:
        metadata = connection.execute(query).fetchone()

    return {
        "exists": True,
        "station_count": int(metadata[0] or 0),
        "record_count": int(metadata[1] or 0),
        "first_datetime": metadata[2],
        "last_datetime": metadata[3],
    }
