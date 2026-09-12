from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests

from scripts.download_inmet_database import download_inmet_database
from scripts.download_inmet_database import InmetDatabaseDownloadError
from src.config.settings import Settings
from src.config.settings import sanitize_sensitive_text


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


def ensure_inmet_database_available(
    settings: Settings,
    downloader: Callable[..., Path] = download_inmet_database,
) -> dict[str, object]:
    """Garante a base DuckDB local, baixando da release quando necessario."""
    database_path = Path(settings.inmet_database_path)
    sensitive_values = [
        settings.github_token,
        settings.inmet_database_url,
        settings.database_url,
    ]

    if database_exists(database_path):
        return {
            "available": True,
            "downloaded": False,
            "path": str(database_path),
            "message": "Base historica INMET ja disponivel localmente.",
            "error_message": "",
        }

    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        saved_path = downloader(
            url=settings.inmet_database_url,
            destination=database_path,
            force=False,
            expected_sha256=settings.inmet_database_sha256,
            release_repo=settings.inmet_database_release_repo,
            release_tag=settings.inmet_database_release_tag,
            asset_name=settings.inmet_database_asset_name,
            github_token=settings.github_token,
        )
    except (
        InmetDatabaseDownloadError,
        requests.RequestException,
        ValueError,
        OSError,
    ) as error:
        return {
            "available": False,
            "downloaded": False,
            "path": str(database_path),
            "message": "",
            "error_message": sanitize_sensitive_text(error, sensitive_values),
        }

    if database_exists(saved_path):
        return {
            "available": True,
            "downloaded": True,
            "path": str(saved_path),
            "message": "Base historica INMET carregada com sucesso.",
            "error_message": "",
        }

    return {
        "available": False,
        "downloaded": False,
        "path": str(database_path),
        "message": "",
        "error_message": (
            "Download finalizado, mas o arquivo DuckDB nao foi encontrado no "
            "caminho configurado."
        ),
    }


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
