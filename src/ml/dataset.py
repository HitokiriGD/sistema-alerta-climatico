from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.alerts.risk_classifier import classify_weather_risk
from src.data.inmet_database import INMET_HOURLY_TABLE


IDENTIFICATION_COLUMNS = ["station_code", "station_name", "state"]
WEATHER_COLUMNS = [
    "temperature",
    "feels_like",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
]
REQUIRED_COLUMNS = [
    "datetime",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
]
FEATURE_COLUMNS = ["year", "month", "day", "hour", "day_of_year"]
LABEL_COLUMNS = ["risk_level", "event_type"]
OUTPUT_COLUMNS = (
    IDENTIFICATION_COLUMNS
    + ["datetime"]
    + FEATURE_COLUMNS
    + WEATHER_COLUMNS
    + LABEL_COLUMNS
)


def parse_station_codes(stations: str | None) -> list[str]:
    """Interpreta lista de codigos de estacao separados por virgula."""
    if not stations:
        return []
    return [
        station.strip().upper()
        for station in stations.split(",")
        if station.strip()
    ]


def load_inmet_history_from_duckdb(
    db_path: str | Path,
    start_year: int,
    end_year: int,
    stations: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Carrega historico INMET do DuckDB para geracao do dataset rotulado."""
    selected_columns = [
        "station_code",
        "station_name",
        "state",
        "datetime",
        "temperature",
        "feels_like",
        "humidity",
        "precipitation",
        "wind_speed",
        "pressure",
    ]
    start_datetime = f"{int(start_year)}-01-01"
    end_datetime = f"{int(end_year) + 1}-01-01"
    params: list[Any] = [start_datetime, end_datetime]

    query = f"""
        SELECT {", ".join(selected_columns)}
        FROM {INMET_HOURLY_TABLE}
        WHERE datetime >= ?
          AND datetime < ?
    """

    station_codes = [
        station.strip().upper()
        for station in stations or []
        if station.strip()
    ]
    if station_codes:
        placeholders = ", ".join("?" for _ in station_codes)
        query += f" AND station_code IN ({placeholders})"
        params.extend(station_codes)

    query += " ORDER BY datetime"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute(query, params).fetchdf()


def build_labeled_dataset(history: pd.DataFrame) -> pd.DataFrame:
    """Gera features e rotulos iniciais a partir do historico do INMET."""
    data = history.copy(deep=True)

    for column in IDENTIFICATION_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    if "feels_like" not in data.columns:
        # O INMET historico pode nao ter sensacao termica; nesse caso, usamos
        # a temperatura como fallback para manter o classificador aplicavel.
        data["feels_like"] = data.get("temperature", pd.NA)

    for column in REQUIRED_COLUMNS + ["feels_like"]:
        if column not in data.columns:
            data[column] = pd.NA

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    for column in WEATHER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["feels_like"] = data["feels_like"].fillna(data["temperature"])
    data = data.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["day_of_year"] = data["datetime"].dt.dayofyear

    labels = data.apply(_classify_history_row, axis=1)
    data = pd.concat([data, labels], axis=1)

    return data[OUTPUT_COLUMNS].reset_index(drop=True)


def summarize_labeled_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    """Retorna resumo simples do dataset supervisionado inicial."""
    if dataset.empty:
        return {
            "total_records": 0,
            "start_datetime": None,
            "end_datetime": None,
            "station_count": 0,
            "risk_level_distribution": {},
            "event_type_distribution": {},
        }

    return {
        "total_records": int(len(dataset)),
        "start_datetime": dataset["datetime"].min(),
        "end_datetime": dataset["datetime"].max(),
        "station_count": int(dataset["station_code"].nunique(dropna=True)),
        "risk_level_distribution": _value_counts(dataset, "risk_level"),
        "event_type_distribution": _value_counts(dataset, "event_type"),
    }


def _classify_history_row(row: pd.Series) -> pd.Series:
    risk = classify_weather_risk(
        {
            "temperature": row["temperature"],
            "feels_like": row["feels_like"],
            "humidity": row["humidity"],
            "precipitation": row["precipitation"],
            "wind_speed": row["wind_speed"],
            "pressure": row["pressure"],
        }
    )
    return pd.Series(
        {
            "risk_level": risk.risk_level,
            "event_type": risk.event_type,
        }
    )


def _value_counts(dataset: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in dataset[column].value_counts(dropna=False).items()
    }
