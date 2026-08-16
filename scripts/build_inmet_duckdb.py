import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.config.settings import load_settings
from src.data.inmet_client import InmetClient
from src.data.inmet_client import InmetHistoricalDataError
from src.data.inmet_database import INMET_HOURLY_COLUMNS
from src.data.inmet_database import INMET_HOURLY_TABLE
from src.processing.preprocessing import preprocess_weather_dataframe


def build_inmet_duckdb(
    start_year: int,
    end_year: int,
    stations: list[str] | None = None,
    output_path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Gera uma base DuckDB historica consultavel por estacao e periodo."""
    current_settings = settings or load_settings()
    destination = output_path or Path(current_settings.inmet_database_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    client = InmetClient(current_settings)
    station_codes = _resolve_station_codes(client, stations)

    with duckdb.connect(str(destination)) as connection:
        connection.execute(f"DROP TABLE IF EXISTS {INMET_HOURLY_TABLE}")
        _create_hourly_table(connection)

        for station_code in station_codes:
            try:
                station_history = client.load_station_history(
                    station_code,
                    start_year,
                    end_year,
                )
            except InmetHistoricalDataError as error:
                print(f"Estacao {station_code} ignorada: {error}")
                continue

            processed_history = _prepare_history_for_database(station_history)
            if processed_history.empty:
                continue

            connection.register("station_history", processed_history)
            connection.execute(
                f"""
                INSERT INTO {INMET_HOURLY_TABLE}
                SELECT {", ".join(INMET_HOURLY_COLUMNS)}
                FROM station_history
                """
            )
            connection.unregister("station_history")

    return destination


def _resolve_station_codes(
    client: InmetClient,
    stations: list[str] | None,
) -> list[str]:
    if stations:
        return [station.strip().upper() for station in stations if station.strip()]

    catalog = client.build_station_catalog()
    return sorted(catalog["station_code"].dropna().astype(str).str.upper().unique())


def _prepare_history_for_database(history: pd.DataFrame) -> pd.DataFrame:
    processed_history = preprocess_weather_dataframe(history)
    processed_history["datetime"] = pd.to_datetime(
        processed_history["datetime"],
        errors="coerce",
    )
    processed_history = processed_history.dropna(subset=["station_code", "datetime"])
    processed_history = processed_history.drop_duplicates(
        subset=["station_code", "datetime"],
        keep="first",
    ).reset_index(drop=True)

    for column in INMET_HOURLY_COLUMNS:
        if column not in processed_history.columns:
            processed_history[column] = pd.NA

    return processed_history[INMET_HOURLY_COLUMNS]


def _create_hourly_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE {INMET_HOURLY_TABLE} (
            station_code VARCHAR,
            station_name VARCHAR,
            state VARCHAR,
            latitude DOUBLE,
            longitude DOUBLE,
            altitude_m DOUBLE,
            date VARCHAR,
            hour VARCHAR,
            datetime TIMESTAMP,
            temperature DOUBLE,
            feels_like DOUBLE,
            humidity DOUBLE,
            precipitation DOUBLE,
            wind_speed DOUBLE,
            pressure DOUBLE,
            source VARCHAR,
            quality_flag VARCHAR
        )
        """
    )


def parse_args() -> argparse.Namespace:
    """Le parametros de linha de comando para gerar o DuckDB historico."""
    parser = argparse.ArgumentParser(
        description="Gera DuckDB historico a partir dos ZIPs locais do INMET.",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--stations",
        default="",
        help="Codigos separados por virgula, por exemplo A001,A101,A312.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Caminho de saida. Padrao: INMET_DATABASE_PATH.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa a geracao local da base DuckDB historica."""
    args = parse_args()
    stations = (
        [station.strip() for station in args.stations.split(",")]
        if args.stations.strip()
        else None
    )
    saved_path = build_inmet_duckdb(
        start_year=args.start_year,
        end_year=args.end_year,
        stations=stations,
        output_path=args.output,
    )
    print(f"Base historica DuckDB salva em: {saved_path}")


if __name__ == "__main__":
    main()
