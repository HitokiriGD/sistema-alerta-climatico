from typing import Any


try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - ambiente sem dependencia instalada
    psycopg = None
    Jsonb = None


class WeatherRepositoryError(RuntimeError):
    """Erro controlado da camada de persistencia meteorologica."""


CREATE_TABLE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS weather_observations (
        id BIGSERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        city TEXT NOT NULL,
        country TEXT NOT NULL,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        weather_datetime TIMESTAMPTZ,
        temperature DOUBLE PRECISION,
        feels_like DOUBLE PRECISION,
        humidity DOUBLE PRECISION,
        precipitation DOUBLE PRECISION,
        wind_speed DOUBLE PRECISION,
        pressure_sea_level_hpa DOUBLE PRECISION,
        pressure_station_hpa DOUBLE PRECISION,
        clouds DOUBLE PRECISION,
        weather_description TEXT,
        raw_payload JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_weather_observations_city_country_collected
    ON weather_observations(city, country, collected_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_weather_observations_weather_datetime
    ON weather_observations(weather_datetime)
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_runs (
        id BIGSERIAL PRIMARY KEY,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        finished_at TIMESTAMPTZ,
        status TEXT NOT NULL,
        records_inserted INTEGER NOT NULL DEFAULT 0,
        records_failed INTEGER NOT NULL DEFAULT 0,
        error_message TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monitored_cities (
        id BIGSERIAL PRIMARY KEY,
        city TEXT NOT NULL,
        country TEXT NOT NULL,
        active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(city, country)
    )
    """,
]


def connect_database(database_url: str):
    """Abre conexao Postgres sem imprimir a URL do banco."""
    if not database_url.strip():
        raise WeatherRepositoryError("Configure DATABASE_URL no .env.")
    if psycopg is None:
        raise WeatherRepositoryError(
            "Dependencia psycopg ausente. Instale requirements.txt."
        )
    return psycopg.connect(database_url)


def create_tables(connection: Any) -> None:
    """Cria tabelas e indices necessarios para a base OpenWeather."""
    with connection.cursor() as cursor:
        for statement in CREATE_TABLE_STATEMENTS:
            cursor.execute(statement)


def upsert_monitored_cities(
    connection: Any,
    cities: tuple[tuple[str, str], ...],
) -> int:
    """Insere ou reativa cidades monitoradas."""
    with connection.cursor() as cursor:
        for city, country in cities:
            cursor.execute(
                """
                INSERT INTO monitored_cities (city, country, active)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (city, country)
                DO UPDATE SET active = TRUE
                """,
                (city, country),
            )
    return len(cities)


def insert_weather_observation(
    connection: Any,
    observation: dict[str, Any],
) -> int | None:
    """Insere um snapshot meteorologico padronizado."""
    columns = [
        "source",
        "city",
        "country",
        "latitude",
        "longitude",
        "weather_datetime",
        "temperature",
        "feels_like",
        "humidity",
        "precipitation",
        "wind_speed",
        "pressure_sea_level_hpa",
        "pressure_station_hpa",
        "clouds",
        "weather_description",
        "raw_payload",
    ]
    values = [observation.get(column) for column in columns]
    values[-1] = _jsonb_or_none(values[-1])

    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"""
        INSERT INTO weather_observations ({", ".join(columns)})
        VALUES ({placeholders})
        RETURNING id
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, tuple(values))
        row = cursor.fetchone()

    if not row:
        return None
    return int(row[0])


def register_ingestion_run(
    connection: Any,
    status: str,
    records_inserted: int,
    records_failed: int,
    error_message: str | None = None,
) -> int | None:
    """Registra o resultado consolidado de uma execucao de coleta."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ingestion_runs (
                finished_at,
                status,
                records_inserted,
                records_failed,
                error_message
            )
            VALUES (NOW(), %s, %s, %s, %s)
            RETURNING id
            """,
            (status, records_inserted, records_failed, error_message),
        )
        row = cursor.fetchone()

    if not row:
        return None
    return int(row[0])


def get_collection_summary(connection: Any) -> dict[str, Any]:
    """Consulta resumo simples da base propria OpenWeather."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_observations,
                MIN(collected_at) AS first_collection,
                MAX(collected_at) AS last_collection,
                COUNT(DISTINCT city || ':' || country) AS collected_cities
            FROM weather_observations
            """
        )
        row = cursor.fetchone() or (0, None, None, 0)
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM monitored_cities
            WHERE active = TRUE
            """
        )
        monitored_row = cursor.fetchone() or (0,)

    return {
        "total_observations": int(row[0] or 0),
        "first_collection": row[1],
        "last_collection": row[2],
        "collected_cities": int(row[3] or 0),
        "monitored_cities": int(monitored_row[0] or 0),
    }


def cleanup_old_observations(
    connection: Any,
    retention_days: int,
    max_rows: int,
) -> dict[str, int]:
    """Remove registros antigos por dias e por limite maximo de linhas."""
    retention_days = max(0, int(retention_days))
    max_rows = max(0, int(max_rows))

    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM weather_observations
            WHERE collected_at < NOW() - (%s * INTERVAL '1 day')
            """,
            (retention_days,),
        )
        deleted_by_retention = max(cursor.rowcount or 0, 0)

        cursor.execute("SELECT COUNT(*) FROM weather_observations")
        current_rows = int((cursor.fetchone() or (0,))[0] or 0)
        rows_to_delete = max(current_rows - max_rows, 0)
        deleted_by_max_rows = 0

        if rows_to_delete > 0:
            cursor.execute(
                """
                DELETE FROM weather_observations
                WHERE id IN (
                    SELECT id
                    FROM weather_observations
                    ORDER BY collected_at ASC, id ASC
                    LIMIT %s
                )
                """,
                (rows_to_delete,),
            )
            deleted_by_max_rows = max(cursor.rowcount or 0, 0)

        cursor.execute("SELECT COUNT(*) FROM weather_observations")
        remaining_rows = int((cursor.fetchone() or (0,))[0] or 0)

    return {
        "deleted_by_retention": deleted_by_retention,
        "deleted_by_max_rows": deleted_by_max_rows,
        "remaining_rows": remaining_rows,
    }


def _jsonb_or_none(value: Any) -> Any:
    if value is None:
        return None
    if Jsonb is None:
        return value
    return Jsonb(value)
