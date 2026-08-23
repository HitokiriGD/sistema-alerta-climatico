import argparse
from dataclasses import replace
from datetime import datetime
from datetime import timezone
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.config.settings import load_settings
from src.data import weather_repository
from src.data.openweather_client import OpenWeatherClient
from src.data.weather_client import normalize_weather_payload


POSTGRES_CONNECTION_ERROR_MESSAGE = (
    "Erro ao conectar ao banco Postgres. Verifique DATABASE_URL no .env."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coleta snapshots atuais da OpenWeather para Postgres.",
    )
    parser.add_argument("--city", help="Cidade especifica para coleta.")
    parser.add_argument("--country", default="BR", help="Pais da cidade.")
    return parser.parse_args()


def resolve_collection_cities(
    settings: Settings,
    city: str | None = None,
    country: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Resolve uma cidade manual ou a lista configurada de capitais."""
    if city and city.strip():
        return ((city.strip(), (country or "BR").strip().upper() or "BR"),)
    return settings.openweather_collection_cities


def build_observation_from_payload(
    payload: dict[str, Any],
    requested_city: str,
    country: str,
    store_raw_payload: bool,
) -> dict[str, Any]:
    """Monta registro de banco a partir do payload OpenWeather."""
    weather_data = normalize_weather_payload(payload)
    weather_items = payload.get("weather", [])
    first_weather = weather_items[0] if weather_items else {}
    clouds = payload.get("clouds", {})
    weather_datetime = None
    if payload.get("dt") is not None:
        weather_datetime = datetime.fromtimestamp(int(payload["dt"]), timezone.utc)

    return {
        "source": "OpenWeather",
        "city": str(weather_data.get("city", requested_city)),
        "country": country.strip().upper() or "BR",
        "latitude": weather_data.get("latitude"),
        "longitude": weather_data.get("longitude"),
        "weather_datetime": weather_datetime,
        "temperature": weather_data.get("temperature"),
        "feels_like": weather_data.get("feels_like"),
        "humidity": weather_data.get("humidity"),
        "precipitation": weather_data.get("precipitation"),
        "wind_speed": weather_data.get("wind_speed"),
        "pressure_sea_level_hpa": weather_data.get("pressure_sea_level_hpa"),
        "pressure_station_hpa": weather_data.get("pressure_station_hpa"),
        "clouds": clouds.get("all"),
        "weather_description": first_weather.get("description"),
        "raw_payload": payload if store_raw_payload else None,
    }


def collect_openweather_snapshots(
    settings: Settings,
    cities: tuple[tuple[str, str], ...],
    repository=weather_repository,
    client_class=OpenWeatherClient,
) -> dict[str, Any]:
    """Coleta e persiste snapshots, mantendo falhas isoladas por cidade."""
    if not settings.database_url.strip():
        print("Configure DATABASE_URL no .env.")
        return {"exit_code": 1}

    records_inserted = 0
    failures: list[dict[str, str]] = []
    cleanup_result = {
        "deleted_by_retention": 0,
        "deleted_by_max_rows": 0,
        "remaining_rows": 0,
    }

    try:
        with repository.connect_database(settings.database_url) as connection:
            for city, country in cities:
                try:
                    city_settings = replace(settings, default_country=country)
                    payload = client_class(city_settings).fetch_current_weather(city)
                    observation = build_observation_from_payload(
                        payload,
                        city,
                        country,
                        settings.openweather_store_raw_payload,
                    )
                    repository.insert_weather_observation(connection, observation)
                    records_inserted += 1
                except Exception as error:
                    failures.append(_safe_failure(city, country, error))

            cleanup_result = repository.cleanup_old_observations(
                connection,
                settings.openweather_retention_days,
                settings.openweather_max_rows,
            )
            status = _ingestion_status(records_inserted, len(failures))
            repository.register_ingestion_run(
                connection,
                status=status,
                records_inserted=records_inserted,
                records_failed=len(failures),
                error_message=_format_failure_summary(failures),
            )
    except Exception:
        print(POSTGRES_CONNECTION_ERROR_MESSAGE)
        return {"exit_code": 1}

    deleted_total = (
        cleanup_result["deleted_by_retention"]
        + cleanup_result["deleted_by_max_rows"]
    )
    print(f"Cidades processadas: {len(cities)}")
    print(f"Registros inseridos: {records_inserted}")
    print(f"Falhas: {len(failures)}")
    if failures:
        print("Falhas por cidade: " + _format_failure_summary(failures))
    print(f"Limpeza concluida: {deleted_total} registros antigos removidos.")
    print(f"Registros restantes: {cleanup_result['remaining_rows']}")

    return {
        "exit_code": 0 if records_inserted > 0 else 1,
        "cities_processed": len(cities),
        "records_inserted": records_inserted,
        "records_failed": len(failures),
        "failures": failures,
        "cleanup": cleanup_result,
    }


def _safe_failure(city: str, country: str, error: Exception) -> dict[str, str]:
    return {
        "city": city,
        "country": country,
        "error_type": error.__class__.__name__,
    }


def _format_failure_summary(failures: list[dict[str, str]]) -> str | None:
    if not failures:
        return None
    return "; ".join(
        f"{failure['city']}/{failure['country']} ({failure['error_type']})"
        for failure in failures
    )


def _ingestion_status(records_inserted: int, records_failed: int) -> str:
    if records_failed == 0:
        return "success"
    if records_inserted > 0:
        return "partial_failed"
    return "failed"


def main() -> int:
    args = parse_args()
    settings = load_settings()
    cities = resolve_collection_cities(settings, args.city, args.country)
    result = collect_openweather_snapshots(settings, cities)
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
