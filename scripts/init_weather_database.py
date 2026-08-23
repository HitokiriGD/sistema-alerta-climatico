import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.config.settings import load_settings
from src.data import weather_repository


POSTGRES_CONNECTION_ERROR_MESSAGE = (
    "Erro ao conectar ao banco Postgres. Verifique DATABASE_URL no .env."
)


def initialize_weather_database(
    settings: Settings,
    repository=weather_repository,
) -> int:
    """Inicializa tabelas e cidades monitoradas no Postgres."""
    if not settings.database_url.strip():
        print("Configure DATABASE_URL no .env.")
        return 1

    try:
        with repository.connect_database(settings.database_url) as connection:
            repository.create_tables(connection)
            city_count = repository.upsert_monitored_cities(
                connection,
                settings.openweather_collection_cities,
            )
    except Exception:
        print(POSTGRES_CONNECTION_ERROR_MESSAGE)
        return 1

    print("Banco inicializado com sucesso.")
    print(f"Cidades monitoradas configuradas: {city_count}")
    return 0


def main() -> int:
    settings = load_settings()
    return initialize_weather_database(settings)


if __name__ == "__main__":
    raise SystemExit(main())
