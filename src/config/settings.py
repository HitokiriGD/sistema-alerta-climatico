from dataclasses import dataclass
import os

from dotenv import load_dotenv


BRAZILIAN_CAPITALS: tuple[tuple[str, str], ...] = (
    ("Rio Branco", "BR"),
    ("Maceio", "BR"),
    ("Macapa", "BR"),
    ("Manaus", "BR"),
    ("Salvador", "BR"),
    ("Fortaleza", "BR"),
    ("Brasilia", "BR"),
    ("Vitoria", "BR"),
    ("Goiania", "BR"),
    ("Sao Luis", "BR"),
    ("Cuiaba", "BR"),
    ("Campo Grande", "BR"),
    ("Belo Horizonte", "BR"),
    ("Belem", "BR"),
    ("Joao Pessoa", "BR"),
    ("Curitiba", "BR"),
    ("Recife", "BR"),
    ("Teresina", "BR"),
    ("Rio de Janeiro", "BR"),
    ("Natal", "BR"),
    ("Porto Alegre", "BR"),
    ("Porto Velho", "BR"),
    ("Boa Vista", "BR"),
    ("Florianopolis", "BR"),
    ("Sao Paulo", "BR"),
    ("Aracaju", "BR"),
    ("Palmas", "BR"),
)
DEFAULT_OPENWEATHER_COLLECTION_CITIES = ",".join(
    f"{city}:{country}" for city, country in BRAZILIAN_CAPITALS
)


@dataclass(frozen=True)
class Settings:
    """Configuracoes basicas carregadas do ambiente."""

    openweather_api_key: str
    default_city: str
    default_country: str
    openweather_base_url: str
    inmet_historical_zip_dir: str
    inmet_historical_start_year: int
    inmet_historical_end_year: int
    inmet_processed_data_path: str
    inmet_station_catalog_path: str
    inmet_database_path: str
    inmet_database_url: str
    inmet_database_release_repo: str
    inmet_database_release_tag: str
    inmet_database_asset_name: str
    inmet_database_sha256: str
    github_token: str
    database_url: str
    openweather_collection_cities: tuple[tuple[str, str], ...]
    openweather_retention_days: int
    openweather_max_rows: int
    openweather_store_raw_payload: bool


def parse_openweather_collection_cities(
    value: str,
) -> tuple[tuple[str, str], ...]:
    """Interpreta lista Cidade:PAIS usada na coleta OpenWeather."""
    raw_value = value.strip() or DEFAULT_OPENWEATHER_COLLECTION_CITIES
    cities: list[tuple[str, str]] = []

    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            city, country = item.rsplit(":", 1)
        else:
            city, country = item, "BR"
        city = city.strip()
        country = country.strip().upper() or "BR"
        if city:
            cities.append((city, country))

    return tuple(cities)


def _parse_bool(value: str, default: bool = False) -> bool:
    if not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "sim", "on"}


def load_settings(load_dotenv_file: bool = True) -> Settings:
    """Carrega variaveis de ambiente usadas pelo projeto."""
    if load_dotenv_file:
        load_dotenv()

    return Settings(
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
        default_city=os.getenv("DEFAULT_CITY", "Brasilia"),
        default_country=os.getenv("DEFAULT_COUNTRY", "BR"),
        openweather_base_url=os.getenv(
            "OPENWEATHER_BASE_URL",
            "https://api.openweathermap.org/data/2.5",
        ),
        inmet_historical_zip_dir=os.getenv(
            "INMET_HISTORICAL_ZIP_DIR",
            "data/raw/inmet/zips",
        ),
        inmet_historical_start_year=int(
            os.getenv("INMET_HISTORICAL_START_YEAR", "2020")
        ),
        inmet_historical_end_year=int(
            os.getenv("INMET_HISTORICAL_END_YEAR", "2026")
        ),
        inmet_processed_data_path=os.getenv(
            "INMET_PROCESSED_DATA_PATH",
            "data/processed/inmet_hourly.parquet",
        ),
        inmet_station_catalog_path=os.getenv(
            "INMET_STATION_CATALOG_PATH",
            "data/processed/inmet_station_catalog.csv",
        ),
        inmet_database_path=os.getenv(
            "INMET_DATABASE_PATH",
            "data/processed/inmet_historical.duckdb",
        ),
        inmet_database_url=os.getenv("INMET_DATABASE_URL", ""),
        inmet_database_release_repo=os.getenv(
            "INMET_DATABASE_RELEASE_REPO",
            "HitokiriGD/sistema-alerta-climatico",
        ),
        inmet_database_release_tag=os.getenv(
            "INMET_DATABASE_RELEASE_TAG",
            "inmet-db-v1",
        ),
        inmet_database_asset_name=os.getenv(
            "INMET_DATABASE_ASSET_NAME",
            "inmet_historical.duckdb",
        ),
        inmet_database_sha256=os.getenv(
            "INMET_DATABASE_SHA256",
            "2621a5ada2f5b1d2f598690a3868639a406c4efe13fd36dd076f0a08eaa6edbe",
        ),
        github_token=os.getenv("GITHUB_TOKEN", ""),
        database_url=os.getenv("DATABASE_URL", ""),
        openweather_collection_cities=parse_openweather_collection_cities(
            os.getenv(
                "OPENWEATHER_COLLECTION_CITIES",
                DEFAULT_OPENWEATHER_COLLECTION_CITIES,
            )
        ),
        openweather_retention_days=int(
            os.getenv("OPENWEATHER_RETENTION_DAYS", "180")
        ),
        openweather_max_rows=int(os.getenv("OPENWEATHER_MAX_ROWS", "100000")),
        openweather_store_raw_payload=_parse_bool(
            os.getenv("OPENWEATHER_STORE_RAW_PAYLOAD", "false")
        ),
    )
