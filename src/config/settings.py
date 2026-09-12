from dataclasses import dataclass
import os
from typing import Iterable

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
SENSITIVE_SETTING_NAMES = (
    "OPENWEATHER_API_KEY",
    "GITHUB_TOKEN",
    "DATABASE_URL",
    "INMET_DATABASE_URL",
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


def _read_streamlit_secret(key: str) -> str:
    """Le uma chave do st.secrets quando o app estiver no Streamlit Cloud."""
    try:
        import streamlit as st

        value = st.secrets.get(key, "")
    except Exception:
        return ""

    if value is None:
        return ""
    return str(value)


def _get_config_value(key: str, default: str = "") -> str:
    """Busca configuracao em variavel de ambiente ou secrets do Streamlit."""
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value

    secret_value = _read_streamlit_secret(key)
    if secret_value:
        return secret_value

    return default


def sanitize_sensitive_text(
    text: object,
    sensitive_values: Iterable[object] = (),
) -> str:
    """Oculta valores sensiveis antes de exibir mensagens ao usuario."""
    sanitized_text = str(text)
    values_to_hide = [
        os.getenv(name, "")
        for name in SENSITIVE_SETTING_NAMES
        if os.getenv(name, "")
    ]
    values_to_hide.extend(str(value) for value in sensitive_values if value)

    for value in values_to_hide:
        if len(value) < 4:
            continue
        sanitized_text = sanitized_text.replace(value, "[valor sensivel oculto]")

    return sanitized_text


def load_settings(load_dotenv_file: bool = True) -> Settings:
    """Carrega variaveis de ambiente usadas pelo projeto."""
    if load_dotenv_file:
        load_dotenv()

    return Settings(
        openweather_api_key=_get_config_value("OPENWEATHER_API_KEY", ""),
        default_city=_get_config_value("DEFAULT_CITY", "Brasilia"),
        default_country=_get_config_value("DEFAULT_COUNTRY", "BR"),
        openweather_base_url=_get_config_value(
            "OPENWEATHER_BASE_URL",
            "https://api.openweathermap.org/data/2.5",
        ),
        inmet_historical_zip_dir=_get_config_value(
            "INMET_HISTORICAL_ZIP_DIR",
            "data/raw/inmet/zips",
        ),
        inmet_historical_start_year=int(
            _get_config_value("INMET_HISTORICAL_START_YEAR", "2020")
        ),
        inmet_historical_end_year=int(
            _get_config_value("INMET_HISTORICAL_END_YEAR", "2026")
        ),
        inmet_processed_data_path=_get_config_value(
            "INMET_PROCESSED_DATA_PATH",
            "data/processed/inmet_hourly.parquet",
        ),
        inmet_station_catalog_path=_get_config_value(
            "INMET_STATION_CATALOG_PATH",
            "data/processed/inmet_station_catalog.csv",
        ),
        inmet_database_path=_get_config_value(
            "INMET_DATABASE_PATH",
            "data/processed/inmet_historical.duckdb",
        ),
        inmet_database_url=_get_config_value("INMET_DATABASE_URL", ""),
        inmet_database_release_repo=_get_config_value(
            "INMET_DATABASE_RELEASE_REPO",
            "HitokiriGD/sistema-alerta-climatico",
        ),
        inmet_database_release_tag=_get_config_value(
            "INMET_DATABASE_RELEASE_TAG",
            "inmet-db-v1",
        ),
        inmet_database_asset_name=_get_config_value(
            "INMET_DATABASE_ASSET_NAME",
            "inmet_historical.duckdb",
        ),
        inmet_database_sha256=_get_config_value(
            "INMET_DATABASE_SHA256",
            "2621a5ada2f5b1d2f598690a3868639a406c4efe13fd36dd076f0a08eaa6edbe",
        ),
        github_token=_get_config_value("GITHUB_TOKEN", ""),
        database_url=_get_config_value("DATABASE_URL", ""),
        openweather_collection_cities=parse_openweather_collection_cities(
            _get_config_value(
                "OPENWEATHER_COLLECTION_CITIES",
                DEFAULT_OPENWEATHER_COLLECTION_CITIES,
            )
        ),
        openweather_retention_days=int(
            _get_config_value("OPENWEATHER_RETENTION_DAYS", "180")
        ),
        openweather_max_rows=int(_get_config_value("OPENWEATHER_MAX_ROWS", "100000")),
        openweather_store_raw_payload=_parse_bool(
            _get_config_value("OPENWEATHER_STORE_RAW_PAYLOAD", "false")
        ),
    )
