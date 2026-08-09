from dataclasses import dataclass
import os

from dotenv import load_dotenv


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
    )
