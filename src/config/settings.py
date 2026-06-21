from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuracoes basicas carregadas do ambiente."""

    openweather_api_key: str
    default_city: str
    default_country: str
    inmet_base_url: str
    openweather_base_url: str


def load_settings() -> Settings:
    """Carrega variaveis de ambiente usadas pelo projeto."""
    load_dotenv()

    return Settings(
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
        default_city=os.getenv("DEFAULT_CITY", "Brasilia"),
        default_country=os.getenv("DEFAULT_COUNTRY", "BR"),
        inmet_base_url=os.getenv("INMET_BASE_URL", "https://apitempo.inmet.gov.br"),
        openweather_base_url=os.getenv(
            "OPENWEATHER_BASE_URL",
            "https://api.openweathermap.org/data/2.5",
        ),
    )
