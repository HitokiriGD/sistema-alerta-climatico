from typing import Any

import requests

from src.config.settings import Settings


class InmetClient:
    """Cliente simples para endpoints publicos do INMET."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch_station_data(self, endpoint: str) -> list[dict[str, Any]]:
        """Busca dados em um endpoint relativo do INMET."""
        url = f"{self.settings.inmet_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return data
        return [data]
