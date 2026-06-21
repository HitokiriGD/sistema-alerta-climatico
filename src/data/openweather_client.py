from typing import Any

import requests

from src.config.settings import Settings


class OpenWeatherClient:
    """Cliente simples para a API OpenWeather."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch_current_weather(self, city: str) -> dict[str, Any]:
        """Busca dados atuais por cidade usando unidades metricas."""
        if not self.settings.openweather_api_key:
            raise ValueError("OPENWEATHER_API_KEY nao foi configurada.")

        response = requests.get(
            f"{self.settings.openweather_base_url}/weather",
            params={
                "q": f"{city},{self.settings.default_country}",
                "appid": self.settings.openweather_api_key,
                "units": "metric",
                "lang": "pt_br",
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
