from typing import Any
import requests
import logging
from src.config.settings import Settings

logger = logging.getLogger(__name__)

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

    def get_current_weather(self, station_code: str = "A001") -> dict[str, float] | None:
        """Busca e normaliza os dados da estação atual."""
        # Usa o método genérico criado pelo seu colega
        endpoint = f"estacao/dados/real/{station_code}"
        
        try:
            data = self.fetch_station_data(endpoint)
            
            if not data:
                logger.warning("INMET: Resposta vazia da API.")
                return None

            # Retorna o primeiro registro normalizado
            return self._normalize_data(data[0])

        except requests.exceptions.RequestException as e:
            logger.error(f"INMET: Erro de conexão - {e}")
            return None
        except ValueError:
            logger.error("INMET: Erro ao processar o JSON retornado.")
            return None

    def _normalize_data(self, raw_data: dict[str, Any]) -> dict[str, float]:
        """Converte os dados brutos do INMET para o padrão do sistema."""
        def parse_float(value: Any) -> float:
            try:
                return float(value) if value is not None else 0.0
            except ValueError:
                return 0.0

        return {
            "temperature": parse_float(raw_data.get("TEM_INS")),
            "feels_like": 0.0,  # Depende de ajuste posterior
            "humidity": parse_float(raw_data.get("UMD_INS")),
            "precipitation": parse_float(raw_data.get("CHUVA")),
            "wind_speed": parse_float(raw_data.get("VEN_VEL")),
            "pressure": parse_float(raw_data.get("PRE_INS"))
        }