from typing import Any, Protocol


class WeatherClient(Protocol):
    """Contrato simples para clientes de dados meteorologicos."""

    def fetch_current_weather(self, city: str) -> dict[str, Any]:
        """Busca dados meteorologicos atuais para uma cidade."""


def normalize_weather_payload(payload: dict[str, Any]) -> dict[str, float]:
    """Normaliza campos meteorologicos comuns vindos de APIs externas."""
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    rain = payload.get("rain", {})
    wind_speed_kmh = float(wind.get("speed", 0.0)) * 3.6

    return {
        "temperature": float(main.get("temp", 0.0)),
        "feels_like": float(main.get("feels_like", 0.0)),
        "humidity": float(main.get("humidity", 0.0)),
        "precipitation": float(rain.get("1h", 0.0)),
        "wind_speed": wind_speed_kmh,
        "pressure": float(main.get("pressure", 0.0)),
    }
