from typing import Any, Protocol


class WeatherClient(Protocol):
    """Contrato simples para clientes de dados meteorologicos."""

    def fetch_current_weather(self, city: str) -> dict[str, Any]:
        """Busca dados meteorologicos atuais para uma cidade."""


def normalize_weather_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normaliza campos meteorologicos comuns vindos de APIs externas.

    Em payloads da OpenWeather, ``main.pressure`` representa a pressao ao nivel
    do mar. O campo legado ``pressure`` e mantido com esse mesmo significado
    para compatibilidade com telas e testes antigos.
    """
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    rain = payload.get("rain", {})
    wind_speed_kmh = float(wind.get("speed", 0.0)) * 3.6
    pressure_sea_level_hpa = float(main.get("pressure", 0.0))
    pressure_station_hpa = (
        float(main["grnd_level"])
        if main.get("grnd_level") is not None
        else None
    )

    weather_data: dict[str, Any] = {
        "temperature": float(main.get("temp", 0.0)),
        "feels_like": float(main.get("feels_like", 0.0)),
        "humidity": float(main.get("humidity", 0.0)),
        "precipitation": float(rain.get("1h", 0.0)),
        "wind_speed": wind_speed_kmh,
        "pressure": pressure_sea_level_hpa,
        "pressure_sea_level_hpa": pressure_sea_level_hpa,
    }

    if pressure_station_hpa is not None:
        weather_data["pressure_station_hpa"] = pressure_station_hpa
        weather_data["pressure_reference"] = "openweather_grnd_level"

    coord = payload.get("coord", {})
    if coord.get("lat") is not None and coord.get("lon") is not None:
        weather_data["latitude"] = float(coord["lat"])
        weather_data["longitude"] = float(coord["lon"])
    if payload.get("name"):
        weather_data["city"] = str(payload["name"])

    return weather_data
