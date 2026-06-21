from dataclasses import dataclass


@dataclass(frozen=True)
class WeatherRisk:
    """Resultado simples da classificacao de risco climatico."""

    level: str
    event: str
    message: str


def classify_weather_risk(weather_data: dict[str, float]) -> WeatherRisk:
    """Classifica riscos basicos por regras interpretaveis."""
    temperature = weather_data.get("temperature", 0.0)
    humidity = weather_data.get("humidity", 0.0)
    precipitation = weather_data.get("precipitation", 0.0)
    wind_speed = weather_data.get("wind_speed", 0.0)

    if precipitation >= 50:
        return WeatherRisk("alto", "chuva intensa", "Risco elevado de chuva intensa.")
    if wind_speed >= 60:
        return WeatherRisk("alto", "ventos extremos", "Risco elevado de ventos fortes.")
    if temperature >= 35 and humidity <= 30:
        return WeatherRisk(
            "medio",
            "calor e umidade critica",
            "Atencao para calor intenso.",
        )
    if temperature <= 5:
        return WeatherRisk("medio", "onda de frio", "Atencao para temperaturas baixas.")

    return WeatherRisk("baixo", "condicao normal", "Sem alerta climatico relevante.")
