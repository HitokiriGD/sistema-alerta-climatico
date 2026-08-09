from src.alerts.risk_classifier import WeatherRisk
from src.alerts.risk_classifier import classify_weather_risk


def make_weather_data(**overrides):
    weather_data = {
        "temperature": 24.0,
        "feels_like": 24.0,
        "humidity": 60.0,
        "precipitation": 0.0,
        "wind_speed": 8.0,
        "pressure": 1012.0,
    }
    weather_data.update(overrides)
    return weather_data


def test_classify_weather_risk_without_relevant_risk() -> None:
    risk = classify_weather_risk(make_weather_data())

    assert risk.risk_level == "baixo"
    assert risk.event_type == "sem_risco_relevante"
    assert risk.triggered_rules == []
    assert "Nenhuma regra" in risk.reason


def test_classify_weather_risk_for_moderate_low_humidity() -> None:
    risk = classify_weather_risk(make_weather_data(humidity=30.0))

    assert risk.risk_level == "moderado"
    assert risk.event_type == "baixa_umidade"
    assert any("baixa_umidade_moderada" in rule for rule in risk.triggered_rules)


def test_classify_weather_risk_for_critical_low_humidity() -> None:
    risk = classify_weather_risk(make_weather_data(humidity=12.0))

    assert risk.risk_level == "critico"
    assert risk.event_type == "baixa_umidade"
    assert any("baixa_umidade_critica" in rule for rule in risk.triggered_rules)


def test_classify_weather_risk_for_extreme_heat() -> None:
    risk = classify_weather_risk(make_weather_data(temperature=36.0))

    assert risk.risk_level == "alto"
    assert risk.event_type == "calor_extremo"
    assert "calor" in risk.reason


def test_classify_weather_risk_for_intense_cold() -> None:
    risk = classify_weather_risk(make_weather_data(temperature=4.0))

    assert risk.risk_level == "alto"
    assert risk.event_type == "frio_intenso"
    assert any("frio_intenso_alto" in rule for rule in risk.triggered_rules)


def test_classify_weather_risk_for_heavy_rain() -> None:
    risk = classify_weather_risk(make_weather_data(precipitation=25.0))

    assert risk.risk_level == "alto"
    assert risk.event_type == "chuva_intensa"
    assert any("chuva_intensa_alta" in rule for rule in risk.triggered_rules)


def test_classify_weather_risk_for_strong_wind() -> None:
    risk = classify_weather_risk(make_weather_data(wind_speed=50.0))

    assert risk.risk_level == "alto"
    assert risk.event_type == "vento_forte"
    assert any("vento_forte_alto" in rule for rule in risk.triggered_rules)


def test_classify_weather_risk_for_fire_risk() -> None:
    risk = classify_weather_risk(
        make_weather_data(
            humidity=18.0,
            temperature=31.0,
            feels_like=32.0,
            precipitation=0.0,
            wind_speed=16.0,
        )
    )

    assert risk.risk_level == "alto"
    assert risk.event_type == "risco_incendio"
    assert any("risco_incendio_alto" in rule for rule in risk.triggered_rules)


def test_classify_weather_risk_chooses_highest_risk_with_multiple_rules() -> None:
    risk = classify_weather_risk(
        make_weather_data(
            humidity=25.0,
            temperature=41.0,
            feels_like=46.0,
            precipitation=12.0,
            wind_speed=35.0,
        )
    )

    assert risk.risk_level == "critico"
    assert risk.event_type == "calor_extremo"
    assert len(risk.triggered_rules) == 4
    assert "maior severidade" in risk.reason


def test_classify_weather_risk_response_structure() -> None:
    risk = classify_weather_risk(make_weather_data())

    assert isinstance(risk, WeatherRisk)
    assert risk.risk_level in {"baixo", "moderado", "alto", "critico"}
    assert isinstance(risk.event_type, str)
    assert isinstance(risk.reason, str)
    assert isinstance(risk.triggered_rules, list)
    assert isinstance(risk.variables, dict)
    assert isinstance(risk.recommendations, list)
    assert set(risk.variables) == {
        "temperature",
        "feels_like",
        "humidity",
        "precipitation",
        "wind_speed",
        "pressure",
    }
