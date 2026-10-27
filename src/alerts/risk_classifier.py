from dataclasses import dataclass
from typing import Any


RISK_LEVEL_SCORES = {
    "baixo": 0,
    "moderado": 1,
    "alto": 2,
    "critico": 3,
}
EVENT_PRIORITY = {
    "sem_risco_relevante": 0,
    "baixa_umidade": 1,
    "calor_extremo": 1,
    "frio_intenso": 1,
    "chuva_intensa": 1,
    "vento_forte": 1,
    "risco_incendio": 2,
}

WEATHER_VARIABLES = [
    "temperature",
    "feels_like",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
]


@dataclass(frozen=True)
class TriggeredRule:
    """Regra meteorologica acionada durante a classificacao."""

    name: str
    risk_level: str
    event_type: str
    reason: str


@dataclass(frozen=True)
class WeatherRisk:
    """Resultado explicavel da classificacao de risco climatico."""

    risk_level: str
    event_type: str
    reason: str
    triggered_rules: list[str]
    variables: dict[str, float]
    recommendations: list[str]

    @property
    def level(self) -> str:
        """Alias de compatibilidade para telas antigas."""
        return self.risk_level

    @property
    def event(self) -> str:
        """Alias de compatibilidade para telas antigas."""
        return self.event_type

    @property
    def message(self) -> str:
        """Alias de compatibilidade para telas antigas."""
        return self.reason


def classify_weather_risk(weather_data: dict[str, Any]) -> WeatherRisk:
    """Classifica risco climatico atual por regras heuristicas explicaveis."""
    variables = _extract_variables(weather_data)
    triggered_rules = _evaluate_rules(variables)

    if not triggered_rules:
        return WeatherRisk(
            risk_level="baixo",
            event_type="sem_risco_relevante",
            reason=(
                "Nenhuma regra de risco relevante foi ativada para os dados "
                "meteorológicos atuais informados."
            ),
            triggered_rules=[],
            variables=variables,
            recommendations=[
                "Acompanhar as atualizações meteorológicas caso as condições mudem.",
                "Manter os dados atuais revisados antes de tomar decisões.",
            ],
        )

    main_rule = max(
        triggered_rules,
        key=lambda rule: (
            RISK_LEVEL_SCORES[rule.risk_level],
            EVENT_PRIORITY[rule.event_type],
        ),
    )
    return WeatherRisk(
        risk_level=main_rule.risk_level,
        event_type=main_rule.event_type,
        reason=_build_reason(main_rule, triggered_rules),
        triggered_rules=[_format_triggered_rule(rule) for rule in triggered_rules],
        variables=variables,
        recommendations=_recommendations_for_event(main_rule.event_type),
    )


def _extract_variables(weather_data: dict[str, Any]) -> dict[str, float]:
    variables = {
        variable: _to_float(weather_data.get(variable, 0.0))
        for variable in WEATHER_VARIABLES
    }
    if "feels_like" not in weather_data or weather_data.get("feels_like") is None:
        variables["feels_like"] = variables["temperature"]

    return variables


def _evaluate_rules(variables: dict[str, float]) -> list[TriggeredRule]:
    rules = []

    humidity = variables["humidity"]
    temperature = variables["temperature"]
    feels_like = variables["feels_like"]
    precipitation = variables["precipitation"]
    wind_speed = variables["wind_speed"]

    if humidity <= 12:
        rules.append(
            TriggeredRule(
                "baixa_umidade_critica",
                "critico",
                "baixa_umidade",
                f"Umidade relativa de {humidity:.1f}% está em nível crítico.",
            )
        )
    elif humidity <= 20:
        rules.append(
            TriggeredRule(
                "baixa_umidade_alta",
                "alto",
                "baixa_umidade",
                f"Umidade relativa de {humidity:.1f}% está muito baixa.",
            )
        )
    elif humidity <= 30:
        rules.append(
            TriggeredRule(
                "baixa_umidade_moderada",
                "moderado",
                "baixa_umidade",
                f"Umidade relativa de {humidity:.1f}% está abaixo do ideal.",
            )
        )

    if temperature >= 40 or feels_like >= 45:
        rules.append(
            TriggeredRule(
                "calor_extremo_critico",
                "critico",
                "calor_extremo",
                (
                    f"Temperatura de {temperature:.1f} °C e sensação térmica de "
                    f"{feels_like:.1f} °C indicam calor extremo crítico."
                ),
            )
        )
    elif temperature >= 35 or feels_like >= 40:
        rules.append(
            TriggeredRule(
                "calor_extremo_alto",
                "alto",
                "calor_extremo",
                (
                    f"Temperatura de {temperature:.1f} °C ou sensação térmica de "
                    f"{feels_like:.1f} °C indicam calor elevado."
                ),
            )
        )
    elif temperature >= 32 or feels_like >= 35:
        rules.append(
            TriggeredRule(
                "calor_extremo_moderado",
                "moderado",
                "calor_extremo",
                (
                    f"Temperatura de {temperature:.1f} °C ou sensação térmica de "
                    f"{feels_like:.1f} °C indicam calor moderado."
                ),
            )
        )

    if temperature <= 0:
        rules.append(
            TriggeredRule(
                "frio_intenso_critico",
                "critico",
                "frio_intenso",
                f"Temperatura de {temperature:.1f} °C está em faixa crítica de frio.",
            )
        )
    elif temperature <= 5:
        rules.append(
            TriggeredRule(
                "frio_intenso_alto",
                "alto",
                "frio_intenso",
                f"Temperatura de {temperature:.1f} °C está muito baixa.",
            )
        )
    elif temperature <= 10:
        rules.append(
            TriggeredRule(
                "frio_intenso_moderado",
                "moderado",
                "frio_intenso",
                f"Temperatura de {temperature:.1f} C indica frio moderado.",
            )
        )

    if precipitation >= 50:
        rules.append(
            TriggeredRule(
                "chuva_intensa_critica",
                "critico",
                "chuva_intensa",
                f"Precipitação de {precipitation:.1f} mm indica chuva intensa.",
            )
        )
    elif precipitation >= 25:
        rules.append(
            TriggeredRule(
                "chuva_intensa_alta",
                "alto",
                "chuva_intensa",
                f"Precipitação de {precipitation:.1f} mm está elevada.",
            )
        )
    elif precipitation >= 10:
        rules.append(
            TriggeredRule(
                "chuva_intensa_moderada",
                "moderado",
                "chuva_intensa",
                f"Precipitação de {precipitation:.1f} mm requer acompanhamento.",
            )
        )

    if wind_speed >= 75:
        rules.append(
            TriggeredRule(
                "vento_forte_critico",
                "critico",
                "vento_forte",
                f"Vento de {wind_speed:.1f} km/h está em faixa crítica.",
            )
        )
    elif wind_speed >= 50:
        rules.append(
            TriggeredRule(
                "vento_forte_alto",
                "alto",
                "vento_forte",
                f"Vento de {wind_speed:.1f} km/h está forte.",
            )
        )
    elif wind_speed >= 30:
        rules.append(
            TriggeredRule(
                "vento_forte_moderado",
                "moderado",
                "vento_forte",
                f"Vento de {wind_speed:.1f} km/h requer atenção.",
            )
        )

    if (
        humidity <= 12
        and temperature >= 35
        and precipitation == 0
        and wind_speed >= 25
    ):
        rules.append(
            TriggeredRule(
                "risco_incendio_critico",
                "critico",
                "risco_incendio",
                (
                    "Umidade muito baixa, calor, ausencia de chuva e vento "
                    "favorecem risco crítico de incêndio."
                ),
            )
        )
    elif (
        humidity <= 20
        and temperature >= 30
        and precipitation == 0
        and wind_speed >= 15
    ):
        rules.append(
            TriggeredRule(
                "risco_incendio_alto",
                "alto",
                "risco_incendio",
                (
                    "Umidade baixa, temperatura elevada, ausencia de chuva e "
                    "vento favorecem risco de incêndio."
                ),
            )
        )

    return rules


def _build_reason(main_rule: TriggeredRule, rules: list[TriggeredRule]) -> str:
    if len(rules) == 1:
        return (
            f"Classificação {main_rule.risk_level} para "
            f"{main_rule.event_type}: {main_rule.reason}"
        )

    return (
        f"Classificação {main_rule.risk_level} para {main_rule.event_type}, "
        f"pois esta foi a regra de maior severidade entre {len(rules)} regras "
        f"acionadas. Principal motivo: {main_rule.reason}"
    )


def _recommendations_for_event(event_type: str) -> list[str]:
    recommendations = {
        "baixa_umidade": [
            "Reforçar a hidratação e evitar esforço físico intenso nos horários secos.",
            "Acompanhar a evolução da umidade nas próximas atualizações.",
        ],
        "calor_extremo": [
            "Evitar a exposição prolongada ao sol nos horários mais quentes.",
            "Priorizar ambientes ventilados e a ingestão regular de água.",
        ],
        "frio_intenso": [
            "Planejar a proteção térmica adequada para atividades externas.",
            "Acompanhar as atualizações de temperatura ao longo do dia.",
        ],
        "chuva_intensa": [
            "Evitar deslocamentos por áreas sujeitas a alagamento.",
            "Acompanhar a previsão local antes de atividades externas.",
        ],
        "vento_forte": [
            "Evitar áreas abertas ou próximas a estruturas instáveis.",
            "Revisar os objetos soltos em áreas externas.",
        ],
        "risco_incendio": [
            "Evitar queimadas e o descarte inadequado de material inflamável.",
            "Acompanhar as orientações locais sobre o manejo do fogo.",
        ],
    }
    return recommendations.get(
        event_type,
        ["Acompanhar as atualizações meteorológicas locais."],
    )


def _format_triggered_rule(rule: TriggeredRule) -> str:
    return f"{rule.name} ({rule.risk_level}): {rule.reason}"


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        text_value = str(value).strip().replace(",", ".")
        try:
            return float(text_value)
        except ValueError:
            return 0.0
