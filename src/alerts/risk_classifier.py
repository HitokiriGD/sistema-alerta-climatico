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
                "meteorologicos atuais informados."
            ),
            triggered_rules=[],
            variables=variables,
            recommendations=[
                "Acompanhar atualizacoes meteorologicas caso as condicoes mudem.",
                "Manter os dados atuais revisados antes de tomar decisoes.",
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
                f"Umidade relativa de {humidity:.1f}% esta em nivel critico.",
            )
        )
    elif humidity <= 20:
        rules.append(
            TriggeredRule(
                "baixa_umidade_alta",
                "alto",
                "baixa_umidade",
                f"Umidade relativa de {humidity:.1f}% esta muito baixa.",
            )
        )
    elif humidity <= 30:
        rules.append(
            TriggeredRule(
                "baixa_umidade_moderada",
                "moderado",
                "baixa_umidade",
                f"Umidade relativa de {humidity:.1f}% esta abaixo do ideal.",
            )
        )

    if temperature >= 40 or feels_like >= 45:
        rules.append(
            TriggeredRule(
                "calor_extremo_critico",
                "critico",
                "calor_extremo",
                (
                    f"Temperatura de {temperature:.1f} C e sensacao termica de "
                    f"{feels_like:.1f} C indicam calor extremo critico."
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
                    f"Temperatura de {temperature:.1f} C ou sensacao termica de "
                    f"{feels_like:.1f} C indicam calor elevado."
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
                    f"Temperatura de {temperature:.1f} C ou sensacao termica de "
                    f"{feels_like:.1f} C indicam calor moderado."
                ),
            )
        )

    if temperature <= 0:
        rules.append(
            TriggeredRule(
                "frio_intenso_critico",
                "critico",
                "frio_intenso",
                f"Temperatura de {temperature:.1f} C esta em faixa critica de frio.",
            )
        )
    elif temperature <= 5:
        rules.append(
            TriggeredRule(
                "frio_intenso_alto",
                "alto",
                "frio_intenso",
                f"Temperatura de {temperature:.1f} C esta muito baixa.",
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
                f"Precipitacao de {precipitation:.1f} mm indica chuva intensa.",
            )
        )
    elif precipitation >= 25:
        rules.append(
            TriggeredRule(
                "chuva_intensa_alta",
                "alto",
                "chuva_intensa",
                f"Precipitacao de {precipitation:.1f} mm esta elevada.",
            )
        )
    elif precipitation >= 10:
        rules.append(
            TriggeredRule(
                "chuva_intensa_moderada",
                "moderado",
                "chuva_intensa",
                f"Precipitacao de {precipitation:.1f} mm requer acompanhamento.",
            )
        )

    if wind_speed >= 75:
        rules.append(
            TriggeredRule(
                "vento_forte_critico",
                "critico",
                "vento_forte",
                f"Vento de {wind_speed:.1f} km/h esta em faixa critica.",
            )
        )
    elif wind_speed >= 50:
        rules.append(
            TriggeredRule(
                "vento_forte_alto",
                "alto",
                "vento_forte",
                f"Vento de {wind_speed:.1f} km/h esta forte.",
            )
        )
    elif wind_speed >= 30:
        rules.append(
            TriggeredRule(
                "vento_forte_moderado",
                "moderado",
                "vento_forte",
                f"Vento de {wind_speed:.1f} km/h requer atencao.",
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
                    "favorecem risco critico de incendio."
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
                    "vento favorecem risco de incendio."
                ),
            )
        )

    return rules


def _build_reason(main_rule: TriggeredRule, rules: list[TriggeredRule]) -> str:
    if len(rules) == 1:
        return (
            f"Classificacao {main_rule.risk_level} para "
            f"{main_rule.event_type}: {main_rule.reason}"
        )

    return (
        f"Classificacao {main_rule.risk_level} para {main_rule.event_type}, "
        f"pois esta foi a regra de maior severidade entre {len(rules)} regras "
        f"acionadas. Principal motivo: {main_rule.reason}"
    )


def _recommendations_for_event(event_type: str) -> list[str]:
    recommendations = {
        "baixa_umidade": [
            "Reforcar hidratacao e evitar esforco fisico intenso nos horarios secos.",
            "Acompanhar a evolucao da umidade nas proximas atualizacoes.",
        ],
        "calor_extremo": [
            "Evitar exposicao prolongada ao sol nos horarios mais quentes.",
            "Priorizar ambientes ventilados e ingestao regular de agua.",
        ],
        "frio_intenso": [
            "Planejar protecao termica adequada para atividades externas.",
            "Acompanhar atualizacoes de temperatura ao longo do dia.",
        ],
        "chuva_intensa": [
            "Evitar deslocamentos por areas sujeitas a alagamento.",
            "Acompanhar a previsao local antes de atividades externas.",
        ],
        "vento_forte": [
            "Evitar areas abertas ou proximas a estruturas instaveis.",
            "Revisar objetos soltos em areas externas.",
        ],
        "risco_incendio": [
            "Evitar queimas e descarte inadequado de material inflamavel.",
            "Acompanhar orientacoes locais sobre manejo de fogo.",
        ],
    }
    return recommendations.get(
        event_type,
        ["Acompanhar atualizacoes meteorologicas locais."],
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
