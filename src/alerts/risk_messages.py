from typing import Any


PROHIBITED_WARNING_TERMS = (
    "vai acontecer",
    "desastre confirmado",
    "alerta oficial",
    "evacuacao",
    "ordem publica",
    "previsao oficial",
)

RISK_LEVEL_MESSAGES = {
    "baixo": {
        "title": "Condição sem aviso relevante",
        "description": (
            "Condição sem risco climático relevante no momento. Recomenda-se "
            "acompanhamento normal."
        ),
        "action_level": "Acompanhamento normal",
        "public": (
            "Manter a rotina normal, acompanhar as atualizações meteorológicas e "
            "revisar a consulta se as condições mudarem."
        ),
        "authority": (
            "Manter o acompanhamento meteorológico de rotina e usar o resultado "
            "como apoio à decisão."
        ),
        "caution": (
            "Aviso interpretativo de apoio à decisão; não substitui comunicados "
            "de órgãos competentes."
        ),
    },
    "moderado": {
        "title": "Atenção meteorológica",
        "description": (
            "Atenção: há uma condição meteorológica que merece acompanhamento. "
            "Recomenda-se observar as atualizações e evitar exposição prolongada "
            "quando aplicável."
        ),
        "action_level": "Atenção e acompanhamento",
        "public": (
            "Acompanhar as atualizações meteorológicas, reduzir a exposição "
            "prolongada quando aplicável e observar mudanças locais."
        ),
        "authority": (
            "Acompanhar a evolução das variáveis, revisar os dados locais e manter "
            "as equipes responsáveis informadas quando necessário."
        ),
        "caution": (
            "Indicação preventiva baseada nos dados disponíveis; requer "
            "interpretação em conjunto com o contexto local."
        ),
    },
    "alto": {
        "title": "Atenção reforçada",
        "description": (
            "Atenção reforçada: o cenário apresenta sinais relevantes de risco "
            "climático potencial. Os responsáveis locais devem acompanhar a "
            "evolução e avaliar medidas preventivas."
        ),
        "action_level": "Atenção reforçada",
        "public": (
            "Acompanhar as atualizações com maior frequência, reduzir a exposição a "
            "condições adversas e priorizar medidas preventivas simples."
        ),
        "authority": (
            "Avaliar medidas preventivas, acompanhar as evidências locais e manter "
            "a comunicação de orientação preparada conforme os protocolos internos."
        ),
        "caution": (
            "O resultado indica risco potencial e não confirma a ocorrência de "
            "um evento; use-o como apoio à decisão."
        ),
    },
    "critico": {
        "title": "Aviso preventivo crítico",
        "description": (
            "Aviso preventivo: o cenário apresenta uma condição crítica de risco "
            "potencial. Responsáveis e autoridades devem avaliar a comunicação "
            "preventiva e as ações de mitigação conforme os protocolos institucionais."
        ),
        "action_level": "Prioridade preventiva",
        "public": (
            "Acompanhar as informações meteorológicas com atenção, evitar a exposição "
            "desnecessária à condição de risco e seguir as orientações dos órgãos "
            "competentes."
        ),
        "authority": (
            "Avaliar a comunicação preventiva, as medidas de mitigação e o "
            "monitoramento contínuo conforme os protocolos institucionais e os dados "
            "locais."
        ),
        "caution": (
            "Classificação preventiva de risco potencial; as decisões operacionais "
            "devem considerar protocolos e fontes institucionais."
        ),
    },
}

EVENT_GUIDANCE = {
    "sem_risco_relevante": {
        "summary": "Nenhum evento principal foi destacado pelas regras técnicas.",
        "public": "Manter o acompanhamento normal das atualizações meteorológicas.",
        "authority": "Manter o monitoramento de rotina e registrar a consulta como referência.",
    },
    "calor_extremo": {
        "summary": "Condição favorável a estresse térmico por calor.",
        "public": (
            "Reforçar a hidratação, evitar a exposição prolongada ao sol e dar "
            "atenção a crianças, idosos e pessoas vulneráveis."
        ),
        "authority": (
            "Acompanhar a temperatura e a sensação térmica, orientar os locais de maior "
            "exposição e revisar as medidas preventivas para grupos vulneráveis."
        ),
    },
    "baixa_umidade": {
        "summary": "Condição favorável a desconforto e atenção respiratória.",
        "public": (
            "Reforçar a hidratação, evitar atividade física intensa nos períodos "
            "mais secos e observar sintomas respiratórios."
        ),
        "authority": (
            "Acompanhar a umidade relativa, orientar as atividades externas e avaliar "
            "medidas preventivas em escolas, no trabalho externo e na saúde."
        ),
    },
    "chuva_intensa": {
        "summary": "Condição favorável a acumulados de chuva e pontos de alagamento.",
        "public": (
            "Acompanhar as atualizações, planejar os deslocamentos e evitar áreas com "
            "histórico de alagamento ou enxurrada."
        ),
        "authority": (
            "Monitorar os acumulados, as áreas sensíveis e as rotas de deslocamento, "
            "mantendo as equipes responsáveis em acompanhamento."
        ),
    },
    "vento_forte": {
        "summary": "Condição favorável a rajadas ou vento acima do padrão esperado.",
        "public": (
            "Revisar os objetos soltos, evitar a proximidade de estruturas frágeis e "
            "ter atenção durante os deslocamentos."
        ),
        "authority": (
            "Acompanhar a velocidade do vento, as estruturas temporárias e os locais "
            "com maior exposição, avaliando medidas preventivas."
        ),
    },
    "frio_intenso": {
        "summary": "Condição favorável a desconforto térmico por frio.",
        "public": (
            "Evitar a exposição prolongada ao frio, reforçar a proteção térmica e "
            "dar atenção aos grupos vulneráveis."
        ),
        "authority": (
            "Acompanhar a temperatura, a sensação térmica e a demanda social, avaliando "
            "medidas preventivas para populações vulneráveis."
        ),
    },
    "risco_incendio": {
        "summary": "Condição favorável a risco potencial de incêndio em vegetação seca.",
        "public": (
            "Evitar queimadas, descartar materiais inflamáveis com cuidado e "
            "acompanhar a umidade e o vento."
        ),
        "authority": (
            "Acompanhar a umidade, o vento e a vegetação seca, reforçando as orientações "
            "preventivas e o monitoramento de áreas sensíveis."
        ),
    },
}


def build_risk_action_level(risk_level: object) -> str:
    """Retorna o nivel de atencao associado ao risco."""
    return _level_message(risk_level)["action_level"]


def build_public_recommendation(
    risk_level: object,
    event_type: object | None = None,
) -> str:
    """Monta recomendacao preventiva para usuario ou populacao."""
    level_text = _level_message(risk_level)["public"]
    event_text = _event_message(event_type)["public"]
    return _join_unique_sentences([level_text, event_text])


def build_authority_recommendation(
    risk_level: object,
    event_type: object | None = None,
) -> str:
    """Monta recomendacao para responsaveis e autoridades."""
    level_text = _level_message(risk_level)["authority"]
    event_text = _event_message(event_type)["authority"]
    return _join_unique_sentences([level_text, event_text])


def build_event_specific_guidance(event_type: object | None) -> dict[str, str]:
    """Retorna orientacao especifica por tipo de evento."""
    event = _event_message(event_type)
    return {
        "event_type": _normalize_event(event_type),
        "summary": event["summary"],
        "public_recommendation": event["public"],
        "authority_recommendation": event["authority"],
    }


def build_risk_warning(
    risk_level: object,
    event_type: object | None = None,
) -> dict[str, str]:
    """Gera campos principais do aviso preventivo."""
    level = _level_message(risk_level)
    event = _event_message(event_type)
    title = level["title"]
    description = level["description"]
    if _normalize_event(event_type) != "sem_risco_relevante":
        description = f"{description} {event['summary']}"

    return {
        "risk_level": _normalize_level(risk_level),
        "event_type": _normalize_event(event_type),
        "title": title,
        "description": _sanitize_warning_text(description),
        "action_level": level["action_level"],
        "public_recommendation": build_public_recommendation(risk_level, event_type),
        "authority_recommendation": build_authority_recommendation(
            risk_level,
            event_type,
        ),
        "event_guidance": event["summary"],
        "caution": level["caution"],
    }


def build_warning_summary(
    risk_level: object,
    event_type: object | None = None,
    ml_risk: object | None = None,
    rule_risk: object | None = None,
    weather_data: dict[str, object] | None = None,
    anomaly_count: int = 0,
    main_anomaly: object | None = None,
    station_label: object | None = None,
) -> dict[str, str]:
    """Consolida aviso, evidencias e recomendacoes para o dashboard."""
    warning = build_risk_warning(risk_level, event_type)
    evidence = _build_evidence_text(
        ml_risk=ml_risk,
        rule_risk=rule_risk,
        weather_data=weather_data,
        anomaly_count=anomaly_count,
        main_anomaly=main_anomaly,
        station_label=station_label,
    )
    why = (
        f"{warning['description']} Evidências consideradas: {evidence}"
        if evidence
        else warning["description"]
    )
    practical = warning["public_recommendation"]

    return {
        **warning,
        "warning_text": warning["description"],
        "evidence_text": evidence or "Evidências complementares indisponíveis.",
        "why_text": _sanitize_warning_text(why),
        "practical_recommendation": practical,
        "summary_text": _sanitize_warning_text(
            f"{warning['title']}. {warning['description']} {practical}"
        ),
    }


def _level_message(risk_level: object) -> dict[str, str]:
    normalized = _normalize_level(risk_level)
    return RISK_LEVEL_MESSAGES.get(normalized, RISK_LEVEL_MESSAGES["baixo"])


def _event_message(event_type: object | None) -> dict[str, str]:
    normalized = _normalize_event(event_type)
    return EVENT_GUIDANCE.get(normalized, EVENT_GUIDANCE["sem_risco_relevante"])


def _normalize_level(risk_level: object) -> str:
    normalized = str(risk_level or "").strip().lower()
    if normalized in RISK_LEVEL_MESSAGES:
        return normalized
    return "baixo"


def _normalize_event(event_type: object | None) -> str:
    normalized = str(event_type or "").strip().lower()
    if normalized in EVENT_GUIDANCE:
        return normalized
    return "sem_risco_relevante"


def _build_evidence_text(
    ml_risk: object | None,
    rule_risk: object | None,
    weather_data: dict[str, object] | None,
    anomaly_count: int,
    main_anomaly: object | None,
    station_label: object | None,
) -> str:
    evidence_parts = []
    if ml_risk and str(ml_risk) != "indisponivel":
        evidence_parts.append(f"ML indicou risco {ml_risk}")
    if rule_risk and str(rule_risk) != "indisponivel":
        evidence_parts.append(f"regras indicaram risco {rule_risk}")

    variable_text = _format_weather_evidence(weather_data or {})
    if variable_text:
        evidence_parts.append(variable_text)

    if anomaly_count:
        anomaly_text = str(main_anomaly or "anomalia histórica").strip()
        evidence_parts.append(
            f"{anomaly_count} anomalia(s) histórica(s), com destaque para {anomaly_text}"
        )
    if station_label:
        evidence_parts.append(f"referência INMET: {station_label}")

    return _sanitize_warning_text("; ".join(evidence_parts))


def _format_weather_evidence(weather_data: dict[str, object]) -> str:
    labels = [
        ("temperature", "temperatura", "C"),
        ("feels_like", "sensação térmica", "°C"),
        ("humidity", "umidade", "%"),
        ("precipitation", "precipitação", "mm"),
        ("wind_speed", "vento", "km/h"),
    ]
    values = []
    for key, label, unit in labels:
        value = _optional_float(weather_data.get(key))
        if value is not None:
            values.append(f"{label} {_format_number(value)} {unit}")
    return ", ".join(values[:5])


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    return f"{value:.1f}"


def _join_unique_sentences(parts: list[str]) -> str:
    unique_parts = []
    for part in parts:
        if part and part not in unique_parts:
            unique_parts.append(part)
    return _sanitize_warning_text(" ".join(unique_parts))


def _sanitize_warning_text(text: str) -> str:
    sanitized = text
    replacements = {
        "alerta oficial": "comunicado institucional",
        "previsao oficial": "informacao meteorologica institucional",
        "desastre confirmado": "risco potencial",
        "vai acontecer": "pode ocorrer",
        "evacuacao": "acao preventiva",
        "ordem publica": "orientacao institucional",
    }
    for forbidden, replacement in replacements.items():
        sanitized = sanitized.replace(forbidden, replacement)
        sanitized = sanitized.replace(forbidden.capitalize(), replacement.capitalize())
    return sanitized
