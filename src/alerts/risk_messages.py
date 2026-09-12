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
        "title": "Condicao sem aviso relevante",
        "description": (
            "Condicao sem risco climatico relevante no momento. Recomenda-se "
            "acompanhamento normal."
        ),
        "action_level": "Acompanhamento normal",
        "public": (
            "Manter rotina normal, acompanhar atualizacoes meteorologicas e "
            "revisar a consulta se as condicoes mudarem."
        ),
        "authority": (
            "Manter acompanhamento meteorologico de rotina e usar o resultado "
            "como apoio a decisao."
        ),
        "caution": (
            "Aviso interpretativo de apoio a decisao; nao substitui comunicados "
            "de orgaos competentes."
        ),
    },
    "moderado": {
        "title": "Atencao meteorologica",
        "description": (
            "Atencao: ha condicao meteorologica que merece acompanhamento. "
            "Recomenda-se observar atualizacoes e evitar exposicao prolongada "
            "quando aplicavel."
        ),
        "action_level": "Atencao e acompanhamento",
        "public": (
            "Acompanhar atualizacoes meteorologicas, reduzir exposicao "
            "prolongada quando aplicavel e observar mudancas locais."
        ),
        "authority": (
            "Acompanhar a evolucao das variaveis, revisar dados locais e manter "
            "equipes responsaveis informadas quando necessario."
        ),
        "caution": (
            "Indicacao preventiva baseada nos dados disponiveis; requer "
            "interpretacao junto ao contexto local."
        ),
    },
    "alto": {
        "title": "Atencao reforcada",
        "description": (
            "Atencao reforcada: o cenario apresenta sinais relevantes de risco "
            "climatico potencial. Responsaveis locais devem acompanhar a "
            "evolucao e avaliar medidas preventivas."
        ),
        "action_level": "Atencao reforcada",
        "public": (
            "Acompanhar atualizacoes com maior frequencia, reduzir exposicao a "
            "condicoes adversas e priorizar medidas preventivas simples."
        ),
        "authority": (
            "Avaliar medidas preventivas, acompanhar evidencias locais e manter "
            "comunicacao de orientacao preparada conforme protocolos internos."
        ),
        "caution": (
            "O resultado indica risco potencial e nao confirma ocorrencia de "
            "evento; use como apoio a decisao."
        ),
    },
    "critico": {
        "title": "Aviso preventivo critico",
        "description": (
            "Aviso preventivo: o cenario apresenta condicao critica de risco "
            "potencial. Responsaveis e autoridades devem avaliar comunicacao "
            "preventiva e acoes de mitigacao conforme protocolos institucionais."
        ),
        "action_level": "Prioridade preventiva",
        "public": (
            "Acompanhar informacoes meteorologicas com atencao, evitar exposicao "
            "desnecessaria a condicao de risco e seguir orientacoes de orgaos "
            "competentes."
        ),
        "authority": (
            "Avaliar comunicacao preventiva, medidas de mitigacao e "
            "monitoramento continuo conforme protocolos institucionais e dados "
            "locais."
        ),
        "caution": (
            "Classificacao preventiva de risco potencial; decisoes operacionais "
            "devem considerar protocolos e fontes institucionais."
        ),
    },
}

EVENT_GUIDANCE = {
    "sem_risco_relevante": {
        "summary": "Nenhum evento principal foi destacado pelas regras tecnicas.",
        "public": "Manter acompanhamento normal das atualizacoes meteorologicas.",
        "authority": "Manter monitoramento de rotina e registrar a consulta como referencia.",
    },
    "calor_extremo": {
        "summary": "Condicao favoravel a estresse termico por calor.",
        "public": (
            "Reforcar hidratacao, evitar exposicao prolongada ao sol e dar "
            "atencao a criancas, idosos e pessoas vulneraveis."
        ),
        "authority": (
            "Acompanhar temperatura e sensacao termica, orientar locais de maior "
            "exposicao e revisar medidas preventivas para grupos vulneraveis."
        ),
    },
    "baixa_umidade": {
        "summary": "Condicao favoravel a desconforto e atencao respiratoria.",
        "public": (
            "Reforcar hidratacao, evitar atividade fisica intensa nos periodos "
            "mais secos e observar sintomas respiratorios."
        ),
        "authority": (
            "Acompanhar umidade relativa, orientar atividades externas e avaliar "
            "medidas preventivas em escolas, trabalho externo e saude."
        ),
    },
    "chuva_intensa": {
        "summary": "Condicao favoravel a acumulados de chuva e pontos de alagamento.",
        "public": (
            "Acompanhar atualizacoes, planejar deslocamentos e evitar areas com "
            "historico de alagamento ou enxurrada."
        ),
        "authority": (
            "Monitorar acumulados, areas sensiveis e rotas de deslocamento, "
            "mantendo equipes responsaveis em acompanhamento."
        ),
    },
    "vento_forte": {
        "summary": "Condicao favoravel a rajadas ou vento acima do padrao esperado.",
        "public": (
            "Revisar objetos soltos, evitar proximidade de estruturas frageis e "
            "ter atencao em deslocamentos."
        ),
        "authority": (
            "Acompanhar velocidade do vento, estruturas temporarias e locais com "
            "maior exposicao, avaliando medidas preventivas."
        ),
    },
    "frio_intenso": {
        "summary": "Condicao favoravel a desconforto termico por frio.",
        "public": (
            "Evitar exposicao prolongada ao frio, reforcar protecao termica e "
            "dar atencao a grupos vulneraveis."
        ),
        "authority": (
            "Acompanhar temperatura, sensacao termica e demanda social, avaliando "
            "medidas preventivas para populacoes vulneraveis."
        ),
    },
    "risco_incendio": {
        "summary": "Condicao favoravel a risco potencial de incendio em vegetacao seca.",
        "public": (
            "Evitar queimadas, descartar materiais inflamaveis com cuidado e "
            "acompanhar umidade e vento."
        ),
        "authority": (
            "Acompanhar umidade, vento e vegetacao seca, reforcando orientacoes "
            "preventivas e monitoramento de areas sensiveis."
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
        f"{warning['description']} Evidencias consideradas: {evidence}"
        if evidence
        else warning["description"]
    )
    practical = warning["public_recommendation"]

    return {
        **warning,
        "warning_text": warning["description"],
        "evidence_text": evidence or "Evidencias complementares indisponiveis.",
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
        anomaly_text = str(main_anomaly or "anomalia historica").strip()
        evidence_parts.append(
            f"{anomaly_count} anomalia(s) historica(s), com destaque para {anomaly_text}"
        )
    if station_label:
        evidence_parts.append(f"referencia INMET: {station_label}")

    return _sanitize_warning_text("; ".join(evidence_parts))


def _format_weather_evidence(weather_data: dict[str, object]) -> str:
    labels = [
        ("temperature", "temperatura", "C"),
        ("feels_like", "sensacao termica", "C"),
        ("humidity", "umidade", "%"),
        ("precipitation", "precipitacao", "mm"),
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
