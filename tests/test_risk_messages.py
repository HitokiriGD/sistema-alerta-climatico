import pytest

from src.alerts.risk_messages import PROHIBITED_WARNING_TERMS
from src.alerts.risk_messages import build_authority_recommendation
from src.alerts.risk_messages import build_event_specific_guidance
from src.alerts.risk_messages import build_public_recommendation
from src.alerts.risk_messages import build_risk_action_level
from src.alerts.risk_messages import build_risk_warning
from src.alerts.risk_messages import build_warning_summary


RISK_LEVELS = ["baixo", "moderado", "alto", "critico"]
EVENT_TYPES = [
    "calor_extremo",
    "baixa_umidade",
    "chuva_intensa",
    "vento_forte",
    "frio_intenso",
    "risco_incendio",
    "sem_risco_relevante",
]


@pytest.mark.parametrize("risk_level", RISK_LEVELS)
def test_build_risk_warning_for_each_level(risk_level: str) -> None:
    warning = build_risk_warning(risk_level, "baixa_umidade")

    assert warning["risk_level"] == risk_level
    assert warning["title"]
    assert warning["description"]
    assert warning["action_level"]
    assert warning["public_recommendation"]
    assert warning["authority_recommendation"]
    assert warning["caution"]


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_build_event_specific_guidance_for_main_events(event_type: str) -> None:
    guidance = build_event_specific_guidance(event_type)

    assert guidance["event_type"] == event_type
    assert guidance["summary"]
    assert guidance["public_recommendation"]
    assert guidance["authority_recommendation"]


def test_public_recommendation_for_low_humidity_mentions_hydration() -> None:
    recommendation = build_public_recommendation("moderado", "baixa_umidade")

    assert "hidratacao" in recommendation
    assert "atividade fisica intensa" in recommendation


def test_authority_recommendation_for_fire_risk_mentions_monitoring() -> None:
    recommendation = build_authority_recommendation("alto", "risco_incendio")

    assert "umidade" in recommendation
    assert "vegetacao seca" in recommendation
    assert "monitoramento" in recommendation


def test_unknown_event_type_uses_safe_fallback() -> None:
    guidance = build_event_specific_guidance("evento_desconhecido")

    assert guidance["event_type"] == "sem_risco_relevante"
    assert "acompanhamento normal" in guidance["public_recommendation"]


def test_action_level_for_critical_risk() -> None:
    assert build_risk_action_level("critico") == "Prioridade preventiva"


def test_warning_summary_uses_evidence_without_absolute_language() -> None:
    summary = build_warning_summary(
        "alto",
        "calor_extremo",
        ml_risk="alto",
        rule_risk="alto",
        weather_data={
            "temperature": 36.0,
            "feels_like": 41.0,
            "humidity": 24.0,
            "precipitation": 0.0,
            "wind_speed": 18.0,
        },
        anomaly_count=1,
        main_anomaly="Temperatura alta",
        station_label="MANAUS - AM | A101",
    )

    combined_text = " ".join(str(value).lower() for value in summary.values())
    assert "ml indicou risco alto" in combined_text
    assert "temperatura 36.0 c" in combined_text
    assert "manaus - am | a101" in combined_text
    for forbidden in PROHIBITED_WARNING_TERMS:
        assert forbidden not in combined_text
