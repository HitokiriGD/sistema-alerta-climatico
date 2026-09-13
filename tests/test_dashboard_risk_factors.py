import inspect

from app import streamlit_app
from app.streamlit_app import build_associated_risk_factors
from app.streamlit_app import build_main_diagnosis_text
from app.streamlit_app import build_main_result_summary
from app.streamlit_app import build_methodological_detail_text
from app.streamlit_app import build_primary_event_label
from src.alerts.risk_classifier import WeatherRisk


PROHIBITED_LANGUAGE = (
    "vai acontecer",
    "desastre confirmado",
    "incendio confirmado",
    "alerta oficial",
    "evacuacao",
    "ordem publica",
)


def make_heat_risk() -> WeatherRisk:
    return WeatherRisk(
        risk_level="alto",
        event_type="calor_extremo",
        reason="Classificacao alta para calor extremo.",
        triggered_rules=[
            "calor_extremo_alto (alto): Temperatura elevada.",
            "vento_forte_moderado (moderado): Vento requer atencao.",
        ],
        variables={
            "temperature": 36.0,
            "feels_like": 41.0,
            "humidity": 22.0,
            "precipitation": 0.0,
            "wind_speed": 32.0,
            "pressure": 1004.0,
        },
        recommendations=[],
    )


def make_manaus_anomalies() -> list[dict[str, object]]:
    return [
        {
            "anomaly_type": "HIGH_TEMPERATURE",
            "severity": "HIGH",
            "reason": "Temperatura acima do padrao historico.",
        },
        {
            "anomaly_type": "HIGH_WIND",
            "severity": "HIGH",
            "reason": "Vento acima do padrao historico.",
        },
        {
            "anomaly_type": "PRESSURE_ANOMALY",
            "severity": "MEDIUM",
            "reason": "Pressao fora do padrao historico.",
        },
        {
            "anomaly_type": "POTENTIAL_FIRE_RISK",
            "severity": "HIGH",
            "reason": "Condicao favoravel a risco potencial de incendio.",
        },
    ]


def sentence_count(text: str) -> int:
    return len([part for part in text.split(".") if part.strip()])


def build_heat_summary() -> dict[str, object]:
    return build_main_result_summary(
        {
            "available": True,
            "prediction": "alto",
            "model_name": "random_forest",
            "selection_metric": "f1_macro",
        },
        make_heat_risk(),
        {"has_historical_data": True, "anomalies": make_manaus_anomalies()},
        weather_data=make_heat_risk().variables,
    )


def test_associated_factors_include_wind_pressure_and_fire_condition() -> None:
    factors = build_associated_risk_factors(
        "calor_extremo",
        triggered_rules=make_heat_risk().triggered_rules,
        anomalies=make_manaus_anomalies(),
        variables=make_heat_risk().variables,
        prediction={"available": True, "prediction": "alto"},
    )

    labels = {factor["label"] for factor in factors}
    assert "Vento acima do padrão histórico" in labels
    assert "Pressão atmosférica fora do padrão histórico" in labels
    assert "Condição favorável a risco potencial de incêndio" in labels
    assert all(
        set(factor) == {"label", "category", "severity", "source", "description"}
        for factor in factors
    )


def test_fire_condition_is_factor_and_does_not_replace_heat_event() -> None:
    summary = build_heat_summary()

    assert summary["primary_event_label"] == "Estresse térmico por calor"
    assert build_primary_event_label("calor_extremo") == "Estresse térmico por calor"
    assert any(
        factor["category"] == "incendio"
        for factor in summary["associated_factors"]
    )


def test_associated_factors_do_not_duplicate_primary_event() -> None:
    summary = build_heat_summary()

    assert all(
        factor["category"] != "calor"
        for factor in summary["associated_factors"]
    )
    assert len({factor["category"] for factor in summary["associated_factors"]}) == len(
        summary["associated_factors"]
    )


def test_executive_text_mentions_factors_and_stays_short() -> None:
    text = build_main_diagnosis_text(build_heat_summary())

    assert sentence_count(text) <= 4
    assert "estresse térmico por calor" in text
    assert "fatores associados" in text
    assert "vento acima do padrão histórico" in text.lower()
    assert "risco potencial de incêndio" in text.lower()


def test_executive_text_omits_complete_variables_and_percentiles() -> None:
    text = build_main_diagnosis_text(build_heat_summary()).lower()

    for technical_term in (
        "temperature",
        "feels_like",
        "humidity",
        "precipitation",
        "wind_speed",
        "pressure",
        "percentil",
    ):
        assert technical_term not in text


def test_methodological_detail_separates_event_and_factors_safely() -> None:
    summary = build_heat_summary()
    detail = build_methodological_detail_text(
        summary,
        risk=make_heat_risk(),
        analysis_result={
            "has_historical_data": True,
            "anomalies": make_manaus_anomalies(),
        },
        weather_data=make_heat_risk().variables,
    )
    combined = f"{build_main_diagnosis_text(summary)} {detail}".lower()

    assert "evento principal" in detail.lower()
    assert "fatores associados" in detail.lower()
    assert "nao substituem o evento principal" in detail.lower()
    assert "nao afirma a ocorrencia real de desastre" in detail.lower()
    assert "condicao meteorologica favoravel" in detail.lower()
    for forbidden in PROHIBITED_LANGUAGE:
        assert forbidden not in combined


def test_result_layout_exposes_primary_event_and_associated_factors() -> None:
    source = inspect.getsource(streamlit_app.show_result_analysis_section)

    assert '"Evento principal"' in source
    assert '"associated_factors"' in source
    assert "_html_factor_chips" in source
