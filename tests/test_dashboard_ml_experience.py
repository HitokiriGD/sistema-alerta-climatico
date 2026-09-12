import json

from app.streamlit_app import build_intelligent_diagnosis_summary
from app.streamlit_app import build_divergence_message
from app.streamlit_app import build_evidence_summary
from app.streamlit_app import build_main_result_summary
from app.streamlit_app import build_model_comparison_table
from app.streamlit_app import build_result_explanation
from app.streamlit_app import build_risk_recommendation
from app.streamlit_app import extract_best_model_summary
from app.streamlit_app import format_probability_table
from app.streamlit_app import get_ml_prediction_for_dashboard
from app.streamlit_app import get_risk_badge_style
from app.streamlit_app import load_ml_evaluation_report
from app.streamlit_app import MODEL_SELECTION_EXPLANATION
from app.streamlit_app import MODEL_SELECTION_METRIC_NOTE
from app.streamlit_app import MODEL_SELECTION_TAB_LABEL
from src.alerts.risk_classifier import WeatherRisk


def make_report(best_model_name: str = "random_forest") -> dict[str, object]:
    return {
        "split_type": "temporal",
        "metrics_by_model": {
            "baseline_most_frequent": {
                "accuracy": 0.50,
                "precision_macro": 0.25,
                "recall_macro": 0.20,
                "f1_macro": 0.22,
            },
            "random_forest": {
                "accuracy": 0.91,
                "precision_macro": 0.88,
                "recall_macro": 0.84,
                "f1_macro": 0.86,
            },
        },
        "best_model_name": best_model_name,
        "selection_metric": "f1_macro",
        "baseline_comparison": {
            "baseline_score": 0.22,
            "best_model_score": 0.86,
            "absolute_difference": 0.64,
        },
    }


def make_risk() -> WeatherRisk:
    return WeatherRisk(
        risk_level="alto",
        event_type="baixa_umidade",
        reason="Classificacao alta por baixa umidade.",
        triggered_rules=["baixa_umidade_alta"],
        variables={"temperature": 34.0, "humidity": 24.0},
        recommendations=[],
    )


def make_analysis_result() -> dict[str, object]:
    return {
        "has_historical_data": True,
        "anomalies": [
            {
                "anomaly_type": "LOW_HUMIDITY",
                "reason": "Umidade abaixo do padrao historico.",
                "variables_used": ["humidity"],
            }
        ],
    }


def make_weather_data() -> dict[str, object]:
    return {
        "temperature": 34.0,
        "feels_like": 36.0,
        "humidity": 24.0,
        "precipitation": 0.0,
        "wind_speed": 18.0,
        "pressure_station_hpa": 1004.0,
    }


def test_load_ml_evaluation_report_prefers_temporal(tmp_path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "risk_level_robust_evaluation_random_report.json").write_text(
        json.dumps({"split_type": "random"}),
        encoding="utf-8",
    )
    (report_dir / "risk_level_robust_evaluation_temporal_report.json").write_text(
        json.dumps({"split_type": "temporal"}),
        encoding="utf-8",
    )

    result = load_ml_evaluation_report(report_dir)

    assert result["available"] is True
    assert result["report"]["split_type"] == "temporal"
    assert result["source"] == "avaliação robusta temporal"


def test_load_ml_evaluation_report_falls_back_to_random(tmp_path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "risk_level_robust_evaluation_random_report.json").write_text(
        json.dumps({"split_type": "random"}),
        encoding="utf-8",
    )

    result = load_ml_evaluation_report(report_dir)

    assert result["available"] is True
    assert result["report"]["split_type"] == "random"
    assert result["source"] == "avaliação robusta aleatória"


def test_load_ml_evaluation_report_returns_friendly_message_when_missing(tmp_path) -> None:
    result = load_ml_evaluation_report(tmp_path / "reports")

    assert result["available"] is False
    assert "Relatorio de avaliacao ML nao encontrado" in result["error_message"]
    assert "evaluate_ml_models.py" in result["error_message"]


def test_build_model_comparison_table_from_report() -> None:
    rows = build_model_comparison_table(make_report())

    assert rows == [
        {
            "Modelo": "baseline_most_frequent",
            "accuracy": 0.5,
            "precision_macro": 0.25,
            "recall_macro": 0.2,
            "f1_macro": 0.22,
        },
        {
            "Modelo": "random_forest",
            "accuracy": 0.91,
            "precision_macro": 0.88,
            "recall_macro": 0.84,
            "f1_macro": 0.86,
        },
    ]


def test_model_selection_section_text_explains_report_scope() -> None:
    assert MODEL_SELECTION_TAB_LABEL == "Como o modelo principal foi escolhido"
    assert "nao muda a cada consulta" in MODEL_SELECTION_EXPLANATION
    assert "conjunto de avaliacao usado no treinamento" in MODEL_SELECTION_EXPLANATION
    assert "melhor modelo salvo localmente" in MODEL_SELECTION_EXPLANATION
    assert "f1_macro, nao accuracy" in MODEL_SELECTION_METRIC_NOTE
    assert "classes desbalanceadas" in MODEL_SELECTION_METRIC_NOTE
    assert "classe majoritaria" in MODEL_SELECTION_METRIC_NOTE


def test_extract_best_model_summary() -> None:
    summary = extract_best_model_summary(make_report())

    assert summary["best_model_name"] == "random_forest"
    assert summary["selection_metric"] == "f1_macro"
    assert summary["baseline_f1_macro"] == 0.22
    assert summary["best_model_f1_macro"] == 0.86
    assert summary["absolute_difference"] == 0.64


def test_format_probability_table_orders_expected_classes() -> None:
    rows = format_probability_table(
        {
            "critico": 0.05,
            "baixo": 0.60,
            "alto": 0.10,
            "moderado": 0.25,
        }
    )

    assert [row["Classe"] for row in rows] == [
        "baixo",
        "moderado",
        "alto",
        "critico",
    ]
    assert rows[0]["Probabilidade (%)"] == "60.0%"


def test_build_intelligent_diagnosis_summary_with_anomaly() -> None:
    summary = build_intelligent_diagnosis_summary(
        {
            "available": True,
            "prediction": "alto",
        },
        make_risk(),
        {
            "has_historical_data": True,
            "anomalies": [
                {
                    "anomaly_type": "LOW_HUMIDITY",
                    "reason": "Umidade abaixo do padrao historico.",
                }
            ],
        },
    )

    assert summary["ml_risk"] == "alto"
    assert summary["rule_risk"] == "alto"
    assert "Umidade baixa" in summary["main_anomaly"]
    assert "modelo supervisionado" in summary["summary_text"]


def test_build_main_result_summary_prioritizes_ml_when_available() -> None:
    summary = build_main_result_summary(
        {
            "available": True,
            "prediction": "moderado",
            "model_name": "random_forest",
            "selection_metric": "f1_macro",
        },
        make_risk(),
        make_analysis_result(),
    )

    assert summary["final_risk"] == "moderado"
    assert summary["ml_risk"] == "moderado"
    assert summary["rule_risk"] == "alto"
    assert summary["event_type"] == "baixa_umidade"
    assert summary["attention_level"] == "Atencao e acompanhamento"
    assert "Atencao" in summary["warning_text"]
    assert "hidratacao" in summary["public_recommendation"]
    assert "autoridades" not in summary["warning_text"].lower()
    assert summary["anomaly_count"] == 1
    assert summary["model_name"] == "random_forest"


def test_build_main_result_summary_falls_back_to_rules_without_model() -> None:
    summary = build_main_result_summary(
        {"available": False, "prediction": None},
        make_risk(),
        None,
    )

    assert summary["final_risk"] == "alto"
    assert summary["ml_risk"] == "indisponivel"
    assert summary["model_name"] == "Nao disponivel"


def test_build_result_explanation_when_ml_and_rules_agree() -> None:
    explanation = build_result_explanation(
        {"available": True, "prediction": "alto"},
        make_risk(),
        make_analysis_result(),
    )

    assert "classificou a consulta como alto" in explanation
    assert "mesmo nivel de risco" in explanation
    assert "Umidade" in explanation


def test_build_result_explanation_when_ml_and_rules_diverge() -> None:
    explanation = build_result_explanation(
        {"available": True, "prediction": "moderado"},
        make_risk(),
        make_analysis_result(),
    )

    assert "indicou moderado" in explanation
    assert "indicaram alto" in explanation
    assert "limites diretos" in explanation


def test_build_divergence_message_without_model() -> None:
    message = build_divergence_message("indisponivel", "alto")

    assert "Machine Learning nao esta disponivel" in message
    assert "regras" in message


def test_build_risk_recommendation_for_all_levels() -> None:
    assert "rotina normal" in build_risk_recommendation("baixo")
    assert "Acompanhar atualizacoes" in build_risk_recommendation("moderado")
    assert "medidas preventivas" in build_risk_recommendation("alto")
    assert "exposicao desnecessaria" in build_risk_recommendation("critico")


def test_build_result_explanation_includes_warning_and_practical_guidance() -> None:
    explanation = build_result_explanation(
        {"available": True, "prediction": "alto"},
        make_risk(),
        make_analysis_result(),
        weather_data=make_weather_data(),
        selected_station={"station_label": "MANAUS - AM | A101"},
    )

    assert "Evidencias consideradas" in explanation
    assert "Recomendacao pratica" in explanation
    assert "MANAUS - AM | A101" in explanation
    assert "hidratacao" in explanation


def test_build_evidence_summary_contains_compact_fields() -> None:
    rows = build_evidence_summary(
        make_weather_data(),
        selected_station={
            "station_label": "MANAUS - AM | A101",
            "station_code": "A101",
        },
        requested_period="2020 a 2026",
        analysis_result=make_analysis_result(),
    )

    values_by_label = {row["label"]: row["value"] for row in rows}
    assert values_by_label["Temperatura"] == "34.0 C"
    assert values_by_label["Estacao INMET"] == "MANAUS - AM | A101"
    assert values_by_label["Periodo historico"] == "2020 a 2026"
    assert "Umidade baixa" in values_by_label["Principal anomalia"]


def test_get_risk_badge_style_returns_distinct_styles() -> None:
    low_style = get_risk_badge_style("baixo")
    critical_style = get_risk_badge_style("critico")
    unknown_style = get_risk_badge_style("sem_modelo")

    assert low_style["text"] != critical_style["text"]
    assert unknown_style == get_risk_badge_style("indisponivel")


def test_dashboard_helper_does_not_break_without_model_or_report(tmp_path) -> None:
    prediction = get_ml_prediction_for_dashboard(
        {"temperature": 30.0},
        model_bundle={
            "available": False,
            "error_message": "Modelo ML ainda nao encontrado localmente.",
        },
    )
    report = load_ml_evaluation_report(tmp_path / "missing_reports")

    assert prediction["available"] is False
    assert report["available"] is False
