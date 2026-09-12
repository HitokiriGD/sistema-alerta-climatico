import math
import sys
from dataclasses import replace
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st

from src.alerts.historical_analyzer import HistoricalAnalyzer
from src.alerts.risk_messages import build_authority_recommendation
from src.alerts.risk_messages import build_public_recommendation
from src.alerts.risk_messages import build_warning_summary
from src.alerts.risk_classifier import WeatherRisk
from src.alerts.risk_classifier import classify_weather_risk
from src.config.settings import Settings
from src.config.settings import load_settings
from src.config.settings import sanitize_sensitive_text
from src.data.inmet_client import InmetClient
from src.data.inmet_client import InmetHistoricalDataError
from src.data.inmet_client import normalize_city_name
from src.data.inmet_database import database_exists
from src.data.inmet_database import get_available_stations_from_database
from src.data.inmet_database import get_database_metadata
from src.data.inmet_database import load_station_history_from_database
from src.data.openweather_client import OpenWeatherClient
from src.data.weather_client import normalize_weather_payload
from src.ml.predict import METHODOLOGICAL_NOTE
from src.ml.predict import DEFAULT_METADATA_PATH
from src.ml.predict import DEFAULT_MODEL_PATH
from src.ml.predict import load_model_bundle
from src.ml.predict import model_files_available
from src.ml.predict import predict_risk_level


STANDARD_WEATHER_FIELDS = [
    "temperature",
    "feels_like",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
    "pressure_sea_level_hpa",
    "pressure_station_hpa",
]
CURRENT_WEATHER_DISPLAY = {
    "temperature": ("Temperatura", "C"),
    "feels_like": ("Sensacao termica", "C"),
    "humidity": ("Umidade", "%"),
    "precipitation": ("Precipitacao", "mm"),
    "wind_speed": ("Vento", "km/h"),
    "pressure_sea_level_hpa": ("Pressao nivel do mar", "hPa"),
    "pressure_station_hpa": ("Pressao nivel da estacao", "hPa"),
}
VARIABLE_DISPLAY = {
    "temperature": {
        "label": "Temperatura",
        "unit": "C",
        "percentile": "P95",
    },
    "humidity": {
        "label": "Umidade",
        "unit": "%",
        "percentile": "P10",
    },
    "precipitation": {
        "label": "Precipitacao",
        "unit": "mm",
        "percentile": "P95",
    },
    "wind_speed": {
        "label": "Vento",
        "unit": "km/h",
        "percentile": "P95",
    },
    "pressure": {
        "label": "Pressao",
        "unit": "hPa",
        "percentile": "P5 a P95",
    },
}
ANOMALY_LABELS = {
    "HIGH_TEMPERATURE": "Temperatura alta",
    "LOW_HUMIDITY": "Umidade baixa",
    "HIGH_PRECIPITATION": "Precipitacao alta",
    "HIGH_WIND": "Vento forte",
    "PRESSURE_ANOMALY": "Pressao atmosferica",
    "POTENTIAL_FIRE_RISK": "Risco potencial de incendio",
}
EVENT_LABELS = {
    "sem_risco_relevante": "sem risco relevante",
    "baixa_umidade": "baixa umidade",
    "calor_extremo": "calor extremo",
    "frio_intenso": "frio intenso",
    "chuva_intensa": "chuva intensa",
    "vento_forte": "vento forte",
    "risco_incendio": "risco de incendio",
}
ANOMALY_SUMMARY_TEXT = {
    "HIGH_TEMPERATURE": "a temperatura esta acima do padrao esperado",
    "LOW_HUMIDITY": "a umidade esta abaixo do padrao esperado",
    "HIGH_PRECIPITATION": "a precipitacao esta acima do padrao esperado",
    "HIGH_WIND": "o vento esta acima do padrao esperado",
    "PRESSURE_ANOMALY": "a pressao esta fora da faixa historica observada",
    "POTENTIAL_FIRE_RISK": (
        "a combinacao de calor, baixa umidade e vento favorece risco de incendio"
    ),
}
DEFAULT_HISTORICAL_START_YEAR = 2020
DEFAULT_HISTORICAL_END_YEAR = 2026
HISTORICAL_MIN_YEAR = 2000
HISTORICAL_MAX_YEAR = 2026
ML_REPORT_FILENAMES = [
    "risk_level_robust_evaluation_temporal_report.json",
    "risk_level_robust_evaluation_random_report.json",
    "risk_level_training_report.json",
]
ML_EVALUATION_COMMAND = (
    "python scripts/evaluate_ml_models.py --split temporal "
    "--train-end-year 2024 --test-start-year 2025"
)
MODEL_SELECTION_TAB_LABEL = "Como o modelo principal foi escolhido"
MODEL_SELECTION_EXPLANATION = (
    "Esta tabela nao muda a cada consulta. Ela mostra o desempenho dos "
    "algoritmos no conjunto de avaliacao usado no treinamento. O dashboard "
    "utiliza o melhor modelo salvo localmente para classificar a cidade "
    "consultada."
)
MODEL_SELECTION_METRIC_NOTE = (
    "A metrica principal de selecao e f1_macro, nao accuracy, porque o "
    "problema possui classes desbalanceadas e accuracy pode favorecer modelos "
    "que acertam apenas a classe majoritaria."
)
ML_DATASET_COMMAND = (
    "python scripts/build_ml_dataset.py --start-year 2020 --end-year 2026 "
    "--stations A001,A101,A312"
)
ML_TRAINING_COMMAND = "python scripts/train_ml_models.py"
INMET_DATABASE_DOWNLOAD_COMMAND = "python scripts/download_inmet_database.py"
RUNTIME_ARTIFACT_DIRS = (
    Path("data/processed"),
    Path("data/models"),
    Path("data/reports"),
)
METHODOLOGICAL_NOTE_SHORT = (
    "O modelo foi treinado com rotulos derivados de regras tecnicas. "
    "A previsao representa uma classificacao supervisionada aprendida, "
    "nao validacao contra eventos reais oficiais."
)
RISK_BADGE_STYLES = {
    "baixo": {
        "background": "rgba(20, 184, 166, 0.16)",
        "border": "rgba(45, 212, 191, 0.54)",
        "text": "#99f6e4",
    },
    "moderado": {
        "background": "rgba(245, 158, 11, 0.16)",
        "border": "rgba(251, 191, 36, 0.58)",
        "text": "#fde68a",
    },
    "alto": {
        "background": "rgba(249, 115, 22, 0.17)",
        "border": "rgba(251, 146, 60, 0.64)",
        "text": "#fed7aa",
    },
    "critico": {
        "background": "rgba(239, 68, 68, 0.18)",
        "border": "rgba(248, 113, 113, 0.70)",
        "text": "#fecaca",
    },
    "indisponivel": {
        "background": "rgba(148, 163, 184, 0.12)",
        "border": "rgba(148, 163, 184, 0.36)",
        "text": "#cbd5e1",
    },
}
RISK_RECOMMENDATIONS = {
    "baixo": "Monitorar normalmente e manter acompanhamento das atualizacoes.",
    "moderado": "Manter atencao e acompanhar atualizacoes meteorologicas.",
    "alto": (
        "Autoridades e responsaveis devem acompanhar com atencao reforcada."
    ),
    "critico": (
        "Autoridades e responsaveis devem avaliar medidas preventivas e "
        "alertas locais."
    ),
}


def ensure_runtime_artifact_directories(
    project_root: str | Path = PROJECT_ROOT,
) -> list[Path]:
    """Cria diretorios locais usados por artefatos gerados em runtime."""
    root = Path(project_root)
    created_or_existing_dirs = []
    for relative_dir in RUNTIME_ARTIFACT_DIRS:
        artifact_dir = root / relative_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        created_or_existing_dirs.append(artifact_dir)
    return created_or_existing_dirs


def sensitive_values_from_settings(settings: Settings) -> list[str]:
    """Lista valores sensiveis que nunca devem aparecer em mensagens."""
    return [
        settings.openweather_api_key,
        settings.github_token,
        settings.database_url,
        settings.inmet_database_url,
    ]


def safe_dashboard_message(message: object, settings: Settings) -> str:
    """Remove segredos conhecidos antes de exibir texto no dashboard."""
    return sanitize_sensitive_text(message, sensitive_values_from_settings(settings))


def build_deploy_artifact_guidance(
    settings: Settings,
    report_dir: str | Path = PROJECT_ROOT / "data/reports",
    model_path: str | Path = PROJECT_ROOT / DEFAULT_MODEL_PATH,
    metadata_path: str | Path = PROJECT_ROOT / DEFAULT_METADATA_PATH,
) -> list[str]:
    """Monta orientacoes curtas quando artefatos locais nao existem."""
    guidance = []

    if not database_exists(settings.inmet_database_path):
        guidance.append(
            "Base historica INMET DuckDB nao encontrada. Configure "
            "INMET_DATABASE_* e execute "
            f"{INMET_DATABASE_DOWNLOAD_COMMAND}, ou gere a base localmente com "
            "python scripts/build_inmet_duckdb.py."
        )

    if not model_files_available(model_path, metadata_path):
        guidance.append(
            "Modelo ML local nao encontrado. Gere o dataset, treine o modelo "
            f"com {ML_DATASET_COMMAND} e {ML_TRAINING_COMMAND}, ou publique os "
            "artefatos em uma release futuramente."
        )

    report_status = load_ml_evaluation_report(report_dir)
    if not report_status["available"]:
        guidance.append(
            "Relatorio de avaliacao ML nao encontrado. Depois de gerar dataset "
            f"e treinar o modelo, execute {ML_EVALUATION_COMMAND}."
        )

    return [safe_dashboard_message(message, settings) for message in guidance]


def show_deploy_artifact_guidance(settings: Settings) -> None:
    """Mostra orientacao amigavel para demos sem artefatos locais."""
    guidance = build_deploy_artifact_guidance(settings)
    if not guidance:
        return

    with st.container(border=True):
        st.markdown("#### Artefatos opcionais da demo")
        st.write(
            "O dashboard abre mesmo sem DuckDB, modelo ou relatorios locais. "
            "Para ativar a experiencia completa:"
        )
        for message in guidance:
            st.write(f"- {message}")


def build_manual_weather_data() -> dict[str, object]:
    """Coleta dados meteorologicos informados manualmente no dashboard."""
    temperature = st.number_input("Temperatura (C)", value=28.0, step=0.5)
    feels_like = st.number_input("Sensacao termica (C)", value=29.0, step=0.5)
    humidity = st.number_input(
        "Umidade relativa (%)",
        value=65.0,
        min_value=0.0,
        max_value=100.0,
    )
    precipitation = st.number_input(
        "Precipitacao (mm/h)",
        value=0.0,
        min_value=0.0,
    )
    wind_speed = st.number_input(
        "Velocidade do vento (km/h)",
        value=12.0,
        min_value=0.0,
    )
    pressure = st.number_input(
        "Pressao atmosferica (hPa)",
        value=1013.0,
        min_value=0.0,
    )

    return {
        "temperature": temperature,
        "feels_like": feels_like,
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "pressure_station_hpa": pressure,
        "pressure_reference": "manual_station_level",
        "data_source": "Entrada manual",
    }


def build_query_form(
    settings: Settings,
    compact: bool = False,
) -> tuple[dict[str, object], bool]:
    """Monta o formulario principal de consulta do dashboard."""
    with st.container(border=True):
        st.subheader("Nova consulta" if compact else "Consulta")
        if not compact:
            st.caption(
                "Informe a cidade e o periodo historico antes de buscar os dados."
            )
        with st.form("weather_query_form"):
            col_city, col_country = st.columns([3, 1])
            city = col_city.text_input("Cidade", value=settings.default_city)
            country = col_country.text_input(
                "Pais",
                value=settings.default_country,
                max_chars=2,
            )

            col_start, col_end, col_source = st.columns([1, 1, 1.4])
            start_year = col_start.number_input(
                "Ano inicial do historico",
                min_value=HISTORICAL_MIN_YEAR,
                max_value=HISTORICAL_MAX_YEAR,
                value=DEFAULT_HISTORICAL_START_YEAR,
                step=1,
            )
            end_year = col_end.number_input(
                "Ano final do historico",
                min_value=HISTORICAL_MIN_YEAR,
                max_value=HISTORICAL_MAX_YEAR,
                value=DEFAULT_HISTORICAL_END_YEAR,
                step=1,
            )
            data_source = col_source.selectbox(
                "Fonte dos dados atuais",
                ["OpenWeather", "Entrada manual"],
                index=0,
            )

            submitted = st.form_submit_button(
                "Buscar dados e comparar com historico",
                type="primary",
            )

    return (
        {
            "city": city,
            "country": country,
            "start_year": int(start_year),
            "end_year": int(end_year),
            "data_source": data_source,
        },
        submitted,
    )


def validate_historical_period(start_year: int, end_year: int) -> tuple[bool, str]:
    """Valida o periodo historico solicitado no formulario."""
    if start_year > end_year:
        return (
            False,
            "O ano inicial do historico deve ser menor ou igual ao ano final.",
        )
    if start_year < HISTORICAL_MIN_YEAR or end_year > HISTORICAL_MAX_YEAR:
        return (
            False,
            (
                "O periodo historico deve ficar entre "
                f"{HISTORICAL_MIN_YEAR} e {HISTORICAL_MAX_YEAR}."
            ),
        )
    return True, ""


def format_requested_period(start_year: int, end_year: int) -> str:
    """Formata o intervalo anual escolhido pelo usuario."""
    return f"{start_year} a {end_year}"


def fetch_openweather_weather_data(
    settings: Settings,
    city: str,
    country: str,
) -> dict[str, object] | None:
    """Coleta dados meteorologicos atuais usando a OpenWeather."""
    city = city.strip()
    country = country.strip().upper() or settings.default_country

    if not settings.openweather_api_key:
        st.error(
            "Configure OPENWEATHER_API_KEY no arquivo .env para usar a "
            "OpenWeather. A entrada manual continua disponivel."
        )
        return None

    if not city:
        st.error("Informe uma cidade para buscar os dados meteorologicos atuais.")
        return None

    try:
        client = OpenWeatherClient(replace(settings, default_country=country))
        payload = client.fetch_current_weather(city)
        weather_data = normalize_weather_payload(payload)
        weather_data["city"] = str(weather_data.get("city", city))
        weather_data["country"] = country
        weather_data["data_source"] = "OpenWeather"
        return weather_data
    except requests.HTTPError as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else "sem status"
        )
        st.error(
            "A OpenWeather retornou um erro "
            f"({status_code}). Verifique a cidade informada e a chave da API."
        )
    except requests.RequestException:
        st.error(
            "Nao foi possivel conectar a OpenWeather. Verifique sua conexao "
            "e tente novamente."
        )
    except ValueError as error:
        st.error(str(error))

    return None


def show_weather_data(weather_data: dict[str, object]) -> None:
    """Exibe os campos meteorologicos padronizados no dashboard."""
    city = str(weather_data.get("city", "-"))
    source = str(weather_data.get("data_source", "-"))
    col_city, col_source = st.columns(2)
    col_city.metric("Cidade consultada", city)
    col_source.metric("Fonte atual", source)

    metric_columns = st.columns(6)
    metric_columns[0].metric(
        "Temperatura",
        _format_value(weather_data.get("temperature"), "C"),
    )
    metric_columns[1].metric(
        "Sensacao termica",
        _format_value(weather_data.get("feels_like"), "C"),
    )
    metric_columns[2].metric(
        "Umidade",
        _format_value(weather_data.get("humidity"), "%"),
    )
    metric_columns[3].metric(
        "Chuva",
        _format_value(weather_data.get("precipitation"), "mm"),
    )
    metric_columns[4].metric(
        "Vento",
        _format_value(weather_data.get("wind_speed"), "km/h"),
    )
    pressure_value = weather_data.get(
        "pressure_station_hpa",
        weather_data.get("pressure_sea_level_hpa", weather_data.get("pressure")),
    )
    metric_columns[5].metric("Pressao", _format_value(pressure_value, "hPa"))
    st.caption(build_current_pressure_text(weather_data))

    with st.expander("Variaveis atuais padronizadas"):
        st.dataframe(
            pd.DataFrame(build_current_weather_rows(weather_data)),
            width="stretch",
            hide_index=True,
        )
    with st.expander("Variaveis brutas e campos tecnicos"):
        st.json(weather_data)


def show_weather_risk(risk: WeatherRisk) -> None:
    """Exibe o resultado explicavel do classificador por regras."""
    col_level, col_event = st.columns(2)
    col_level.metric("Nivel de risco", risk.risk_level.upper())
    col_event.metric("Evento principal", format_event_type(risk.event_type))
    _show_risk_reason(risk)
    st.caption(
        "As regras justificam tecnicamente a classificacao e tambem geraram "
        "os rotulos usados no treinamento supervisionado."
    )

    st.write("Recomendacoes gerais")
    for recommendation in risk.recommendations:
        st.write(f"- {recommendation}")

    st.write("Recomendacoes preventivas aprimoradas")
    st.write("- Usuario/populacao: " + build_public_recommendation(
        risk.risk_level,
        risk.event_type,
    ))
    st.write("- Autoridades/responsaveis: " + build_authority_recommendation(
        risk.risk_level,
        risk.event_type,
    ))
    st.caption(
        "Texto preventivo de apoio a decisao, baseado em dados disponiveis e "
        "sem substituir comunicados de orgaos competentes."
    )

    with st.expander("Regras acionadas"):
        if risk.triggered_rules:
            for rule in risk.triggered_rules:
                st.write(f"- {rule}")
        else:
            st.write("Nenhuma regra relevante foi acionada.")

    with st.expander("Variaveis usadas no alerta por regras"):
        st.dataframe(pd.DataFrame([risk.variables]), width="stretch", hide_index=True)


@st.cache_resource
def load_ml_model_bundle_for_app() -> dict[str, object]:
    """Carrega o modelo ML uma vez por sessao do Streamlit."""
    return load_model_bundle()


def get_ml_prediction_for_dashboard(
    weather_data: dict[str, object] | None,
    model_bundle: dict[str, object] | None = None,
) -> dict[str, object]:
    """Monta predicao ML sem quebrar o dashboard quando faltar modelo."""
    if not weather_data:
        return {
            "available": False,
            "prediction": None,
            "model_name": None,
            "selection_metric": None,
            "features_used": [],
            "missing_features": [],
            "observations": [],
            "methodological_note": METHODOLOGICAL_NOTE,
            "error_message": "Dados meteorologicos atuais ausentes para previsao ML.",
        }

    bundle = model_bundle or load_ml_model_bundle_for_app()
    return predict_risk_level(weather_data, model_bundle=bundle)


@st.cache_data
def load_ml_evaluation_report_for_app(report_dir: str = "data/reports") -> dict[str, Any]:
    """Carrega relatorio de avaliacao ML com cache do Streamlit."""
    return load_ml_evaluation_report(report_dir)


def load_ml_evaluation_report(report_dir: str | Path = "data/reports") -> dict[str, Any]:
    """Carrega o melhor relatorio local disponivel, priorizando temporal."""
    base_dir = Path(report_dir)
    for filename in ML_REPORT_FILENAMES:
        report_path = base_dir / filename
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {
                "available": False,
                "report": None,
                "report_path": str(report_path),
                "source": "",
                "error_message": f"Nao foi possivel ler o relatorio ML: {error}",
            }
        return {
            "available": True,
            "report": report,
            "report_path": str(report_path),
            "source": _report_source_from_filename(filename),
            "error_message": "",
        }

    return {
        "available": False,
        "report": None,
        "report_path": "",
        "source": "",
        "error_message": (
            "Relatorio de avaliacao ML nao encontrado. Depois de gerar o "
            "dataset e treinar o modelo, gere a avaliacao com "
            f"{ML_EVALUATION_COMMAND}."
        ),
    }


def build_model_comparison_table(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Monta tabela simples com metricas principais por modelo."""
    metrics_by_model = _metrics_by_model_from_report(report)
    rows = []
    for model_name, metrics in metrics_by_model.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "Modelo": model_name,
                "accuracy": _round_metric(metrics.get("accuracy")),
                "precision_macro": _round_metric(metrics.get("precision_macro")),
                "recall_macro": _round_metric(metrics.get("recall_macro")),
                "f1_macro": _round_metric(metrics.get("f1_macro")),
            }
        )
    return rows


def extract_best_model_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    """Extrai resumo do melhor modelo e comparacao com baseline."""
    if not report:
        return {
            "best_model_name": "",
            "selection_metric": "",
            "baseline_f1_macro": None,
            "best_model_f1_macro": None,
            "absolute_difference": None,
        }

    best_model_name = str(
        report.get("best_model_name")
        or _report_metadata(report).get("selected_model_name")
        or ""
    )
    selection_metric = str(
        report.get("selection_metric")
        or _report_metadata(report).get("metric_used")
        or "f1_macro"
    )
    metrics_by_model = _metrics_by_model_from_report(report)
    best_metrics = metrics_by_model.get(best_model_name, {})
    baseline_metrics = metrics_by_model.get("baseline_most_frequent", {})
    baseline_comparison = report.get("baseline_comparison", {})
    if not isinstance(baseline_comparison, dict):
        baseline_comparison = {}

    baseline_f1 = baseline_comparison.get("baseline_score")
    if baseline_f1 is None:
        baseline_f1 = baseline_metrics.get("f1_macro")
    best_f1 = baseline_comparison.get("best_model_score")
    if best_f1 is None:
        best_f1 = best_metrics.get("f1_macro")

    baseline_value = _optional_float(baseline_f1)
    best_value = _optional_float(best_f1)
    if baseline_value is None or best_value is None:
        difference = None
    else:
        difference = best_value - baseline_value

    return {
        "best_model_name": best_model_name,
        "selection_metric": selection_metric,
        "baseline_f1_macro": baseline_value,
        "best_model_f1_macro": best_value,
        "absolute_difference": difference,
    }


def format_probability_table(probabilities: dict[str, object] | None) -> list[dict[str, Any]]:
    """Formata probabilidades por classe em ordem de risco."""
    if not probabilities:
        return []

    ordered_labels = ["baixo", "moderado", "alto", "critico"]
    rows = []
    for label in ordered_labels:
        probability = _optional_float(probabilities.get(label))
        if probability is None:
            continue
        rows.append(
            {
                "Classe": label,
                "Probabilidade": round(probability, 4),
                "Probabilidade (%)": f"{probability * 100:.1f}%",
            }
        )
    return rows


def build_intelligent_diagnosis_summary(
    ml_prediction: dict[str, object] | None,
    risk: WeatherRisk | None,
    analysis_result: dict[str, object] | None,
) -> dict[str, str]:
    """Consolida predicao ML, regras e principal anomalia historica."""
    ml_risk = (
        str((ml_prediction or {}).get("prediction", ""))
        if (ml_prediction or {}).get("available")
        else ""
    )
    rule_risk = risk.risk_level if risk is not None else ""
    anomaly = _main_historical_anomaly_text(analysis_result)

    if ml_risk:
        summary = (
            f"O modelo supervisionado indica risco {ml_risk}. "
            f"As regras tecnicas apontam risco {rule_risk or 'nao calculado'}."
        )
    elif (ml_prediction or {}).get("error_message"):
        summary = (
            "A camada de Machine Learning ainda nao esta disponivel localmente. "
            f"As regras tecnicas apontam risco {rule_risk or 'nao calculado'}."
        )
    else:
        summary = (
            f"As regras tecnicas apontam risco {rule_risk or 'nao calculado'}."
        )

    if anomaly:
        summary += f" Principal anomalia historica: {anomaly}."
    else:
        summary += " Nenhuma anomalia historica principal foi destacada."

    return {
        "ml_risk": ml_risk or "indisponivel",
        "rule_risk": rule_risk or "indisponivel",
        "main_anomaly": anomaly or "sem anomalia destacada",
        "summary_text": summary,
    }


def build_risk_recommendation(
    risk_level: object,
    event_type: object | None = None,
) -> str:
    """Retorna recomendacao pratica para apoio a decisao."""
    return build_public_recommendation(risk_level, event_type)


def build_divergence_message(
    ml_risk: object,
    rule_risk: object,
) -> str:
    """Explica concordancia ou divergencia entre ML supervisionado e regras."""
    ml_text = str(ml_risk or "").strip().lower()
    rule_text = str(rule_risk or "").strip().lower()
    if not ml_text or ml_text == "indisponivel":
        return (
            "A camada de Machine Learning nao esta disponivel para esta consulta. "
            "O resultado atual usa a classificacao tecnica por regras."
        )
    if not rule_text or rule_text == "indisponivel":
        return (
            f"O modelo supervisionado indicou {ml_text}, mas as regras tecnicas "
            "nao foram calculadas para esta consulta."
        )
    if ml_text == rule_text:
        return (
            "O modelo supervisionado e as regras tecnicas chegaram ao mesmo "
            "nivel de risco."
        )
    return (
        f"O modelo supervisionado indicou {ml_text}, enquanto as regras tecnicas "
        f"indicaram {rule_text}. Essa diferenca pode ocorrer porque o modelo "
        "aprende combinacoes entre variaveis historicas rotuladas, enquanto as "
        "regras aplicam limites diretos."
    )


def _selected_station_label(selected_station: dict[str, object] | None) -> str:
    if not selected_station:
        return ""
    return str(
        selected_station.get("station_label")
        or selected_station.get("station_code")
        or ""
    )


def build_main_result_summary(
    ml_prediction: dict[str, object] | None,
    risk: WeatherRisk | None,
    analysis_result: dict[str, object] | None,
    weather_data: dict[str, object] | None = None,
    selected_station: dict[str, object] | None = None,
) -> dict[str, object]:
    """Consolida os campos principais exibidos no resultado da analise."""
    prediction = ml_prediction or {}
    ml_available = bool(prediction.get("available") and prediction.get("prediction"))
    ml_risk = str(prediction.get("prediction")) if ml_available else "indisponivel"
    rule_risk = risk.risk_level if risk is not None else "indisponivel"
    final_risk = ml_risk if ml_available else rule_risk
    event_type = risk.event_type if risk is not None else "sem_risco_relevante"
    anomaly_count = _historical_anomaly_count(analysis_result)
    main_anomaly = (
        _main_historical_anomaly_text(analysis_result) or "sem anomalia destacada"
    )
    warning = build_warning_summary(
        final_risk,
        event_type,
        ml_risk=ml_risk,
        rule_risk=rule_risk,
        weather_data=weather_data,
        anomaly_count=anomaly_count,
        main_anomaly=main_anomaly,
        station_label=_selected_station_label(selected_station),
    )

    return {
        "final_risk": final_risk or "indisponivel",
        "ml_risk": ml_risk,
        "model_name": str(prediction.get("model_name") or "Nao disponivel"),
        "selection_metric": str(
            prediction.get("selection_metric") or "Nao informada"
        ),
        "rule_risk": rule_risk or "indisponivel",
        "event_type": event_type,
        "anomaly_count": anomaly_count,
        "main_anomaly": main_anomaly,
        "attention_level": warning["action_level"],
        "warning_title": warning["title"],
        "warning_text": warning["warning_text"],
        "recommendation": warning["practical_recommendation"],
        "public_recommendation": warning["public_recommendation"],
        "authority_recommendation": warning["authority_recommendation"],
        "methodological_caution": warning["caution"],
        "event_guidance": warning["event_guidance"],
        "evidence_text": warning["evidence_text"],
        "why_text": warning["why_text"],
    }


def build_result_explanation(
    ml_prediction: dict[str, object] | None,
    risk: WeatherRisk | None,
    analysis_result: dict[str, object] | None,
    weather_data: dict[str, object] | None = None,
    selected_station: dict[str, object] | None = None,
) -> str:
    """Monta explicacao curta sobre a classificacao final."""
    summary = build_main_result_summary(
        ml_prediction,
        risk,
        analysis_result,
        weather_data=weather_data,
        selected_station=selected_station,
    )
    ml_risk = str(summary["ml_risk"])
    rule_risk = str(summary["rule_risk"])
    variable_text = _attention_variable_text(risk, analysis_result)
    rule_text = (
        f"As regras tecnicas indicaram risco {rule_risk}"
        if rule_risk != "indisponivel"
        else "As regras tecnicas nao foram calculadas"
    )
    history_text = _historical_explanation_text(analysis_result)
    divergence_text = build_divergence_message(ml_risk, rule_risk)

    if ml_risk != "indisponivel":
        ml_text = f"O modelo de Machine Learning supervisionado classificou a consulta como {ml_risk}."
    else:
        ml_text = (
            "O modelo de Machine Learning supervisionado nao esta disponivel "
            "localmente; a consulta permanece apoiada pelas regras tecnicas."
        )

    return " ".join(
        [
            ml_text,
            str(summary["why_text"]),
            variable_text,
            rule_text + ".",
            history_text,
            divergence_text,
            "Recomendacao pratica: " + str(summary["recommendation"]),
        ]
    )


def build_evidence_summary(
    weather_data: dict[str, object] | None,
    selected_station: dict[str, object] | None = None,
    requested_period: str = "",
    analysis_result: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    """Monta cards compactos com evidencias usadas na consulta."""
    weather_data = weather_data or {}
    station_label = "-"
    if selected_station:
        station_label = str(
            selected_station.get("station_label")
            or selected_station.get("station_code")
            or "-"
        )

    pressure_value = weather_data.get(
        "pressure_station_hpa",
        weather_data.get("pressure_sea_level_hpa", weather_data.get("pressure")),
    )
    return [
        {
            "label": "Temperatura",
            "value": _format_value(weather_data.get("temperature"), "C"),
        },
        {
            "label": "Sensacao termica",
            "value": _format_value(weather_data.get("feels_like"), "C"),
        },
        {
            "label": "Umidade",
            "value": _format_value(weather_data.get("humidity"), "%"),
        },
        {
            "label": "Chuva",
            "value": _format_value(weather_data.get("precipitation"), "mm"),
        },
        {
            "label": "Vento",
            "value": _format_value(weather_data.get("wind_speed"), "km/h"),
        },
        {
            "label": "Pressao",
            "value": _format_value(pressure_value, "hPa"),
        },
        {
            "label": "Estacao INMET",
            "value": station_label,
        },
        {
            "label": "Periodo historico",
            "value": requested_period or "-",
        },
        {
            "label": "Principal anomalia",
            "value": _main_historical_anomaly_text(analysis_result)
            or "sem anomalia destacada",
        },
    ]


def get_risk_badge_style(risk_level: object) -> dict[str, str]:
    """Retorna cores do badge para um nivel de risco."""
    normalized = str(risk_level or "").strip().lower()
    return RISK_BADGE_STYLES.get(normalized, RISK_BADGE_STYLES["indisponivel"])


def show_intelligent_diagnosis_section(
    ml_prediction: dict[str, object] | None,
    risk: WeatherRisk | None,
    analysis_result: dict[str, object] | None,
) -> None:
    """Mostra o diagnostico principal priorizando a camada ML."""
    summary = build_intelligent_diagnosis_summary(
        ml_prediction,
        risk,
        analysis_result,
    )
    st.subheader("2. Diagnóstico Inteligente")
    with st.container(border=True):
        st.markdown(
            "<div class='sac-section-kicker'>Machine Learning supervisionado</div>",
            unsafe_allow_html=True,
        )
        col_ml, col_rules, col_history = st.columns(3)
        _render_risk_card(
            col_ml,
            "Predição ML",
            summary["ml_risk"],
            "Modelo treinado localmente",
        )
        _render_risk_card(
            col_rules,
            "Risco por regras",
            summary["rule_risk"],
            "Classificação técnica explicável",
        )
        col_history.markdown(
            _html_card(
                "Anomalia histórica",
                summary["main_anomaly"],
                "Comparação com INMET",
                css_class="sac-card-muted",
            ),
            unsafe_allow_html=True,
        )
        st.write(summary["summary_text"])
        st.caption(METHODOLOGICAL_NOTE)


def show_ml_prediction_section(
    weather_data: dict[str, object] | None,
    prediction_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Exibe a predicao ML treinada localmente, quando disponivel."""
    prediction_result = prediction_result or get_ml_prediction_for_dashboard(weather_data)

    with st.container(border=True):
        if not prediction_result.get("available"):
            st.info(str(prediction_result.get("error_message", "")))
            st.caption(str(prediction_result.get("methodological_note", METHODOLOGICAL_NOTE)))
            return prediction_result

        if prediction_result.get("prediction"):
            col_prediction, col_model, col_metric, col_features = st.columns(4)
            _render_risk_card(
                col_prediction,
                "Risk level previsto",
                str(prediction_result["prediction"]),
                "Predição supervisionada",
            )
            col_model.markdown(
                _html_card(
                    "Modelo selecionado",
                    str(prediction_result.get("model_name") or "Nao informado"),
                    "Treinado fora do dashboard",
                    css_class="sac-card-ml",
                ),
                unsafe_allow_html=True,
            )
            col_metric.markdown(
                _html_card(
                    "Metrica de selecao",
                    str(prediction_result.get("selection_metric") or "Nao informada"),
                    "Critério salvo no metadata",
                    css_class="sac-card-ml",
                ),
                unsafe_allow_html=True,
            )
            feature_count = len(prediction_result.get("features_used", []))
            col_features.markdown(
                _html_card(
                    "Features usadas",
                    str(feature_count),
                    "Variáveis do metadata",
                    css_class="sac-card-ml",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info(str(prediction_result.get("error_message", "")))

        observations = prediction_result.get("observations", [])
        if observations:
            for observation in observations:
                st.caption(str(observation))

        with st.expander("Features usadas na previsao ML"):
            rows = [
                {"Feature": feature}
                for feature in prediction_result.get("features_used", [])
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.write("Nenhuma feature informada.")

            missing_features = prediction_result.get("missing_features", [])
            if missing_features:
                st.write("Features ausentes: " + ", ".join(missing_features))

        probability_rows = format_probability_table(
            prediction_result.get("probabilities", {})
        )
        with st.expander("Probabilidades por classe"):
            if probability_rows:
                st.dataframe(
                    pd.DataFrame(probability_rows),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("Este modelo não disponibiliza probabilidades de classe.")

        st.caption(str(prediction_result.get("methodological_note", "")))
    return prediction_result


def show_model_comparison_section(report_dir: str | Path = "data/reports") -> None:
    """Mostra comparacao dos modelos treinados quando ha relatorio local."""
    report_result = load_ml_evaluation_report_for_app(str(report_dir))

    with st.container(border=True):
        st.write(MODEL_SELECTION_EXPLANATION)
        st.info(MODEL_SELECTION_METRIC_NOTE)
        if not report_result["available"]:
            st.info(str(report_result["error_message"]))
            st.code(ML_EVALUATION_COMMAND)
            return

        st.caption(f"Relatorio usado: {report_result['source']}")
        rows = build_model_comparison_table(report_result["report"])
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("O relatorio ML nao possui metricas por modelo para exibir.")

        summary = extract_best_model_summary(report_result["report"])
        col_baseline, col_best, col_diff = st.columns(3)
        col_baseline.metric(
            "F1 macro baseline",
            _format_metric(summary["baseline_f1_macro"]),
        )
        col_best.metric(
            "F1 macro melhor modelo",
            _format_metric(summary["best_model_f1_macro"]),
        )
        col_diff.metric(
            "Diferença absoluta",
            _format_metric(summary["absolute_difference"]),
        )
        if summary["best_model_name"]:
            st.caption(
                "Melhor modelo: "
                f"{summary['best_model_name']} "
                f"pela métrica {summary['selection_metric']}."
            )


def show_analysis_result_flow(
    weather_data: dict[str, object],
    risk: WeatherRisk | None,
    prediction_result: dict[str, object] | None,
    analysis_result: dict[str, object] | None = None,
    history: pd.DataFrame | None = None,
    selected_station: dict[str, object] | None = None,
    station_label: str = "",
    source_status: dict[str, str] | None = None,
    requested_period: str = "",
    station_message: str = "",
) -> None:
    """Exibe o fluxo simplificado: resultado, evidencias e detalhes."""
    show_result_analysis_section(
        prediction_result,
        risk,
        analysis_result,
        weather_data=weather_data,
        selected_station=selected_station,
    )
    show_evidence_section(
        weather_data,
        selected_station=selected_station,
        requested_period=requested_period,
        analysis_result=analysis_result,
    )
    show_technical_details_section(
        weather_data=weather_data,
        risk=risk,
        prediction_result=prediction_result,
        analysis_result=analysis_result,
        history=history,
        selected_station=selected_station,
        station_label=station_label,
        source_status=source_status,
        requested_period=requested_period,
        station_message=station_message,
    )


def show_result_analysis_section(
    ml_prediction: dict[str, object] | None,
    risk: WeatherRisk | None,
    analysis_result: dict[str, object] | None,
    weather_data: dict[str, object] | None = None,
    selected_station: dict[str, object] | None = None,
) -> None:
    """Mostra a conclusao principal da consulta."""
    summary = build_main_result_summary(
        ml_prediction,
        risk,
        analysis_result,
        weather_data=weather_data,
        selected_station=selected_station,
    )
    explanation = build_result_explanation(
        ml_prediction,
        risk,
        analysis_result,
        weather_data=weather_data,
        selected_station=selected_station,
    )

    st.subheader("Resultado da analise")
    with st.container(border=True):
        col_summary, col_explanation = st.columns([1, 1.35])
        with col_summary:
            _render_risk_card(
                st,
                "Risco previsto pela IA",
                str(summary["final_risk"]),
                "Machine Learning supervisionado quando disponivel",
            )
            metric_col_a, metric_col_b = st.columns(2)
            metric_col_a.metric("Modelo usado", str(summary["model_name"]))
            metric_col_b.metric("Metrica", str(summary["selection_metric"]))
            metric_col_c, metric_col_d = st.columns(2)
            metric_col_c.metric("Risco por regras", str(summary["rule_risk"]).upper())
            metric_col_d.metric("Anomalias historicas", summary["anomaly_count"])
            st.metric("Nivel de atencao", str(summary["attention_level"]))
            st.info(str(summary["warning_text"]))
            st.write(str(summary["recommendation"]))

        with col_explanation:
            st.markdown(f"#### {summary['warning_title']}")
            st.write(explanation)
            st.info(str(summary["event_guidance"]))
            st.caption(METHODOLOGICAL_NOTE_SHORT)
            with st.expander("Detalhe metodologico"):
                st.write(METHODOLOGICAL_NOTE)


def show_evidence_section(
    weather_data: dict[str, object],
    selected_station: dict[str, object] | None = None,
    requested_period: str = "",
    analysis_result: dict[str, object] | None = None,
) -> None:
    """Mostra evidencias compactas usadas na analise."""
    st.subheader("Evidencias usadas")
    rows = build_evidence_summary(
        weather_data,
        selected_station=selected_station,
        requested_period=requested_period,
        analysis_result=analysis_result,
    )
    columns = st.columns(3)
    for index, row in enumerate(rows):
        columns[index % 3].markdown(
            _html_card(row["label"], row["value"], "", css_class="sac-card-muted"),
            unsafe_allow_html=True,
        )


def show_warning_recommendations_section(
    weather_data: dict[str, object],
    risk: WeatherRisk | None,
    prediction_result: dict[str, object] | None,
    analysis_result: dict[str, object] | None,
    selected_station: dict[str, object] | None = None,
) -> None:
    """Mostra recomendacoes preventivas completas do aviso."""
    summary = build_main_result_summary(
        prediction_result,
        risk,
        analysis_result,
        weather_data=weather_data,
        selected_station=selected_station,
    )

    st.markdown(f"#### {summary['warning_title']}")
    col_attention, col_event = st.columns(2)
    col_attention.metric("Nivel de atencao", str(summary["attention_level"]))
    col_event.metric("Evento principal", format_event_type(summary["event_type"]))
    st.write(str(summary["warning_text"]))

    with st.expander("Evidencias consideradas", expanded=True):
        st.write(str(summary["evidence_text"]))

    with st.expander("Recomendacao para usuario/populacao", expanded=True):
        st.write(str(summary["public_recommendation"]))

    with st.expander("Recomendacao para autoridades/responsaveis", expanded=True):
        st.write(str(summary["authority_recommendation"]))

    with st.expander("Cautela metodologica"):
        st.write(str(summary["methodological_caution"]))
        st.write(METHODOLOGICAL_NOTE_SHORT)


def show_technical_details_section(
    weather_data: dict[str, object],
    risk: WeatherRisk | None,
    prediction_result: dict[str, object] | None,
    analysis_result: dict[str, object] | None = None,
    history: pd.DataFrame | None = None,
    selected_station: dict[str, object] | None = None,
    station_label: str = "",
    source_status: dict[str, str] | None = None,
    requested_period: str = "",
    station_message: str = "",
) -> None:
    """Agrupa os detalhes tecnicos em abas."""
    st.subheader("Detalhes tecnicos")
    tabs = st.tabs(
        [
            "Avisos e recomendacoes",
            MODEL_SELECTION_TAB_LABEL,
            "Probabilidades por classe",
            "Features usadas pelo modelo",
            "Regras acionadas",
            "Estatisticas historicas",
            "Variaveis brutas",
        ]
    )

    with tabs[0]:
        show_warning_recommendations_section(
            weather_data=weather_data,
            risk=risk,
            prediction_result=prediction_result,
            analysis_result=analysis_result,
            selected_station=selected_station,
        )

    with tabs[1]:
        show_model_comparison_section()

    with tabs[2]:
        probability_rows = format_probability_table(
            (prediction_result or {}).get("probabilities", {})
        )
        if probability_rows:
            st.dataframe(pd.DataFrame(probability_rows), width="stretch", hide_index=True)
        else:
            st.caption("Este modelo nao disponibiliza probabilidades de classe.")

    with tabs[3]:
        if not (prediction_result or {}).get("available"):
            st.info(str((prediction_result or {}).get("error_message", "")))
        features = [
            {"Feature": feature}
            for feature in (prediction_result or {}).get("features_used", [])
        ]
        if features:
            st.dataframe(pd.DataFrame(features), width="stretch", hide_index=True)
        missing_features = (prediction_result or {}).get("missing_features", [])
        if missing_features:
            st.write("Features ausentes: " + ", ".join(missing_features))
        observations = (prediction_result or {}).get("observations", [])
        for observation in observations:
            st.caption(str(observation))

    with tabs[4]:
        if risk is None:
            st.info("Classificacao por regras nao calculada para esta consulta.")
        else:
            show_weather_risk(risk)

    with tabs[5]:
        _show_historical_details(
            weather_data=weather_data,
            analysis_result=analysis_result,
            history=history,
            selected_station=selected_station,
            station_label=station_label,
            source_status=source_status,
            requested_period=requested_period,
            station_message=station_message,
        )

    with tabs[6]:
        st.write("Dados atuais padronizados")
        st.dataframe(
            pd.DataFrame(build_current_weather_rows(weather_data)),
            width="stretch",
            hide_index=True,
        )
        st.caption(build_current_pressure_text(weather_data))
        with st.expander("Payload atual completo"):
            st.json(weather_data)
        if selected_station is not None:
            with st.expander("Estacao INMET selecionada"):
                st.json(selected_station)


def _render_risk_card(
    target: Any,
    label: str,
    risk_level: str,
    caption: str,
) -> None:
    target.markdown(
        _html_risk_card(label, risk_level, caption),
        unsafe_allow_html=True,
    )


def _html_risk_card(label: str, risk_level: str, caption: str) -> str:
    style = get_risk_badge_style(risk_level)
    risk_text = str(risk_level or "indisponivel")
    return (
        "<div class='sac-card sac-card-ml'>"
        f"<div class='sac-card-label'>{_escape_html(label)}</div>"
        f"<div class='sac-risk-badge' style='background:{style['background']};"
        f"border-color:{style['border']};color:{style['text']}'>"
        f"{_escape_html(risk_text.upper())}</div>"
        f"<div class='sac-card-caption'>{_escape_html(caption)}</div>"
        "</div>"
    )


def _html_card(
    label: str,
    value: str,
    caption: str,
    css_class: str = "",
) -> str:
    class_names = f"sac-card {css_class}".strip()
    return (
        f"<div class='{class_names}'>"
        f"<div class='sac-card-label'>{_escape_html(label)}</div>"
        f"<div class='sac-card-value'>{_escape_html(value)}</div>"
        f"<div class='sac-card-caption'>{_escape_html(caption)}</div>"
        "</div>"
    )


def _escape_html(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _metrics_by_model_from_report(
    report: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    metrics = report.get("metrics_by_model")
    if isinstance(metrics, dict):
        return metrics
    metadata_metrics = _report_metadata(report).get("metrics_by_model")
    if isinstance(metadata_metrics, dict):
        return metadata_metrics
    return {}


def _report_metadata(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    metadata = report.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _report_source_from_filename(filename: str) -> str:
    if "temporal" in filename:
        return "avaliação robusta temporal"
    if "random" in filename:
        return "avaliação robusta aleatória"
    return "treinamento básico"


def _round_metric(value: object) -> float | None:
    numeric_value = _optional_float(value)
    if numeric_value is None:
        return None
    return round(numeric_value, 4)


def _format_metric(value: object) -> str:
    numeric_value = _optional_float(value)
    if numeric_value is None:
        return "-"
    return f"{numeric_value:.4f}"


def _main_historical_anomaly_text(
    analysis_result: dict[str, object] | None,
) -> str:
    if not analysis_result or not analysis_result.get("has_historical_data"):
        return ""

    anomalies = analysis_result.get("anomalies", [])
    if not isinstance(anomalies, list) or not anomalies:
        return ""

    for anomaly in anomalies:
        if not isinstance(anomaly, dict):
            continue
        anomaly_type = str(anomaly.get("anomaly_type", ""))
        label = ANOMALY_LABELS.get(anomaly_type, anomaly_type)
        reason = str(anomaly.get("reason", "")).strip()
        if reason:
            return f"{label}: {reason}"
        return label
    return ""


def _historical_anomaly_count(analysis_result: dict[str, object] | None) -> int:
    if not analysis_result or not analysis_result.get("has_historical_data"):
        return 0
    anomalies = analysis_result.get("anomalies", [])
    return len(anomalies) if isinstance(anomalies, list) else 0


def _attention_variable_text(
    risk: WeatherRisk | None,
    analysis_result: dict[str, object] | None,
) -> str:
    if analysis_result and analysis_result.get("has_historical_data"):
        variables = sorted(_anomalous_variables(analysis_result))
        labels = [
            VARIABLE_DISPLAY.get(variable, {}).get("label", variable)
            for variable in variables
            if variable
        ]
        if labels:
            return "As variaveis que mais chamaram atencao foram: " + ", ".join(
                labels[:3]
            ) + "."

    if risk and risk.triggered_rules:
        return (
            "As regras acionadas apontam pontos de atencao em "
            + ", ".join(str(rule) for rule in risk.triggered_rules[:2])
            + "."
        )

    if risk and risk.variables:
        labels = [
            VARIABLE_DISPLAY.get(variable, {}).get("label", str(variable))
            for variable, value in risk.variables.items()
            if value is not None
        ]
        if labels:
            return "As variaveis observadas incluem " + ", ".join(labels[:3]) + "."

    return "Nenhuma variavel isolada foi destacada como anomalia principal."


def _historical_explanation_text(
    analysis_result: dict[str, object] | None,
) -> str:
    if not analysis_result or not analysis_result.get("has_historical_data"):
        return "A comparacao historica ainda nao esta disponivel para esta consulta."

    anomaly_count = _historical_anomaly_count(analysis_result)
    if anomaly_count == 0:
        return (
            "A comparacao historica com o INMET nao encontrou anomalias "
            "principais nos valores atuais."
        )

    anomaly_text = _main_historical_anomaly_text(analysis_result)
    return (
        f"A comparacao historica com o INMET encontrou {anomaly_count} "
        f"anomalia(s); principal evidencia: {anomaly_text}."
    )


def _show_risk_reason(risk: WeatherRisk) -> None:
    if risk.risk_level in {"critico", "alto"}:
        st.error(risk.reason)
    elif risk.risk_level == "moderado":
        st.warning(risk.reason)
    else:
        st.info(risk.reason)


def historical_analysis_status(
    weather_data: dict[str, object] | None,
    history: pd.DataFrame | None,
) -> tuple[bool, str]:
    """Indica se a comparacao historica pode ser executada."""
    if not weather_data:
        return (
            False,
            "Carregue dados meteorologicos atuais para comparar com o INMET.",
        )
    if history is None or history.empty:
        return (
            False,
            "Carregue o historico INMET da estacao selecionada.",
        )
    return True, ""


def format_historical_anomalies(
    anomalies: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Formata anomalias historicas para exibicao em tabela."""
    rows = []
    for anomaly in anomalies:
        variable = _first_variable(anomaly)
        unit = _unit_for_variable(variable)
        rows.append(
            {
                "Tipo": ANOMALY_LABELS.get(
                    str(anomaly.get("anomaly_type")),
                    str(anomaly.get("anomaly_type", "")),
                ),
                "Severidade": str(anomaly.get("severity", "")),
                "Valor atual": _format_value(anomaly.get("current_value"), unit),
                "Referencia historica": _format_historical_reference(anomaly, unit),
                "Justificativa": str(anomaly.get("reason", "")),
            }
        )

    return rows


def build_historical_statistics_rows(
    weather_data: dict[str, object] | None,
    analysis_result: dict[str, object],
) -> list[dict[str, object]]:
    """Monta resumo das estatisticas historicas usadas na comparacao."""
    if not weather_data or not analysis_result.get("has_historical_data"):
        return []

    statistics = analysis_result.get("statistics", {})
    if not isinstance(statistics, dict):
        return []

    pressure_comparison = analysis_result.get("pressure_comparison", {})
    if not isinstance(pressure_comparison, dict):
        pressure_comparison = {}

    anomalous_variables = _anomalous_variables(analysis_result)
    rows = []
    for variable in VARIABLE_DISPLAY:
        stats = statistics.get(variable)
        if not isinstance(stats, dict):
            continue

        current_value = _current_value_for_historical_variable(
            variable,
            weather_data,
            pressure_comparison,
        )
        rows.append(
            {
                "Variavel": VARIABLE_DISPLAY[variable]["label"],
                "Valor atual": _format_value(
                    current_value,
                    VARIABLE_DISPLAY[variable]["unit"],
                ),
                "Media historica": _format_value(
                    stats.get("mean"),
                    VARIABLE_DISPLAY[variable]["unit"],
                ),
                "Percentil usado": _format_percentile_value(variable, stats),
                "Interpretacao": _interpret_historical_variable(
                    variable,
                    current_value,
                    pressure_comparison,
                    anomalous_variables,
                ),
            }
        )

    return rows


def build_pressure_reference_details(
    weather_data: dict[str, object] | None,
    pressure_comparison: dict[str, object],
) -> str:
    """Descreve o referencial de pressao usado na comparacao historica."""
    if not pressure_comparison.get("evaluated"):
        return str(pressure_comparison.get("message", "Pressao nao avaliada."))

    station_pressure = _format_value(
        pressure_comparison.get("pressure_station_hpa"),
        "hPa",
    )
    sea_level_pressure = _format_value(
        (weather_data or {}).get("pressure_sea_level_hpa"),
        "hPa",
    )
    sea_level_text = (
        f"Pressao ao nivel do mar informada pela OpenWeather: {sea_level_pressure}. "
        if sea_level_pressure != "-"
        else ""
    )
    suffix = " Comparacao feita no nivel da estacao INMET."
    if pressure_comparison.get("is_estimated"):
        altitude = pressure_comparison.get("altitude_m")
        altitude_text = (
            f" com altitude de {float(altitude):.1f} m"
            if _is_number(altitude)
            else ""
        )
        return (
            sea_level_text
            + (
            "Pressao estimada ao nivel da estacao "
            f"({station_pressure}){altitude_text}." + suffix
            )
        )

    pressure_reference = (weather_data or {}).get("pressure_reference")
    if pressure_reference == "openweather_grnd_level":
        return (
            sea_level_text
            + (
            "Pressao ao nivel da estacao informada por grnd_level da "
            f"OpenWeather ({station_pressure})." + suffix
            )
        )
    if pressure_reference == "manual_station_level":
        return (
            "Pressao manual tratada como pressao ao nivel da estacao "
            f"({station_pressure})." + suffix
        )

    return (
        "Pressao atual ao nivel da estacao usada diretamente "
        f"({station_pressure})." + suffix
    )


def find_nearest_station_by_coordinates(
    latitude: float | None,
    longitude: float | None,
    station_catalog: pd.DataFrame,
) -> dict[str, object] | None:
    """Seleciona a estacao INMET mais proxima das coordenadas informadas."""
    if latitude is None or longitude is None or station_catalog.empty:
        return None

    stations = station_catalog.copy()
    stations["latitude"] = pd.to_numeric(stations.get("latitude"), errors="coerce")
    stations["longitude"] = pd.to_numeric(stations.get("longitude"), errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude"])
    if stations.empty:
        return None

    stations["distance_km"] = stations.apply(
        lambda row: calculate_distance_km(
            float(latitude),
            float(longitude),
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )
    nearest = stations.sort_values("distance_km").iloc[0].to_dict()
    nearest["station_resolution_method"] = "coordinates"
    return nearest


def find_station_by_city_name(
    city: str | None,
    station_catalog: pd.DataFrame,
) -> dict[str, object] | None:
    """Busca estacao INMET pelo nome da cidade quando nao ha coordenadas."""
    if not city or station_catalog.empty:
        return None

    city_normalized = normalize_city_name(city)
    catalog = station_catalog.copy()
    matches = catalog[
        (catalog["city_normalized"] == city_normalized)
        | (catalog["station_name_normalized"] == city_normalized)
    ].reset_index(drop=True)
    if matches.empty:
        return None

    station = matches.sort_values(["state", "station_code"]).iloc[0].to_dict()
    station["station_resolution_method"] = "city"
    return station


def resolve_station_for_weather_data(
    weather_data: dict[str, object] | None,
    station_catalog: pd.DataFrame,
) -> tuple[dict[str, object] | None, str]:
    """Resolve a estacao INMET para dados atuais, priorizando coordenadas."""
    if not weather_data:
        return None, "Dados atuais ausentes."

    latitude = _optional_float(weather_data.get("latitude"))
    longitude = _optional_float(weather_data.get("longitude"))
    station = find_nearest_station_by_coordinates(latitude, longitude, station_catalog)
    if station is not None:
        return station, "Estacao INMET associada automaticamente por coordenadas."

    station = find_station_by_city_name(
        str(weather_data.get("city", "")),
        station_catalog,
    )
    if station is not None:
        return station, "Estacao INMET associada automaticamente pelo nome da cidade."

    return None, "Nao foi possivel associar automaticamente uma estacao INMET."


def calculate_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calcula distancia aproximada em km pela formula de Haversine."""
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))


def normalize_historical_datetime(history: pd.DataFrame) -> pd.DataFrame:
    """Normaliza datetime historico e remove registros sem data valida."""
    if history is None or history.empty:
        return pd.DataFrame()

    normalized_history = history.copy()
    if "datetime" in normalized_history.columns:
        datetime_values = normalized_history["datetime"]
    elif "date" in normalized_history.columns:
        datetime_values = normalized_history["date"]
    else:
        return normalized_history.iloc[0:0].copy()

    numeric_mask = datetime_values.map(
        lambda value: isinstance(value, int | float) and not isinstance(value, bool)
    )
    datetime_values = datetime_values.mask(numeric_mask)
    normalized_history["datetime"] = pd.to_datetime(
        datetime_values,
        errors="coerce",
    )
    return normalized_history.dropna(subset=["datetime"]).reset_index(drop=True)


def historical_period_text(history: pd.DataFrame) -> str:
    """Formata periodo historico sem quebrar com datas de tipos mistos."""
    normalized_history = normalize_historical_datetime(history)
    if normalized_history.empty:
        return "Nao identificado"

    start_date = normalized_history["datetime"].min()
    end_date = normalized_history["datetime"].max()
    if pd.isna(start_date) or pd.isna(end_date):
        return "Nao identificado"

    return f"{start_date:%Y-%m-%d} a {end_date:%Y-%m-%d}"


def build_database_station_catalog(db_path: str) -> pd.DataFrame:
    """Monta catalogo de estacoes a partir do DuckDB processado."""
    stations = get_available_stations_from_database(db_path)
    if stations.empty:
        return stations

    catalog = stations.copy()
    catalog["station_name"] = catalog["station_name"].fillna("")
    catalog["city"] = catalog["station_name"]
    catalog["station_name_normalized"] = catalog["station_name"].map(
        normalize_city_name
    )
    catalog["city_normalized"] = catalog["city"].map(normalize_city_name)
    catalog["station_label"] = catalog.apply(
        lambda row: (
            f"{row['station_name']} - {row['state']} | {row['station_code']}"
        ),
        axis=1,
    )
    return catalog


def format_safe_table_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Converte linhas para strings seguras para o Streamlit/Arrow."""
    safe_rows = []
    for row in rows:
        safe_rows.append(
            {str(key): _format_table_cell(value) for key, value in row.items()}
        )
    return safe_rows


def build_current_weather_rows(weather_data: dict[str, object]) -> list[dict[str, str]]:
    """Monta uma tabela pequena dos dados atuais."""
    rows = []
    for field, (label, unit) in CURRENT_WEATHER_DISPLAY.items():
        if field in weather_data:
            rows.append(
                {
                    "Variavel": label,
                    "Valor": _format_value(weather_data.get(field), unit),
                }
            )
    return format_safe_table_rows(rows)


def build_current_pressure_text(weather_data: dict[str, object] | None) -> str:
    """Explica a pressao atual sem expor apenas nomes tecnicos."""
    if not weather_data:
        return "Pressao atual indisponivel."

    pressure_reference = weather_data.get("pressure_reference")
    if pressure_reference == "openweather_grnd_level":
        return (
            "Pressao usada na comparacao: valor ao nivel da estacao informado "
            "pela OpenWeather. Esse referencial e compativel com o historico INMET."
        )
    if pressure_reference == "manual_station_level":
        return (
            "Pressao usada na comparacao: valor manual tratado como pressao ao "
            "nivel da estacao."
        )
    if weather_data.get("pressure_station_hpa") is not None:
        return (
            "Pressao usada na comparacao: valor atual ao nivel da estacao, "
            "quando disponivel."
        )
    if weather_data.get("pressure_sea_level_hpa") is not None:
        return (
            "A OpenWeather informou pressao ao nivel do mar. Para comparar com "
            "o INMET, o sistema estima a pressao ao nivel da estacao usando a "
            "altitude da estacao associada."
        )
    return "Pressao atual indisponivel para comparacao historica."


def format_event_type(event_type: str | None) -> str:
    """Converte o codigo do evento em texto legivel."""
    return EVENT_LABELS.get(str(event_type), str(event_type or "-"))


def build_interpretive_summary(
    weather_data: dict[str, object] | None,
    risk: WeatherRisk | None,
    analysis_result: dict[str, object] | None,
    station_label: str | None,
) -> str:
    """Monta resumo final em linguagem simples para apresentacao."""
    if not weather_data:
        return (
            "Nao ha dados meteorologicos atuais carregados. Informe uma cidade "
            "na OpenWeather ou use a entrada manual para iniciar a analise."
        )

    if risk is None:
        return (
            "Os dados atuais foram carregados, mas o alerta por regras ainda "
            "nao foi calculado."
        )

    event_label = format_event_type(risk.event_type)
    summary = (
        f"A condicao atual apresenta risco {risk.risk_level} por {event_label}."
    )

    if not analysis_result or not analysis_result.get("has_historical_data"):
        return (
            summary
            + " A comparacao com o historico INMET ainda nao esta disponivel "
            "para esta consulta."
        )

    station_text = (
        f" da estacao INMET associada ({station_label})"
        if station_label
        else " da estacao INMET associada"
    )
    anomalies = analysis_result.get("anomalies", [])
    if not isinstance(anomalies, list) or not anomalies:
        return (
            summary
            + f" Em comparacao com o historico{station_text}, os valores "
            "atuais estao dentro do padrao estatistico observado para o periodo."
        )

    anomaly_text = _summarize_first_historical_anomaly(anomalies)
    return (
        summary
        + f" Em comparacao com o historico{station_text}, {anomaly_text}."
    )


def _summarize_first_historical_anomaly(
    anomalies: list[dict[str, object]],
) -> str:
    for anomaly in anomalies:
        if not isinstance(anomaly, dict):
            continue
        anomaly_type = str(anomaly.get("anomaly_type", ""))
        return ANOMALY_SUMMARY_TEXT.get(
            anomaly_type,
            "ha pelo menos uma variavel fora do padrao historico observado",
        )
    return "ha pelo menos uma variavel fora do padrao historico observado"


def load_station_catalog_for_app(
    settings: Settings,
    client: InmetClient,
) -> tuple[pd.DataFrame, str]:
    """Carrega catalogo pelo DuckDB quando existir, senao pelos ZIPs locais."""
    if database_exists(settings.inmet_database_path):
        return (
            build_database_station_catalog(settings.inmet_database_path),
            "database",
        )

    return client.build_station_catalog(), "zip"


def load_station_history_for_app(
    settings: Settings,
    client: InmetClient,
    station_code: str,
    start_year: int,
    end_year: int,
    source: str,
) -> pd.DataFrame:
    """Carrega historico filtrado pela fonte disponivel."""
    if source == "database":
        history = load_station_history_from_database(
            station_code,
            start_year,
            end_year,
            settings.inmet_database_path,
        )
    else:
        history = client.load_station_history(station_code, start_year, end_year)

    return normalize_historical_datetime(history)


def build_historical_source_status(
    historical_source: str,
    metadata: dict[str, object] | None = None,
) -> dict[str, str]:
    """Formata o status da fonte historica usada no dashboard."""
    if historical_source == "database":
        metadata = metadata or {}
        return {
            "source_label": "DuckDB",
            "status": "Base historica processada encontrada",
            "details": (
                "Consulta historica usando DuckDB filtrado por estacao e periodo. "
                f"Estacoes: {_format_table_cell(metadata.get('station_count'))}; "
                f"registros: {_format_table_cell(metadata.get('record_count'))}."
            ),
            "guidance": "",
        }

    return {
        "source_label": "ZIPs locais",
        "status": "Base historica processada nao encontrada",
        "details": (
            "O dashboard tentara usar os ZIPs locais do INMET como fallback "
            "quando eles estiverem disponiveis. Em ambiente online sem ZIPs, "
            "a consulta historica ficara indisponivel sem derrubar o app."
        ),
        "guidance": (
            "Para baixar a base processada, execute: "
            "python scripts/download_inmet_database.py"
        ),
    }


def show_simplified_analysis_section(
    settings: Settings,
    weather_data: dict[str, object] | None,
    data_source: str,
    risk: WeatherRisk | None,
    start_year: int,
    end_year: int,
) -> None:
    """Executa a consulta historica e mostra o fluxo compacto do dashboard."""
    if not weather_data:
        st.info(
            "Carregue dados atuais por Entrada manual ou OpenWeather para "
            "executar a analise."
        )
        return

    prediction_result = get_ml_prediction_for_dashboard(weather_data)
    requested_period = format_requested_period(start_year, end_year)
    client = InmetClient(settings)

    try:
        station_catalog, historical_source = load_station_catalog_for_app(
            settings,
            client,
        )
    except InmetHistoricalDataError as error:
        st.warning(safe_dashboard_message(error, settings))
        st.info(
            "A consulta continua com dados atuais, regras tecnicas e camada ML "
            "quando houver modelo. Para habilitar historico INMET online, "
            "configure o DuckDB por release ou inclua ZIPs locais no ambiente."
        )
        show_analysis_result_flow(
            weather_data=weather_data,
            risk=risk,
            prediction_result=prediction_result,
            requested_period=requested_period,
        )
        return

    metadata = (
        get_database_metadata(settings.inmet_database_path)
        if historical_source == "database"
        else None
    )
    source_status = build_historical_source_status(historical_source, metadata)
    station_selection_key = (
        f"{normalize_city_name(str(weather_data.get('city', '')))}_"
        f"{data_source}_{start_year}_{end_year}"
    )
    selected_station, station_message = _select_station_for_historical_flow(
        data_source,
        weather_data,
        station_catalog,
        station_selection_key,
    )
    if selected_station is None:
        st.warning(station_message)
        show_analysis_result_flow(
            weather_data=weather_data,
            risk=risk,
            prediction_result=prediction_result,
            source_status=source_status,
            requested_period=requested_period,
            station_message=station_message,
        )
        return

    station_code = str(selected_station["station_code"])
    station_label = str(selected_station.get("station_label", station_code))

    try:
        history = load_station_history_for_app(
            settings,
            client,
            station_code,
            int(start_year),
            int(end_year),
            historical_source,
        )
    except InmetHistoricalDataError as error:
        st.warning(safe_dashboard_message(error, settings))
        show_analysis_result_flow(
            weather_data=weather_data,
            risk=risk,
            prediction_result=prediction_result,
            selected_station=selected_station,
            station_label=station_label,
            source_status=source_status,
            requested_period=requested_period,
            station_message=station_message,
        )
        return

    if history.empty:
        st.warning("Nenhum registro historico foi carregado para esta estacao.")
        show_analysis_result_flow(
            weather_data=weather_data,
            risk=risk,
            prediction_result=prediction_result,
            history=history,
            selected_station=selected_station,
            station_label=station_label,
            source_status=source_status,
            requested_period=requested_period,
            station_message=station_message,
        )
        return

    can_analyze, status_message = historical_analysis_status(weather_data, history)
    if not can_analyze:
        st.info(status_message)
        show_analysis_result_flow(
            weather_data=weather_data,
            risk=risk,
            prediction_result=prediction_result,
            history=history,
            selected_station=selected_station,
            station_label=station_label,
            source_status=source_status,
            requested_period=requested_period,
            station_message=station_message,
        )
        return

    analyzer = HistoricalAnalyzer()
    analysis_result = analyzer.analyze(
        weather_data,
        history,
        station_metadata=selected_station,
    )
    show_analysis_result_flow(
        weather_data=weather_data,
        risk=risk,
        prediction_result=prediction_result,
        analysis_result=analysis_result,
        history=history,
        selected_station=selected_station,
        station_label=station_label,
        source_status=source_status,
        requested_period=requested_period,
        station_message=station_message,
    )


def show_inmet_historical_section(
    settings: Settings,
    weather_data: dict[str, object] | None,
    data_source: str,
    risk: WeatherRisk | None,
    start_year: int,
    end_year: int,
) -> None:
    """Exibe comparacao historica local do INMET sem gerar alerta principal."""
    if not weather_data:
        st.info(
            "Carregue dados atuais por Entrada manual ou OpenWeather para "
            "executar a comparacao historica."
        )
        return

    prediction_result = get_ml_prediction_for_dashboard(weather_data)
    client = InmetClient(settings)
    try:
        station_catalog, historical_source = load_station_catalog_for_app(
            settings,
            client,
        )
    except InmetHistoricalDataError as error:
        st.warning(safe_dashboard_message(error, settings))
        if risk is not None:
            show_intelligent_diagnosis_section(prediction_result, risk, None)
            st.subheader("3. Predição por Machine Learning")
            show_ml_prediction_section(weather_data, prediction_result)
            st.subheader(f"4. {MODEL_SELECTION_TAB_LABEL}")
            show_model_comparison_section()
            st.subheader("5. Explicabilidade Técnica por Regras")
            show_weather_risk(risk)
            st.subheader("6. Resumo")
            st.write(build_interpretive_summary(weather_data, risk, None, None))
        return

    if historical_source == "database":
        metadata = get_database_metadata(settings.inmet_database_path)
    else:
        metadata = None

    source_status = build_historical_source_status(historical_source, metadata)
    station_selection_key = (
        f"{normalize_city_name(str(weather_data.get('city', '')))}_"
        f"{data_source}_{start_year}_{end_year}"
    )

    selected_station, station_message = _select_station_for_historical_flow(
        data_source,
        weather_data,
        station_catalog,
        station_selection_key,
    )
    if selected_station is None:
        st.warning(station_message)
        if risk is not None:
            show_intelligent_diagnosis_section(prediction_result, risk, None)
            st.subheader("3. Predição por Machine Learning")
            show_ml_prediction_section(weather_data, prediction_result)
            st.subheader(f"4. {MODEL_SELECTION_TAB_LABEL}")
            show_model_comparison_section()
            st.subheader("5. Explicabilidade Técnica por Regras")
            show_weather_risk(risk)
            st.subheader("6. Resumo")
            st.write(build_interpretive_summary(weather_data, risk, None, None))
        return

    station_code = str(selected_station["station_code"])
    station_label = str(selected_station.get("station_label", station_code))

    try:
        history = load_station_history_for_app(
            settings,
            client,
            station_code,
            int(start_year),
            int(end_year),
            historical_source,
        )
    except InmetHistoricalDataError as error:
        st.warning(safe_dashboard_message(error, settings))
        if risk is not None:
            show_intelligent_diagnosis_section(prediction_result, risk, None)
            st.subheader("3. Predição por Machine Learning")
            show_ml_prediction_section(weather_data, prediction_result)
            st.subheader(f"4. {MODEL_SELECTION_TAB_LABEL}")
            show_model_comparison_section()
            st.subheader("5. Explicabilidade Técnica por Regras")
            show_weather_risk(risk)
            st.subheader("6. Resumo")
            st.write(
                build_interpretive_summary(weather_data, risk, None, station_label)
            )
        return

    if history.empty:
        st.warning("Nenhum registro historico foi carregado para esta estacao.")
        if risk is not None:
            show_intelligent_diagnosis_section(prediction_result, risk, None)
            st.subheader("3. Predição por Machine Learning")
            show_ml_prediction_section(weather_data, prediction_result)
            st.subheader(f"4. {MODEL_SELECTION_TAB_LABEL}")
            show_model_comparison_section()
            st.subheader("5. Explicabilidade Técnica por Regras")
            show_weather_risk(risk)
            st.subheader("6. Resumo")
            st.write(
                build_interpretive_summary(weather_data, risk, None, station_label)
            )
        return

    averages = history[
        ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]
    ].mean()

    can_analyze, status_message = historical_analysis_status(weather_data, history)
    if not can_analyze:
        st.info(status_message)
        return

    analyzer = HistoricalAnalyzer()
    analysis_result = analyzer.analyze(
        weather_data,
        history,
        station_metadata=selected_station,
    )

    show_intelligent_diagnosis_section(prediction_result, risk, analysis_result)

    st.subheader("3. Predição por Machine Learning")
    show_ml_prediction_section(weather_data, prediction_result)

    st.subheader(f"4. {MODEL_SELECTION_TAB_LABEL}")
    show_model_comparison_section()

    st.subheader("5. Base historica associada")
    with st.container(border=True):
        _show_station_selection_summary(
            selected_station,
            station_label,
            source_status["source_label"],
            format_requested_period(start_year, end_year),
            len(history),
        )
        st.caption(station_message)
        with st.expander("Detalhes da base historica"):
            st.write(source_status["status"])
            st.write(source_status["details"])
            if source_status["guidance"]:
                st.write(source_status["guidance"])
                st.code("python scripts/download_inmet_database.py")
            _show_loaded_history_summary(
                history,
                selected_station,
                station_label,
                source_status["source_label"],
                format_requested_period(start_year, end_year),
            )

    st.subheader("6. Comparacao historica")
    with st.container(border=True):
        _show_historical_analysis_result(weather_data, analysis_result)
        with st.expander("Registros carregados e medias historicas"):
            metric_columns = st.columns(5)
            metric_columns[0].metric(
                "Temperatura media",
                f"{averages['temperature']:.1f} C",
            )
            metric_columns[1].metric("Umidade media", f"{averages['humidity']:.1f}%")
            metric_columns[2].metric(
                "Pressao media",
                f"{averages['pressure']:.1f} hPa",
            )
            metric_columns[3].metric(
                "Vento medio",
                f"{averages['wind_speed']:.1f} km/h",
            )
            metric_columns[4].metric(
                "Precipitacao media",
                f"{averages['precipitation']:.1f} mm",
            )
            st.caption(
                "A pressao historica do INMET esta ao nivel da estacao. "
                "A comparacao usa grnd_level da OpenWeather quando disponivel "
                "ou estima a pressao pela altitude da estacao."
            )

    if risk is not None:
        st.subheader("7. Explicabilidade Técnica por Regras")
        show_weather_risk(risk)

        st.subheader("8. Resumo interpretativo")
        with st.container(border=True):
            st.write(
                build_interpretive_summary(
                    weather_data,
                    risk,
                    analysis_result,
                    station_label,
                )
            )


def _select_station_for_historical_flow(
    data_source: str,
    weather_data: dict[str, object],
    station_catalog: pd.DataFrame,
    selection_key: str,
) -> tuple[dict[str, object] | None, str]:
    station_options = [station.to_dict() for _, station in station_catalog.iterrows()]
    if not station_options:
        return None, "Catalogo INMET sem estacoes disponiveis."

    option_labels = [str(station["station_label"]) for station in station_options]
    auto_station = None
    auto_message = ""
    if data_source == "OpenWeather":
        auto_station, auto_message = resolve_station_for_weather_data(
            weather_data,
            station_catalog,
        )

    if data_source == "OpenWeather" and auto_station is not None:
        auto_label = str(auto_station.get("station_label", ""))
        default_index = (
            option_labels.index(auto_label) if auto_label in option_labels else 0
        )
        with st.expander("Opcao avancada: alterar estacao INMET manualmente"):
            selected_label = st.selectbox(
                "Estacao INMET de referencia",
                option_labels,
                index=default_index,
                key=f"historical_station_override_{selection_key}",
            )
        if selected_label == auto_label:
            return auto_station, auto_message

        selected_station = station_options[option_labels.index(selected_label)]
        selected_station["station_resolution_method"] = "manual"
        return selected_station, "Estacao INMET alterada manualmente pelo usuario."

    if data_source == "OpenWeather":
        st.warning(auto_message)
        expander_title = "Opcao avancada: alterar estacao INMET manualmente"
        expander_expanded = True
    else:
        st.info(
            "Na entrada manual, selecione a estacao INMET de referencia para "
            "comparacao historica."
        )
        expander_title = "Estacao INMET de referencia"
        expander_expanded = True

    with st.expander(expander_title, expanded=expander_expanded):
        selected_label = st.selectbox(
            "Estacao INMET",
            option_labels,
            index=0,
            key=f"historical_station_manual_{selection_key}",
        )

    selected_station = station_options[option_labels.index(selected_label)]
    selected_station["station_resolution_method"] = "manual"
    return selected_station, "Estacao INMET selecionada manualmente."


def _show_station_selection_summary(
    selected_station: dict[str, object],
    station_label: str,
    historical_source_label: str,
    requested_period: str,
    record_count: int,
) -> None:
    col_station, col_code, col_state = st.columns([2, 1, 1])
    col_station.metric("Estacao INMET associada", station_label)
    col_code.metric("Codigo", str(selected_station["station_code"]))
    col_state.metric("UF", str(selected_station["state"]))

    altitude_m = selected_station.get("altitude_m") or selected_station.get(
        "altitude"
    )
    distance_km = selected_station.get("distance_km")
    details = [
        f"Fonte historica: {historical_source_label}",
        f"Periodo usado: {requested_period}",
        f"Registros carregados: {record_count}",
    ]
    if _is_number(distance_km):
        details.append(f"Distancia aproximada: {float(distance_km):.1f} km")
    if _is_number(altitude_m):
        details.append(f"Altitude: {float(altitude_m):.1f} m")
    st.caption(" | ".join(details))


def _show_loaded_history_summary(
    history: pd.DataFrame,
    selected_station: dict[str, object] | None,
    station_label: str,
    historical_source_label: str,
    requested_period: str,
) -> None:
    station_name = station_label
    station_state = ""
    station_code = ""
    altitude_m = None
    distance_km = None

    if selected_station is not None:
        station_name = str(selected_station.get("station_name", station_label))
        station_state = str(selected_station.get("state", ""))
        station_code = str(selected_station.get("station_code", ""))
        altitude_m = selected_station.get("altitude_m") or selected_station.get(
            "altitude"
        )
        distance_km = selected_station.get("distance_km")
    elif not history.empty:
        first_row = history.iloc[0]
        station_name = str(first_row.get("station_name", station_label))
        station_state = str(first_row.get("state", ""))
        station_code = str(first_row.get("station_code", ""))
        altitude_m = first_row.get("altitude_m")
        distance_km = None

    summary_rows = [
        {"Campo": "Estacao", "Valor": station_name},
        {"Campo": "UF", "Valor": station_state or "-"},
        {"Campo": "Codigo", "Valor": station_code or "-"},
        {
            "Campo": "Altitude",
            "Valor": (
                f"{float(altitude_m):.1f} m"
                if _is_number(altitude_m)
                else "-"
            ),
        },
        {
            "Campo": "Distancia aproximada",
            "Valor": (
                f"{float(distance_km):.1f} km"
                if _is_number(distance_km)
                else "-"
            ),
        },
        {
            "Campo": "Periodo solicitado",
            "Valor": requested_period,
        },
        {
            "Campo": "Periodo encontrado nos registros",
            "Valor": historical_period_text(history),
        },
        {"Campo": "Registros", "Valor": str(len(history))},
        {"Campo": "Fonte usada", "Valor": historical_source_label},
    ]
    st.dataframe(
        pd.DataFrame(format_safe_table_rows(summary_rows)),
        width="stretch",
        hide_index=True,
    )


def _show_historical_analysis_result(
    weather_data: dict[str, object] | None,
    analysis_result: dict[str, object],
) -> None:
    if not analysis_result.get("has_historical_data"):
        st.info(str(analysis_result.get("message", "Analise historica indisponivel.")))
        return

    anomalies = analysis_result.get("anomalies", [])
    if not isinstance(anomalies, list):
        anomalies = []

    st.write("Resultado da comparacao historica")
    col_anomalies, col_pressure = st.columns(2)
    col_anomalies.metric("Anomalias historicas", len(anomalies))
    pressure_comparison = analysis_result.get("pressure_comparison", {})
    if not isinstance(pressure_comparison, dict):
        pressure_comparison = {}
    pressure_status = (
        "avaliada" if pressure_comparison.get("evaluated") else "nao avaliada"
    )
    col_pressure.metric("Pressao", pressure_status)

    if anomalies:
        anomaly_rows = format_safe_table_rows(format_historical_anomalies(anomalies))
        for index, row in enumerate(anomaly_rows, start=1):
            st.markdown(
                "**Anomalia "
                f"{index}: {row['Tipo']}**  \n"
                f"Severidade: {row['Severidade']}  \n"
                f"Valor atual: {row['Valor atual']}  \n"
                f"Referencia historica: {row['Referencia historica']}  \n"
                f"Explicacao: {row['Justificativa']}"
            )
    else:
        st.info("Nenhuma anomalia historica identificada para os dados atuais.")

    summary_rows = build_historical_statistics_rows(weather_data, analysis_result)
    with st.expander("Percentis e estatisticas historicas usadas"):
        if summary_rows:
            st.dataframe(
                pd.DataFrame(format_safe_table_rows(summary_rows)),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Nao ha estatisticas suficientes para exibir o resumo.")

    with st.expander("Detalhes tecnicos da pressao"):
        st.write(build_pressure_reference_details(weather_data, pressure_comparison))


def _show_historical_details(
    weather_data: dict[str, object] | None,
    analysis_result: dict[str, object] | None,
    history: pd.DataFrame | None,
    selected_station: dict[str, object] | None,
    station_label: str,
    source_status: dict[str, str] | None,
    requested_period: str,
    station_message: str,
) -> None:
    if selected_station is not None:
        record_count = len(history) if history is not None else 0
        _show_station_selection_summary(
            selected_station,
            station_label or str(selected_station.get("station_code", "-")),
            (source_status or {}).get("source_label", "-"),
            requested_period,
            record_count,
        )
        if station_message:
            st.caption(station_message)
    else:
        st.info("Estacao INMET nao selecionada para esta consulta.")

    if source_status:
        with st.expander("Fonte historica usada"):
            st.write(source_status.get("status", ""))
            st.write(source_status.get("details", ""))
            if source_status.get("guidance"):
                st.write(source_status["guidance"])
                st.code("python scripts/download_inmet_database.py")

    if history is not None and not history.empty:
        with st.expander("Registros historicos carregados"):
            _show_loaded_history_summary(
                history,
                selected_station,
                station_label or "-",
                (source_status or {}).get("source_label", "-"),
                requested_period,
            )
            averages = history[
                [
                    "temperature",
                    "humidity",
                    "pressure",
                    "wind_speed",
                    "precipitation",
                ]
            ].mean()
            metric_columns = st.columns(5)
            metric_columns[0].metric(
                "Temperatura media",
                f"{averages['temperature']:.1f} C",
            )
            metric_columns[1].metric("Umidade media", f"{averages['humidity']:.1f}%")
            metric_columns[2].metric(
                "Pressao media",
                f"{averages['pressure']:.1f} hPa",
            )
            metric_columns[3].metric(
                "Vento medio",
                f"{averages['wind_speed']:.1f} km/h",
            )
            metric_columns[4].metric(
                "Precipitacao media",
                f"{averages['precipitation']:.1f} mm",
            )

    if analysis_result is None:
        st.info("Comparacao historica nao executada para esta consulta.")
        return

    _show_historical_analysis_result(weather_data, analysis_result)


def _first_variable(anomaly: dict[str, object]) -> str:
    variables = anomaly.get("variables_used", [])
    if isinstance(variables, list) and variables:
        variable = str(variables[0])
        if variable == "pressure_station_hpa":
            return "pressure"
        return variable
    return ""


def _unit_for_variable(variable: str) -> str:
    return VARIABLE_DISPLAY.get(variable, {}).get("unit", "")


def _format_value(value: object, unit: str = "") -> str:
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {_format_table_cell(item)}" for key, item in value.items()
        )
    if not _is_number(value):
        return "-"

    numeric_value = float(value)
    suffix = f" {unit}" if unit else ""
    return f"{numeric_value:.1f}{suffix}"


def _format_table_cell(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {_format_table_cell(item)}" for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_table_cell(item) for item in value)
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except (TypeError, ValueError):
        pass
    return str(value)


def _format_historical_reference(
    anomaly: dict[str, object],
    unit: str,
) -> str:
    reference = str(anomaly.get("historical_reference", ""))
    historical_value = anomaly.get("historical_value")
    if isinstance(historical_value, dict):
        formatted_value = ", ".join(
            f"{key}: {_format_value(value, unit)}"
            for key, value in historical_value.items()
        )
    else:
        formatted_value = _format_value(historical_value, unit)
    return f"{reference}: {formatted_value}" if reference else formatted_value


def _format_percentile_value(variable: str, stats: dict[str, object]) -> str:
    unit = VARIABLE_DISPLAY[variable]["unit"]
    percentile = VARIABLE_DISPLAY[variable]["percentile"]
    if variable == "pressure":
        return (
            f"{percentile}: "
            f"{_format_value(stats.get('p5'), unit)} a "
            f"{_format_value(stats.get('p95'), unit)}"
        )

    percentile_key = "p10" if variable == "humidity" else "p95"
    return f"{percentile}: {_format_value(stats.get(percentile_key), unit)}"


def _current_value_for_historical_variable(
    variable: str,
    weather_data: dict[str, object],
    pressure_comparison: dict[str, object],
) -> object | None:
    if variable == "pressure":
        if pressure_comparison.get("evaluated"):
            return pressure_comparison.get("pressure_station_hpa")
        return None
    return weather_data.get(variable)


def _interpret_historical_variable(
    variable: str,
    current_value: object | None,
    pressure_comparison: dict[str, object],
    anomalous_variables: set[str],
) -> str:
    if variable == "pressure" and not pressure_comparison.get("evaluated"):
        return "Pressao nao avaliada no mesmo referencial."
    if current_value is None:
        return "Dado atual indisponivel."
    if variable in anomalous_variables:
        return "Fora do padrao historico observado."
    return "Dentro do padrao historico observado."


def _anomalous_variables(analysis_result: dict[str, object]) -> set[str]:
    anomalies = analysis_result.get("anomalies", [])
    if not isinstance(anomalies, list):
        return set()

    variables = set()
    for anomaly in anomalies:
        if not isinstance(anomaly, dict):
            continue
        for variable in anomaly.get("variables_used", []):
            variable_name = str(variable)
            if variable_name == "pressure_station_hpa":
                variable_name = "pressure"
            variables.add(variable_name)
    return variables


def _is_number(value: object) -> bool:
    if isinstance(value, (dict, list, tuple, set)):
        return False
    if value is None or pd.isna(value):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _optional_float(value: object) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def apply_dashboard_style() -> None:
    """Aplica pequenos ajustes visuais sem criar dependencia de tema externo."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #111827 0%, #1f2937 54%, #0f172a 100%);
            color: #e5edf6;
        }
        .block-container {
            padding-top: 2.8rem;
            padding-bottom: 3rem;
        }
        .sac-hero {
            border: 1px solid rgba(125, 211, 252, 0.25);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-top: 1.4rem;
            margin-bottom: 1.2rem;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.82)),
                linear-gradient(90deg, rgba(14, 165, 233, 0.18), rgba(20, 184, 166, 0.10));
            box-shadow: 0 18px 45px rgba(2, 6, 23, 0.26);
        }
        .sac-hero h1 {
            margin: 0.2rem 0 0.45rem 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.02;
            letter-spacing: 0;
            color: #f8fafc;
        }
        .sac-hero p {
            margin: 0;
            max-width: 920px;
            color: #cbd5e1;
            font-size: 1.02rem;
        }
        .sac-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        .sac-pill {
            border: 1px solid rgba(125, 211, 252, 0.28);
            background: rgba(8, 47, 73, 0.36);
            color: #bae6fd;
            border-radius: 999px;
            padding: 0.34rem 0.72rem;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .sac-section-kicker {
            color: #67e8f9;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.6rem;
            text-transform: uppercase;
        }
        .sac-card {
            min-height: 118px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 1rem;
            background: rgba(30, 41, 59, 0.68);
            box-shadow: 0 14px 30px rgba(2, 6, 23, 0.20);
        }
        .sac-card-ml {
            border-color: rgba(34, 211, 238, 0.30);
            background: linear-gradient(180deg, rgba(22, 78, 99, 0.48), rgba(30, 41, 59, 0.72));
        }
        .sac-card-muted {
            background: rgba(30, 41, 59, 0.52);
            min-height: 92px;
        }
        .sac-card-label {
            color: #94a3b8;
            font-size: 0.80rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .sac-card-value {
            color: #f8fafc;
            font-size: 1.45rem;
            line-height: 1.12;
            font-weight: 800;
            margin-top: 0.5rem;
        }
        .sac-card-caption {
            color: #cbd5e1;
            font-size: 0.86rem;
            margin-top: 0.55rem;
        }
        .sac-risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid;
            border-radius: 999px;
            padding: 0.42rem 0.72rem;
            margin-top: 0.62rem;
            font-size: 1.02rem;
            font-weight: 900;
            letter-spacing: 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 23, 42, 0.70);
            border-color: rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
        }
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.42);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 10px;
            padding: 0.55rem 0.65rem;
        }
        .stAlert {
            border-radius: 10px;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_dashboard_header() -> None:
    """Exibe cabecalho visual para apresentacao do TCC."""
    st.markdown(
        """
        <section class="sac-hero">
            <div class="sac-section-kicker">TCC | Machine Learning supervisionado</div>
            <h1>Sistema Inteligente de Alerta Climático</h1>
            <p>
                Dados atuais da OpenWeather, histórico INMET e modelos supervisionados
                para apoiar a leitura técnica de risco climático por cidade.
            </p>
            <div class="sac-pill-row">
                <span class="sac-pill">OpenWeather atual</span>
                <span class="sac-pill">INMET histórico</span>
                <span class="sac-pill">RiskClassifier explicável</span>
                <span class="sac-pill">Predição ML local</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_dashboard_header() -> None:
    """Exibe cabecalho visual simples para apresentacao do TCC."""
    st.markdown(
        """
        <section class="sac-hero">
            <div class="sac-section-kicker">TCC | Machine Learning supervisionado</div>
            <h1>Sistema Inteligente de Alerta Climatico</h1>
            <p>
                Sistema academico que consulta dados meteorologicos atuais,
                compara com historico do INMET e aplica Machine Learning
                supervisionado para apoiar a classificacao de risco climatico.
            </p>
            <div class="sac-pill-row">
                <span class="sac-pill">Consulta por cidade</span>
                <span class="sac-pill">ML supervisionado local</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_initial_state_guidance() -> None:
    """Mostra orientacao curta antes da primeira busca."""
    with st.container(border=True):
        st.markdown("#### Fluxo da consulta")
        st.write(
            "Busque uma cidade, confira o resultado principal e abra os detalhes "
            "tecnicos apenas quando precisar ver regras, historico, features ou "
            "desempenho dos modelos."
        )


def main() -> None:
    """Executa o dashboard Streamlit do projeto."""
    settings = load_settings()
    ensure_runtime_artifact_directories()

    st.set_page_config(page_title="Sistema de Alerta Climatico", layout="wide")
    apply_dashboard_style()
    show_dashboard_header()

    form_is_compact = bool(st.session_state.get("dashboard_query"))
    query, submitted = build_query_form(settings, compact=form_is_compact)
    with st.expander("Opcao secundaria: entrada manual de dados atuais"):
        st.caption(
            "Use apenas quando nao houver chave da OpenWeather ou para demonstrar "
            "um cenario controlado."
        )
        manual_weather_data = build_manual_weather_data()

    if submitted:
        start_year = int(query["start_year"])
        end_year = int(query["end_year"])
        is_valid_period, period_message = validate_historical_period(
            start_year,
            end_year,
        )
        if not is_valid_period:
            st.session_state.pop("dashboard_query", None)
            st.session_state.pop("dashboard_weather_data", None)
            st.error(period_message)
        elif str(query["data_source"]) == "Entrada manual":
            weather_data = {
                **manual_weather_data,
                "city": str(query["city"]).strip() or settings.default_city,
                "country": str(query["country"]).strip().upper()
                or settings.default_country,
            }
            st.session_state["dashboard_query"] = query
            st.session_state["dashboard_weather_data"] = weather_data
        else:
            weather_data = fetch_openweather_weather_data(
                settings,
                str(query["city"]),
                str(query["country"]),
            )
            if weather_data:
                st.session_state["dashboard_query"] = query
                st.session_state["dashboard_weather_data"] = weather_data
            else:
                st.session_state.pop("dashboard_query", None)
                st.session_state.pop("dashboard_weather_data", None)

    active_query = st.session_state.get("dashboard_query")
    weather_data = st.session_state.get("dashboard_weather_data")

    if not active_query or not weather_data:
        show_initial_state_guidance()
        show_deploy_artifact_guidance(settings)
        return

    active_start_year = int(active_query["start_year"])
    active_end_year = int(active_query["end_year"])
    active_data_source = str(active_query["data_source"])

    risk = classify_weather_risk(weather_data)
    show_simplified_analysis_section(
        settings,
        weather_data,
        active_data_source,
        risk,
        active_start_year,
        active_end_year,
    )


if __name__ == "__main__":
    main()
