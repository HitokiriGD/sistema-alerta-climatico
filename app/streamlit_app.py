import math
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st

from src.alerts.historical_analyzer import HistoricalAnalyzer
from src.alerts.risk_classifier import WeatherRisk
from src.alerts.risk_classifier import classify_weather_risk
from src.config.settings import Settings
from src.config.settings import load_settings
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
from src.ml.predict import load_model_bundle
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


def build_query_form(settings: Settings) -> tuple[dict[str, object], bool]:
    """Monta o formulario principal de consulta do dashboard."""
    with st.container(border=True):
        st.subheader("Consulta")
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
        "Classificacao por regras explicaveis do prototipo academico. "
        "A previsao por Machine Learning aparece em secao separada."
    )

    st.write("Recomendacoes gerais")
    for recommendation in risk.recommendations:
        st.write(f"- {recommendation}")

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


def show_ml_prediction_section(weather_data: dict[str, object] | None) -> None:
    """Exibe a previsao ML treinada localmente, quando disponivel."""
    prediction_result = get_ml_prediction_for_dashboard(weather_data)

    with st.container(border=True):
        if not prediction_result.get("available"):
            st.info(str(prediction_result.get("error_message", "")))
            return

        if prediction_result.get("prediction"):
            col_prediction, col_model, col_metric = st.columns(3)
            col_prediction.metric(
                "Risk level previsto",
                str(prediction_result["prediction"]).upper(),
            )
            col_model.metric(
                "Modelo selecionado",
                str(prediction_result.get("model_name") or "Nao informado"),
            )
            col_metric.metric(
                "Metrica de selecao",
                str(prediction_result.get("selection_metric") or "Nao informada"),
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

        st.caption(str(prediction_result.get("methodological_note", "")))


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
            "O dashboard usara os ZIPs locais do INMET como fallback quando "
            "eles estiverem disponiveis."
        ),
        "guidance": (
            "Para baixar a base processada, execute: "
            "python scripts/download_inmet_database.py"
        ),
    }


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

    client = InmetClient(settings)
    try:
        station_catalog, historical_source = load_station_catalog_for_app(
            settings,
            client,
        )
    except InmetHistoricalDataError as error:
        st.warning(str(error))
        if risk is not None:
            st.subheader("3. Alerta por regras")
            show_weather_risk(risk)
            st.subheader("4. Previsão por Machine Learning")
            show_ml_prediction_section(weather_data)
            st.subheader("5. Resumo")
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
            st.subheader("3. Alerta por regras")
            show_weather_risk(risk)
            st.subheader("4. Previsão por Machine Learning")
            show_ml_prediction_section(weather_data)
            st.subheader("5. Resumo")
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
        st.warning(str(error))
        if risk is not None:
            st.subheader("3. Alerta por regras")
            show_weather_risk(risk)
            st.subheader("4. Previsão por Machine Learning")
            show_ml_prediction_section(weather_data)
            st.subheader("5. Resumo")
            st.write(
                build_interpretive_summary(weather_data, risk, None, station_label)
            )
        return

    if history.empty:
        st.warning("Nenhum registro historico foi carregado para esta estacao.")
        if risk is not None:
            st.subheader("3. Alerta por regras")
            show_weather_risk(risk)
            st.subheader("4. Previsão por Machine Learning")
            show_ml_prediction_section(weather_data)
            st.subheader("5. Resumo")
            st.write(
                build_interpretive_summary(weather_data, risk, None, station_label)
            )
        return

    st.subheader("2. Base historica associada")
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

    if risk is not None:
        st.subheader("3. Alerta por regras")
        show_weather_risk(risk)

    st.subheader("4. Comparacao historica")

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
    st.subheader("5. Previsão por Machine Learning")
    show_ml_prediction_section(weather_data)

    if risk is not None:
        st.subheader("6. Resumo interpretativo")
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
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(38, 39, 48, 0.72);
            border-color: rgba(130, 148, 170, 0.28);
            border-radius: 8px;
        }
        div[data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.24);
            border-radius: 8px;
            padding: 0.45rem 0.55rem;
        }
        .stAlert {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Executa o dashboard Streamlit do projeto."""
    settings = load_settings()

    st.set_page_config(page_title="Sistema de Alerta Climatico", layout="wide")
    apply_dashboard_style()
    st.title("Sistema Inteligente de Alerta Climatico")
    st.write(
        "Consulta dados atuais da OpenWeather, associa uma estacao historica "
        "do INMET e apresenta alerta por regras e previsao ML local."
    )
    st.caption(
        "O modelo ML e carregado apenas quando existir em data/models/."
    )

    query, submitted = build_query_form(settings)
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
        st.info("Preencha o formulario e clique em buscar para iniciar a analise.")
        return

    active_start_year = int(active_query["start_year"])
    active_end_year = int(active_query["end_year"])
    active_data_source = str(active_query["data_source"])

    st.subheader("1. Dados atuais")
    with st.container(border=True):
        st.caption(
            "Consulta exibida: "
            f"{weather_data.get('city', '-')}, "
            f"{format_requested_period(active_start_year, active_end_year)}."
        )
        show_weather_data(weather_data)

    risk = classify_weather_risk(weather_data)
    show_inmet_historical_section(
        settings,
        weather_data,
        active_data_source,
        risk,
        active_start_year,
        active_end_year,
    )


if __name__ == "__main__":
    main()
