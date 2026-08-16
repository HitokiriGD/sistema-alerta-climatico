import math

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


def build_openweather_weather_data(settings: Settings) -> dict[str, object] | None:
    """Coleta dados meteorologicos atuais usando a OpenWeather."""
    city = st.text_input("Cidade", value=settings.default_city)

    if not settings.openweather_api_key:
        st.warning(
            "Configure OPENWEATHER_API_KEY no arquivo .env para usar a "
            "OpenWeather. A entrada manual continua disponivel."
        )
        return None

    if not city.strip():
        st.info("Informe uma cidade para buscar os dados meteorologicos atuais.")
        return None

    cached_city = st.session_state.get("openweather_city")
    cached_data = st.session_state.get("openweather_data")
    should_fetch = st.button("Buscar dados atuais")
    if not should_fetch:
        if cached_data and cached_city == city.strip():
            return cached_data
        st.info("Clique no botao para consultar a OpenWeather.")
        return None

    try:
        client = OpenWeatherClient(settings)
        payload = client.fetch_current_weather(city.strip())
        weather_data = normalize_weather_payload(payload)
        weather_data["city"] = str(weather_data.get("city", city.strip()))
        weather_data["data_source"] = "OpenWeather"
        st.session_state["openweather_city"] = city.strip()
        st.session_state["openweather_data"] = weather_data
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
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Temperatura",
        _format_value(weather_data.get("temperature"), "C"),
    )
    metric_columns[1].metric(
        "Umidade",
        _format_value(weather_data.get("humidity"), "%"),
    )
    metric_columns[2].metric(
        "Vento",
        _format_value(weather_data.get("wind_speed"), "km/h"),
    )
    pressure_value = weather_data.get(
        "pressure_station_hpa",
        weather_data.get("pressure_sea_level_hpa", weather_data.get("pressure")),
    )
    metric_columns[3].metric("Pressao", _format_value(pressure_value, "hPa"))

    with st.expander("Detalhes dos dados atuais"):
        st.dataframe(
            pd.DataFrame(build_current_weather_rows(weather_data)),
            width="stretch",
            hide_index=True,
        )
    if weather_data.get("pressure_reference") == "openweather_grnd_level":
        st.caption(
            "Pressao atual para comparacao historica: pressao ao nivel da "
            "estacao informada por grnd_level da OpenWeather."
        )
    elif "pressure_sea_level_hpa" in weather_data:
        st.caption(
            "A pressao principal da OpenWeather e ao nivel do mar. Para "
            "comparacao com o INMET historico, o sistema usa grnd_level quando "
            "disponivel ou estima a pressao ao nivel da estacao pela altitude."
        )


def show_weather_risk(risk: WeatherRisk) -> None:
    """Exibe o resultado explicavel do classificador por regras."""
    st.metric("Nivel de risco", risk.risk_level.upper())
    st.write(f"Evento climatico: {risk.event_type}")
    st.info(risk.reason)
    st.caption(
        "Classificacao heuristica do prototipo academico; nao substitui "
        "alertas oficiais de defesa civil ou orgaos meteorologicos."
    )

    st.write("Variaveis consideradas")
    st.dataframe(pd.DataFrame([risk.variables]), width="stretch")

    if risk.triggered_rules:
        st.write("Regras ativadas")
        for rule in risk.triggered_rules:
            st.write(f"- {rule}")
    else:
        st.write("Regras ativadas: nenhuma regra relevante.")

    st.write("Orientacoes gerais")
    for recommendation in risk.recommendations:
        st.write(f"- {recommendation}")


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
        "Campo pressure_station_hpa usado diretamente "
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


def show_inmet_historical_section(
    settings: Settings,
    weather_data: dict[str, object] | None,
    data_source: str,
) -> None:
    """Exibe comparacao historica local do INMET sem gerar alerta principal."""
    st.divider()
    st.subheader("Comparacao com historico INMET")
    st.caption(
        "O INMET e usado como base historica oficial. A comparacao complementa "
        "o alerta por regras e nao substitui os dados atuais."
    )

    col_start, col_end = st.columns(2)
    with col_start:
        start_year = st.number_input(
            "Ano inicial",
            min_value=2000,
            max_value=2100,
            value=settings.inmet_historical_start_year,
            step=1,
        )
    with col_end:
        end_year = st.number_input(
            "Ano final",
            min_value=2000,
            max_value=2100,
            value=settings.inmet_historical_end_year,
            step=1,
        )

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
        return

    if historical_source == "database":
        metadata = get_database_metadata(settings.inmet_database_path)
        st.success("Base historica processada encontrada.")
        st.caption(
            "Consulta historica usando DuckDB filtrado por estacao e periodo. "
            f"Estacoes: {metadata['station_count']}; "
            f"registros: {metadata['record_count']}."
        )
    else:
        st.info("Base processada nao encontrada; usando ZIPs locais.")

    selected_station, station_message = _select_station_for_historical_flow(
        data_source,
        weather_data,
        station_catalog,
    )
    if selected_station is None:
        st.warning(station_message)
        return

    station_code = str(selected_station["station_code"])
    station_label = str(selected_station.get("station_label", station_code))
    _show_station_selection_summary(selected_station, station_label)

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
        return

    if history.empty:
        st.warning("Nenhum registro historico foi carregado para esta estacao.")
        return

    _show_historical_flow_status(data_source, station_label, station_message)
    _show_loaded_history_summary(history, selected_station, station_label)

    averages = history[
        ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]
    ].mean()
    metric_columns = st.columns(5)
    metric_columns[0].metric("Temperatura media", f"{averages['temperature']:.1f} C")
    metric_columns[1].metric("Umidade media", f"{averages['humidity']:.1f}%")
    metric_columns[2].metric("Pressao media", f"{averages['pressure']:.1f} hPa")
    metric_columns[3].metric("Vento medio", f"{averages['wind_speed']:.1f} km/h")
    metric_columns[4].metric(
        "Precipitacao media",
        f"{averages['precipitation']:.1f} mm",
    )
    st.caption(
        "A pressao historica do INMET esta ao nivel da estacao. Comparacoes "
        "com OpenWeather devem usar grnd_level ou pressao estimada ao nivel da "
        "estacao pela altitude."
    )

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
    st.success("Analise historica executada.")
    _show_historical_analysis_result(weather_data, analysis_result)


def _select_station_for_historical_flow(
    data_source: str,
    weather_data: dict[str, object],
    station_catalog: pd.DataFrame,
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
                key="historical_station_override_openweather",
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
            key=f"historical_station_manual_{data_source}",
        )

    selected_station = station_options[option_labels.index(selected_label)]
    selected_station["station_resolution_method"] = "manual"
    return selected_station, "Estacao INMET selecionada manualmente."


def _show_historical_flow_status(
    data_source: str,
    station_label: str,
    station_message: str,
) -> None:
    if data_source == "OpenWeather":
        st.success("Dados atuais obtidos via OpenWeather.")
    else:
        st.success("Dados atuais informados manualmente.")
    st.write(f"Estacao INMET associada: {station_label}.")
    st.caption(station_message)
    st.success("Historico INMET carregado para comparacao.")


def _show_station_selection_summary(
    selected_station: dict[str, object],
    station_label: str,
) -> None:
    st.write(f"Estacao selecionada: {station_label}")
    st.write(f"Codigo da estacao: {selected_station['station_code']}")
    st.write(f"UF: {selected_station['state']}")
    altitude_m = selected_station.get("altitude_m") or selected_station.get(
        "altitude"
    )
    if altitude_m is not None and pd.notna(altitude_m):
        st.write(f"Altitude da estacao: {float(altitude_m):.1f} m")


def _show_loaded_history_summary(
    history: pd.DataFrame,
    selected_station: dict[str, object] | None,
    station_label: str,
) -> None:
    station_name = station_label
    station_state = ""
    station_code = ""
    altitude_m = None

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
            "Campo": "Periodo historico",
            "Valor": historical_period_text(history),
        },
        {"Campo": "Registros", "Valor": str(len(history))},
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
        st.dataframe(
            pd.DataFrame(anomaly_rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("Nenhuma anomalia historica identificada para os dados atuais.")

    summary_rows = build_historical_statistics_rows(weather_data, analysis_result)
    with st.expander("Estatisticas historicas usadas na comparacao", expanded=True):
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


def main() -> None:
    """Executa o dashboard Streamlit do projeto."""
    settings = load_settings()

    st.set_page_config(page_title="Sistema de Alerta Climatico", layout="wide")
    st.title("Sistema Inteligente de Alerta Climatico")
    st.caption(
        "Prototipo academico para classificacao inicial de risco por cidade."
    )

    st.sidebar.header("Configuracao")
    st.sidebar.write(f"Cidade padrao: {settings.default_city}")
    st.sidebar.write(f"Pais padrao: {settings.default_country}")

    data_source = st.sidebar.radio(
        "Fonte dos dados atuais",
        ["Entrada manual", "OpenWeather"],
    )

    if data_source == "Entrada manual":
        weather_data = build_manual_weather_data()
    else:
        weather_data = build_openweather_weather_data(settings)

    col_metrics, col_alert = st.columns([2, 1])

    with col_metrics:
        st.subheader("Dados meteorologicos atuais")
        if weather_data:
            show_weather_data(weather_data)
        else:
            st.info("Nenhum dado meteorologico atual carregado.")

    with col_alert:
        st.subheader("Alerta")
        if weather_data:
            risk = classify_weather_risk(weather_data)
            show_weather_risk(risk)
        else:
            st.info("Carregue dados meteorologicos atuais para gerar o alerta.")

    show_inmet_historical_section(settings, weather_data, data_source)


if __name__ == "__main__":
    main()
