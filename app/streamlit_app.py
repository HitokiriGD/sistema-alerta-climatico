import pandas as pd
import requests
import streamlit as st

from src.alerts.risk_classifier import WeatherRisk
from src.alerts.risk_classifier import classify_weather_risk
from src.config.settings import Settings
from src.config.settings import load_settings
from src.data.inmet_client import InmetClient
from src.data.inmet_client import InmetHistoricalDataError
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

    if not st.button("Buscar dados atuais"):
        st.info("Clique no botao para consultar a OpenWeather.")
        return None

    try:
        client = OpenWeatherClient(settings)
        payload = client.fetch_current_weather(city.strip())
        return normalize_weather_payload(payload)
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
    ordered_data = {
        field: weather_data.get(field)
        for field in STANDARD_WEATHER_FIELDS
    }
    st.dataframe(pd.DataFrame([ordered_data]), width="stretch")
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


def show_inmet_historical_section(settings: Settings) -> None:
    """Exibe consulta historica local do INMET sem gerar alerta."""
    st.divider()
    st.subheader("Base historica INMET")
    st.caption(
        "O INMET e usado como base historica oficial para comparacao e futuro "
        "treinamento dos modelos. Ele nao e usado como clima atual."
    )

    client = InmetClient(settings)
    selected_station: dict[str, object] | None = None
    station_code = ""
    station_label = ""

    try:
        station_options = client.list_station_options()
        option_labels = [
            str(station["station_label"])
            for station in station_options
        ]
        selected_label = st.selectbox(
            "Estacao INMET",
            option_labels,
            index=0,
        )
        selected_station = station_options[option_labels.index(selected_label)]
        station_code = str(selected_station["station_code"])
        station_label = selected_label
    except InmetHistoricalDataError as error:
        st.warning(str(error))

    with st.expander("Opcao avancada: informar codigo manualmente"):
        manual_station_code = st.text_input(
            "Codigo da estacao INMET",
            value="",
            placeholder="Exemplo: A101",
        )
        if manual_station_code.strip():
            station_code = manual_station_code.strip().upper()
            station_label = f"{station_code} - codigo informado manualmente"
            selected_station = None

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

    if selected_station is not None:
        st.write(f"Estacao selecionada: {station_label}")
        st.write(f"Codigo da estacao: {selected_station['station_code']}")
        st.write(f"UF: {selected_station['state']}")
        altitude_m = selected_station.get("altitude_m") or selected_station.get(
            "altitude"
        )
        if altitude_m is not None and pd.notna(altitude_m):
            st.write(f"Altitude da estacao: {float(altitude_m):.1f} m")

    if not st.button("Carregar historico INMET"):
        st.info("Selecione uma estacao e carregue a base historica local do INMET.")
        return

    try:
        if not station_code:
            st.warning("Selecione uma estacao ou informe um codigo valido.")
            return

        if selected_station is None:
            st.write(f"Estacao selecionada: {station_label}")
        history = client.load_station_history(
            station_code,
            int(start_year),
            int(end_year),
        )
    except InmetHistoricalDataError as error:
        st.warning(str(error))
        return

    if history.empty:
        st.warning("Nenhum registro historico foi carregado para esta estacao.")
        return

    st.success("Historico INMET carregado com sucesso.")
    st.write(
        "Periodo disponivel: "
        f"{history['date'].min()} a {history['date'].max()}"
    )
    st.write(f"Quantidade de registros: {len(history)}")

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

    show_inmet_historical_section(settings)


if __name__ == "__main__":
    main()
