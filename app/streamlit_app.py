import pandas as pd
import streamlit as st

from src.alerts.risk_classifier import classify_weather_risk
from src.config.settings import load_settings


def build_manual_weather_data() -> dict[str, float]:
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
    }


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

    weather_data = build_manual_weather_data()
    risk = classify_weather_risk(weather_data)

    col_metrics, col_alert = st.columns([2, 1])

    with col_metrics:
        st.subheader("Dados meteorologicos")
        st.dataframe(pd.DataFrame([weather_data]), width="stretch")

    with col_alert:
        st.subheader("Alerta")
        st.metric("Nivel de risco", risk.level.upper())
        st.write(f"Evento: {risk.event}")
        st.info(risk.message)


if __name__ == "__main__":
    main()
