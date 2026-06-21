import pandas as pd


WEATHER_COLUMNS = [
    "temperature",
    "feels_like",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
]


def clean_weather_data(data: pd.DataFrame) -> pd.DataFrame:
    """Seleciona variaveis meteorologicas e trata valores ausentes."""
    missing_columns = [column for column in WEATHER_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Colunas ausentes: {', '.join(missing_columns)}")

    cleaned_data = data.loc[:, WEATHER_COLUMNS].copy()
    cleaned_data = cleaned_data.apply(pd.to_numeric, errors="coerce")
    return cleaned_data.fillna(cleaned_data.median(numeric_only=True))


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Prepara a matriz de atributos usada pelos modelos."""
    return clean_weather_data(data)
