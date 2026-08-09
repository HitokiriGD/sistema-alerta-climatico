import unicodedata
from typing import Any

import pandas as pd


WEATHER_COLUMNS = [
    "temperature",
    "feels_like",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
]
REQUIRED_WEATHER_COLUMNS = [
    *WEATHER_COLUMNS,
    "datetime",
    "source",
]
QUALITY_FLAG_COLUMN = "quality_flag"
QUALITY_COMPLETE = "complete"
QUALITY_INCOMPLETE = "incomplete"

NUMERIC_LIMITS = {
    "temperature": (-20.0, 60.0),
    "feels_like": (-30.0, 70.0),
    "humidity": (0.0, 100.0),
    "precipitation": (0.0, None),
    "wind_speed": (0.0, None),
    "pressure": (0.0, None),
}

COLUMN_NAME_MAPPING = {
    "codigo_estacao": "station_code",
    "codigo_wmo": "station_code",
    "estacao": "station_name",
    "nome_estacao": "station_name",
    "uf": "state",
    "data": "date",
    "hora_utc": "hour",
    "data_hora": "datetime",
    "datetime": "datetime",
    "timestamp": "datetime",
    "origem": "source",
    "source": "source",
    "temperatura": "temperature",
    "temperatura_c": "temperature",
    "temperatura_do_ar_bulbo_seco_horaria_c": "temperature",
    "temperatura_do_ar_bulbo_seco_horaria_degc": "temperature",
    "temp": "temperature",
    "temperature": "temperature",
    "sensacao_termica": "feels_like",
    "sensacao_termica_c": "feels_like",
    "feels_like": "feels_like",
    "umidade": "humidity",
    "umidade_relativa": "humidity",
    "umidade_relativa_do_ar_horaria": "humidity",
    "humidity": "humidity",
    "precipitacao": "precipitation",
    "precipitacao_total_horario_mm": "precipitation",
    "rain": "precipitation",
    "precipitation": "precipitation",
    "vento_velocidade_horaria_ms": "wind_speed",
    "velocidade_vento": "wind_speed",
    "wind_speed": "wind_speed",
    "pressao": "pressure",
    "pressao_atmosferica_ao_nivel_da_estacao_horaria_mb": "pressure",
    "pressure": "pressure",
}


def standardize_weather_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes conhecidos de colunas meteorologicas."""
    standardized_data = data.copy()
    rename_map = {}

    for column in standardized_data.columns:
        standard_name = COLUMN_NAME_MAPPING.get(_normalize_column_key(column))
        if standard_name and standard_name not in standardized_data.columns:
            rename_map[column] = standard_name

    return standardized_data.rename(columns=rename_map)


def convert_decimal_comma_to_float(
    data: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Converte valores numericos com virgula decimal para float."""
    converted_data = data.copy()
    numeric_columns = columns or WEATHER_COLUMNS

    for column in numeric_columns:
        if column in converted_data:
            converted_data[column] = converted_data[column].map(_to_float)

    return converted_data


def validate_required_columns(data: pd.DataFrame) -> None:
    """Valida se as colunas obrigatorias estao presentes."""
    missing_columns = [
        column for column in REQUIRED_WEATHER_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(missing_columns)}")


def validate_plausible_limits(data: pd.DataFrame) -> pd.DataFrame:
    """Substitui valores meteorologicos fora dos limites plausiveis por nulo."""
    validated_data = data.copy()

    for column, (minimum, maximum) in NUMERIC_LIMITS.items():
        if column not in validated_data:
            continue

        values = pd.to_numeric(validated_data[column], errors="coerce")
        invalid_mask = pd.Series(False, index=validated_data.index)

        if minimum is not None:
            invalid_mask = invalid_mask | (values < minimum)
        if maximum is not None:
            invalid_mask = invalid_mask | (values > maximum)

        validated_data[column] = values.mask(invalid_mask)

    return validated_data


def handle_missing_weather_values(data: pd.DataFrame) -> pd.DataFrame:
    """Remove registros essenciais incompletos e marca qualidade dos demais."""
    cleaned_data = data.copy()
    cleaned_data = cleaned_data.dropna(subset=["datetime", "temperature"])

    complete_mask = cleaned_data[REQUIRED_WEATHER_COLUMNS].notna().all(axis=1)
    cleaned_data[QUALITY_FLAG_COLUMN] = complete_mask.map(
        {True: QUALITY_COMPLETE, False: QUALITY_INCOMPLETE}
    )
    return cleaned_data


def sort_weather_data(data: pd.DataFrame) -> pd.DataFrame:
    """Ordena registros meteorologicos por data e hora."""
    return data.sort_values("datetime").reset_index(drop=True)


def remove_weather_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicidades por estacao e data/hora quando houver estacao."""
    if "station_code" not in data.columns:
        return data.reset_index(drop=True)

    return data.drop_duplicates(
        subset=["station_code", "datetime"],
        keep="first",
    ).reset_index(drop=True)


def preprocess_weather_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Executa o pipeline central de pre-tratamento meteorologico."""
    processed_data = standardize_weather_columns(data)
    validate_required_columns(processed_data)

    processed_data = convert_decimal_comma_to_float(processed_data)
    processed_data["datetime"] = pd.to_datetime(
        processed_data["datetime"],
        errors="coerce",
    )
    processed_data = validate_plausible_limits(processed_data)
    processed_data = handle_missing_weather_values(processed_data)
    processed_data = sort_weather_data(processed_data)
    processed_data = remove_weather_duplicates(processed_data)

    return processed_data


def clean_weather_data(data: pd.DataFrame) -> pd.DataFrame:
    """Seleciona variaveis meteorologicas e trata valores ausentes para modelos."""
    missing_columns = [column for column in WEATHER_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Colunas ausentes: {', '.join(missing_columns)}")

    cleaned_data = data.loc[:, WEATHER_COLUMNS].copy()
    cleaned_data = convert_decimal_comma_to_float(cleaned_data)
    return cleaned_data.fillna(cleaned_data.median(numeric_only=True))


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Prepara a matriz de atributos usada pelos modelos."""
    return clean_weather_data(data)


def _normalize_column_key(column: Any) -> str:
    column_name = str(column).strip().lower()
    replacements = {
        " ": "_",
        "-": "_",
        "/": "_",
        "(": "",
        ")": "",
        "%": "",
        ",": "",
        ";": "",
        ".": "",
        ":": "",
        "\N{DEGREE SIGN}": "",
    }

    normalized = "".join(
        char
        for char in unicodedata.normalize("NFKD", column_name)
        if not unicodedata.combining(char)
    )
    for old_value, new_value in replacements.items():
        normalized = normalized.replace(old_value, new_value)

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def _to_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, int | float):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")

    numeric_value = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric_value):
        return None
    return float(numeric_value)
