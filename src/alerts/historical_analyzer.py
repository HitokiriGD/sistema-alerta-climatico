from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def normalize_pressure_to_station_level(
    pressure_sea_level_hpa: float,
    altitude_m: float,
) -> float:
    """Estima pressao ao nivel da estacao pela formula barometrica."""
    return pressure_sea_level_hpa * (1 - 0.0065 * altitude_m / 288.15) ** 5.255


def resolve_current_pressure_for_historical_comparison(
    current_data: Dict[str, Any],
    station_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a pressao atual no mesmo referencial do historico INMET."""
    station_pressure = _to_optional_float(current_data.get("pressure_station_hpa"))
    if station_pressure is not None:
        return {
            "evaluated": True,
            "pressure_station_hpa": station_pressure,
            "reference_used": "pressure_station_hpa",
            "reference_label": "pressao ao nivel da estacao",
            "is_estimated": False,
            "message": "Pressao ao nivel da estacao usada na comparacao historica.",
        }

    altitude_m = _resolve_altitude_m(station_metadata)
    sea_level_pressure = _to_optional_float(current_data.get("pressure_sea_level_hpa"))
    if sea_level_pressure is not None:
        if altitude_m is None:
            return {
                "evaluated": False,
                "pressure_station_hpa": None,
                "reference_used": "pressure_sea_level_hpa",
                "reference_label": "pressao ao nivel do mar",
                "is_estimated": False,
                "message": (
                    "Pressao nao avaliada: a OpenWeather informou pressao ao "
                    "nivel do mar e nao ha altitude da estacao INMET para "
                    "estimar a pressao ao nivel da estacao."
                ),
            }

        estimated_pressure = normalize_pressure_to_station_level(
            sea_level_pressure,
            altitude_m,
        )
        return {
            "evaluated": True,
            "pressure_station_hpa": estimated_pressure,
            "reference_used": "pressure_estimated_station_hpa",
            "reference_label": "pressao estimada ao nivel da estacao",
            "is_estimated": True,
            "altitude_m": altitude_m,
            "message": (
                "Pressao estimada ao nivel da estacao usada na comparacao "
                f"historica, com altitude de {altitude_m:.1f} m."
            ),
        }

    legacy_reference = current_data.get("pressure_reference")
    legacy_pressure = _to_optional_float(current_data.get("pressure"))
    if (
        legacy_reference in {"manual_station_level", "station_level"}
        and legacy_pressure is not None
    ):
        return {
            "evaluated": True,
            "pressure_station_hpa": legacy_pressure,
            "reference_used": "pressure_legacy_station_hpa",
            "reference_label": "pressao ao nivel da estacao",
            "is_estimated": False,
            "message": (
                "Pressao do campo legado usada como pressao ao nivel da "
                "estacao. Para OpenWeather, prefira pressure_sea_level_hpa "
                "e pressure_station_hpa."
            ),
        }

    return {
        "evaluated": False,
        "pressure_station_hpa": None,
        "reference_used": None,
        "reference_label": None,
        "is_estimated": False,
        "message": (
            "Pressao nao avaliada: informe pressure_station_hpa ou forneca "
            "pressure_sea_level_hpa com altitude da estacao INMET."
        ),
    }


class HistoricalAnalyzer:
    """Analisador estatistico para comparar dados atuais com o historico INMET."""

    COLUMN_MAPPING = {
        "temperature": ["temperature", "temp_c", "temperatura", "TEM_INS"],
        "humidity": ["humidity", "umidade", "UMD_INS"],
        "precipitation": ["precipitation", "rain_mm", "chuva", "CHUVA"],
        "wind_speed": ["wind_speed", "wind_ms", "vento", "VEN_VEL"],
        "pressure": ["pressure", "pressure_hpa", "pressao", "PRE_INS"],
    }

    def __init__(self, min_samples_seasonal: int = 30):
        self.min_samples_seasonal = min_samples_seasonal

    def _resolve_column(self, df: pd.DataFrame, var_name: str) -> Optional[str]:
        candidates = self.COLUMN_MAPPING.get(var_name, [var_name])
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def calculate_statistics(
        self, df_historical: pd.DataFrame, month: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        if df_historical is None or df_historical.empty:
            return {}

        df = df_historical.copy()

        if month is not None:
            date_columns = ["datetime", "date", "data", "DT_MEDICAO"]
            date_col = next((c for c in date_columns if c in df.columns), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df_month = df[df[date_col].dt.month == month]
                if len(df_month) >= self.min_samples_seasonal:
                    df = df_month

        stats: Dict[str, Dict[str, float]] = {}

        for var_name in self.COLUMN_MAPPING.keys():
            col = self._resolve_column(df, var_name)
            if not col or df[col].dropna().empty:
                continue

            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue

            stats[var_name] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "p10": float(np.percentile(series, 10)),
                "p25": float(np.percentile(series, 25)),
                "p75": float(np.percentile(series, 75)),
                "p90": float(np.percentile(series, 90)),
                "p95": float(np.percentile(series, 95)),
                "p5": float(np.percentile(series, 5)),
            }

        return stats

    def analyze(
        self,
        current_data: Dict[str, Any],
        df_historical: pd.DataFrame,
        current_date: Optional[datetime] = None,
        station_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if df_historical is None or df_historical.empty or not current_data:
            return {
                "has_historical_data": False,
                "anomalies": [],
                "statistics": {},
                "pressure_comparison": _not_evaluated_pressure(
                    "Pressao nao avaliada: historico ausente ou dados atuais "
                    "incompletos."
                ),
                "message": "Historico ausente ou dados atuais incompletos.",
            }

        target_month = current_date.month if current_date else datetime.now().month
        stats = self.calculate_statistics(df_historical, month=target_month)

        if not stats:
            return {
                "has_historical_data": False,
                "anomalies": [],
                "statistics": {},
                "pressure_comparison": _not_evaluated_pressure(
                    "Pressao nao avaliada: estatisticas historicas insuficientes."
                ),
                "message": (
                    "Dados historicos insuficientes para calculo de "
                    "estatisticas."
                ),
            }

        anomalies: List[Dict[str, Any]] = []

        temp = current_data.get("temperature", current_data.get("temp_c"))
        hum = current_data.get("humidity")
        precip = current_data.get("precipitation", current_data.get("rain_mm", 0.0))
        wind = current_data.get("wind_speed", current_data.get("wind_ms"))
        station_metadata = station_metadata or _metadata_from_historical_data(
            df_historical
        )
        pressure_comparison = resolve_current_pressure_for_historical_comparison(
            current_data,
            station_metadata,
        )

        if temp is not None and "temperature" in stats:
            s_temp = stats["temperature"]
            if temp >= s_temp["p95"]:
                anomalies.append({
                    "anomaly_type": "HIGH_TEMPERATURE",
                    "severity": "CRITICAL" if temp > s_temp["max"] else "HIGH",
                    "reason": (
                        f"Temperatura atual ({temp:.1f} C) acima do percentil "
                        f"95 historico ({s_temp['p95']:.1f} C)."
                    ),
                    "historical_reference": "Percentil 95",
                    "current_value": temp,
                    "historical_value": s_temp["p95"],
                    "variables_used": ["temperature"],
                })

        if hum is not None and "humidity" in stats:
            s_hum = stats["humidity"]
            if hum <= s_hum["p10"]:
                anomalies.append({
                    "anomaly_type": "LOW_HUMIDITY",
                    "severity": "CRITICAL" if hum < s_hum["min"] else "HIGH",
                    "reason": (
                        f"Umidade atual ({hum:.1f}%) abaixo do percentil "
                        f"10 historico ({s_hum['p10']:.1f}%)."
                    ),
                    "historical_reference": "Percentil 10",
                    "current_value": hum,
                    "historical_value": s_hum["p10"],
                    "variables_used": ["humidity"],
                })

        if precip is not None and precip > 0 and "precipitation" in stats:
            s_precip = stats["precipitation"]
            if precip >= s_precip["p95"]:
                anomalies.append({
                    "anomaly_type": "HIGH_PRECIPITATION",
                    "severity": "HIGH",
                    "reason": (
                        f"Precipitacao atual ({precip:.1f} mm) acima do "
                        f"percentil 95 historico ({s_precip['p95']:.1f} mm)."
                    ),
                    "historical_reference": "Percentil 95",
                    "current_value": precip,
                    "historical_value": s_precip["p95"],
                    "variables_used": ["precipitation"],
                })

        if wind is not None and "wind_speed" in stats:
            s_wind = stats["wind_speed"]
            if wind >= s_wind["p95"]:
                anomalies.append({
                    "anomaly_type": "HIGH_WIND",
                    "severity": "HIGH",
                    "reason": (
                        f"Velocidade do vento ({wind:.1f} km/h) acima do "
                        f"percentil 95 historico ({s_wind['p95']:.1f} km/h)."
                    ),
                    "historical_reference": "Percentil 95",
                    "current_value": wind,
                    "historical_value": s_wind["p95"],
                    "variables_used": ["wind_speed"],
                })

        if "pressure" in stats:
            pressure_comparison = self._evaluate_pressure_anomaly(
                pressure_comparison,
                stats["pressure"],
                anomalies,
            )
        else:
            pressure_comparison = _not_evaluated_pressure(
                "Pressao nao avaliada: historico INMET sem coluna de pressao."
            )

        if (
            temp is not None
            and hum is not None
            and "humidity" in stats
            and "temperature" in stats
        ):
            if hum <= stats["humidity"]["p25"] and temp >= stats["temperature"]["p75"]:
                is_windy = (
                    wind is not None
                    and "wind_speed" in stats
                    and wind >= stats["wind_speed"]["median"]
                )
                anomalies.append({
                    "anomaly_type": "POTENTIAL_FIRE_RISK",
                    "severity": "CRITICAL" if is_windy else "HIGH",
                    "reason": (
                        "Condicao favoravel a risco potencial de incendio "
                        "(baixa umidade, alta temperatura e vento acima do padrao)."
                    ),
                    "historical_reference": "Combinacao P25 Umidade + P75 Temperatura",
                    "current_value": {"temp": temp, "humidity": hum, "wind": wind},
                    "historical_value": {
                        "p25_hum": stats["humidity"]["p25"],
                        "p75_temp": stats["temperature"]["p75"],
                    },
                    "variables_used": ["temperature", "humidity", "wind_speed"],
                })

        return {
            "has_historical_data": True,
            "anomalies": anomalies,
            "statistics": stats,
            "pressure_comparison": pressure_comparison,
            "message": (
                "Analise concluida com sucesso. "
                f"{len(anomalies)} anomalia(s) identificada(s)."
            ),
        }

    def _evaluate_pressure_anomaly(
        self,
        pressure_comparison: Dict[str, Any],
        pressure_stats: Dict[str, float],
        anomalies: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not pressure_comparison["evaluated"]:
            return pressure_comparison

        press = pressure_comparison["pressure_station_hpa"]
        pressure_comparison = {
            **pressure_comparison,
            "historical_reference": "Percentil 5 a Percentil 95",
            "historical_value": {
                "p5": pressure_stats["p5"],
                "p95": pressure_stats["p95"],
            },
        }

        if press <= pressure_stats["p5"] or press >= pressure_stats["p95"]:
            if press <= pressure_stats["p5"]:
                ref_val = pressure_stats["p5"]
                ref_label = "Percentil 5"
            else:
                ref_val = pressure_stats["p95"]
                ref_label = "Percentil 95"
            anomalies.append({
                "anomaly_type": "PRESSURE_ANOMALY",
                "severity": "MEDIUM",
                "reason": (
                    f"{pressure_comparison['reference_label'].capitalize()} "
                    f"({press:.1f} hPa) fora do padrao historico INMET "
                    f"({ref_label}: {ref_val:.1f} hPa)."
                ),
                "historical_reference": f"{ref_label} INMET nivel da estacao",
                "current_value": press,
                "historical_value": ref_val,
                "variables_used": ["pressure_station_hpa"],
                "reference_used": pressure_comparison["reference_used"],
                "is_estimated": pressure_comparison["is_estimated"],
            })

        return pressure_comparison


def _metadata_from_historical_data(df_historical: pd.DataFrame) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for column in ["altitude_m", "station_altitude", "altitude"]:
        if column in df_historical.columns:
            values = df_historical[column].dropna()
            if not values.empty:
                metadata[column] = values.iloc[0]
                break
    return metadata


def _resolve_altitude_m(station_metadata: Optional[Dict[str, Any]]) -> float | None:
    if not station_metadata:
        return None

    for key in ["altitude_m", "station_altitude", "altitude"]:
        altitude_m = _to_optional_float(station_metadata.get(key))
        if altitude_m is not None:
            return altitude_m

    return None


def _not_evaluated_pressure(message: str) -> Dict[str, Any]:
    return {
        "evaluated": False,
        "pressure_station_hpa": None,
        "reference_used": None,
        "reference_label": None,
        "is_estimated": False,
        "message": message,
    }


def _to_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        text_value = str(value).strip().replace(",", ".")
        if not text_value:
            return None
        try:
            return float(text_value)
        except ValueError:
            return None
