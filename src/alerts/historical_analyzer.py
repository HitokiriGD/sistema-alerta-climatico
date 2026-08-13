from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class HistoricalAnalyzer:
    """Analisador estatístico para comparar dados climáticos atuais com a base histórica do INMET."""

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
            date_col = next((c for c in ["datetime", "date", "data", "DT_MEDICAO"] if c in df.columns), None)
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
        self, current_data: Dict[str, Any], df_historical: pd.DataFrame, current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        if df_historical is None or df_historical.empty or not current_data:
            return {
                "has_historical_data": False,
                "anomalies": [],
                "statistics": {},
                "message": "Histórico ausente ou dados atuais incompletos.",
            }

        target_month = current_date.month if current_date else datetime.now().month
        stats = self.calculate_statistics(df_historical, month=target_month)

        if not stats:
            return {
                "has_historical_data": False,
                "anomalies": [],
                "statistics": {},
                "message": "Dados históricos insuficientes para cálculo de estatísticas.",
            }

        anomalies: List[Dict[str, Any]] = []

        temp = current_data.get("temperature", current_data.get("temp_c"))
        hum = current_data.get("humidity")
        precip = current_data.get("precipitation", current_data.get("rain_mm", 0.0))
        wind = current_data.get("wind_speed", current_data.get("wind_ms"))
        press = current_data.get("pressure", current_data.get("pressure_hpa"))

        # Temperatura
        if temp is not None and "temperature" in stats:
            s_temp = stats["temperature"]
            if temp >= s_temp["p95"]:
                anomalies.append({
                    "anomaly_type": "HIGH_TEMPERATURE",
                    "severity": "CRITICAL" if temp > s_temp["max"] else "HIGH",
                    "reason": f"Temperatura atual ({temp:.1f}°C) acima do percentil 95 histórico ({s_temp['p95']:.1f}°C).",
                    "historical_reference": "Percentil 95",
                    "current_value": temp,
                    "historical_value": s_temp["p95"],
                    "variables_used": ["temperature"],
                })

        # Umidade
        if hum is not None and "humidity" in stats:
            s_hum = stats["humidity"]
            if hum <= s_hum["p10"]:
                anomalies.append({
                    "anomaly_type": "LOW_HUMIDITY",
                    "severity": "CRITICAL" if hum < s_hum["min"] else "HIGH",
                    "reason": f"Umidade atual ({hum:.1f}%) abaixo do percentil 10 histórico ({s_hum['p10']:.1f}%).",
                    "historical_reference": "Percentil 10",
                    "current_value": hum,
                    "historical_value": s_hum["p10"],
                    "variables_used": ["humidity"],
                })

        # Chuva
        if precip is not None and precip > 0 and "precipitation" in stats:
            s_precip = stats["precipitation"]
            if precip >= s_precip["p95"]:
                anomalies.append({
                    "anomaly_type": "HIGH_PRECIPITATION",
                    "severity": "HIGH",
                    "reason": f"Precipitação atual ({precip:.1f}mm) acima do percentil 95 histórico ({s_precip['p95']:.1f}mm).",
                    "historical_reference": "Percentil 95",
                    "current_value": precip,
                    "historical_value": s_precip["p95"],
                    "variables_used": ["precipitation"],
                })

        # Vento
        if wind is not None and "wind_speed" in stats:
            s_wind = stats["wind_speed"]
            if wind >= s_wind["p95"]:
                anomalies.append({
                    "anomaly_type": "HIGH_WIND",
                    "severity": "HIGH",
                    "reason": f"Velocidade do vento ({wind:.1f}m/s) acima do percentil 95 histórico ({s_wind['p95']:.1f}m/s).",
                    "historical_reference": "Percentil 95",
                    "current_value": wind,
                    "historical_value": s_wind["p95"],
                    "variables_used": ["wind_speed"],
                })

        # Pressão
        if press is not None and "pressure" in stats:
            s_press = stats["pressure"]
            if press <= s_press["p5"] or press >= s_press["p95"]:
                ref_val = s_press["p5"] if press <= s_press["p5"] else s_press["p95"]
                ref_label = "Percentil 5" if press <= s_press["p5"] else "Percentil 95"
                anomalies.append({
                    "anomaly_type": "PRESSURE_ANOMALY",
                    "severity": "MEDIUM",
                    "reason": f"Pressão atmosférica ({press:.1f} hPa) fora do padrão histórico ({ref_label}: {ref_val:.1f} hPa).",
                    "historical_reference": ref_label,
                    "current_value": press,
                    "historical_value": ref_val,
                    "variables_used": ["pressure"],
                })

        # Risco de Incêndio Contextual
        if temp is not None and hum is not None and "humidity" in stats and "temperature" in stats:
            if hum <= stats["humidity"]["p25"] and temp >= stats["temperature"]["p75"]:
                is_windy = wind is not None and "wind_speed" in stats and wind >= stats["wind_speed"]["median"]
                anomalies.append({
                    "anomaly_type": "POTENTIAL_FIRE_RISK",
                    "severity": "CRITICAL" if is_windy else "HIGH",
                    "reason": "Condição favorável a risco potencial de incêndio (baixa umidade, alta temperatura e vento acima do padrão).",
                    "historical_reference": "Combinação P25 Umidade + P75 Temperatura",
                    "current_value": {"temp": temp, "humidity": hum, "wind": wind},
                    "historical_value": {"p25_hum": stats["humidity"]["p25"], "p75_temp": stats["temperature"]["p75"]},
                    "variables_used": ["temperature", "humidity", "wind_speed"],
                })

        return {
            "has_historical_data": True,
            "anomalies": anomalies,
            "statistics": stats,
            "message": f"Análise concluída com sucesso. {len(anomalies)} anomalia(s) identificada(s).",
        }