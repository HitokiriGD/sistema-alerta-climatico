import unicodedata
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from src.config.settings import Settings

HISTORICAL_SOURCE_NAME = "INMET Histórico"
METADATA_ROWS = 8
STANDARD_COLUMNS = [
    "station_code",
    "station_name",
    "state",
    "date",
    "hour",
    "datetime",
    "temperature",
    "feels_like",
    "humidity",
    "precipitation",
    "wind_speed",
    "pressure",
    "source",
]
INMET_COLUMN_MAPPING = {
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (degC)": "temperature",
    "UMIDADE RELATIVA DO AR, HORARIA (%)": "humidity",
    "PRECIPITACAO TOTAL, HORARIO (mm)": "precipitation",
    "VENTO, VELOCIDADE HORARIA (m/s)": "wind_speed",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "pressure",
}


class InmetHistoricalDataError(Exception):
    """Erro esperado ao carregar a base historica local do INMET."""


class InmetClient:
    """Cliente local para a base historica oficial do INMET."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.zip_dir = Path(settings.inmet_historical_zip_dir)

    def list_available_zips(self) -> list[Path]:
        """Lista ZIPs anuais disponiveis na pasta configurada."""
        if not self.zip_dir.exists():
            raise InmetHistoricalDataError(
                f"Pasta historica do INMET nao encontrada: {self.zip_dir}"
            )
        if not self.zip_dir.is_dir():
            raise InmetHistoricalDataError(
                f"Caminho historico do INMET nao e uma pasta: {self.zip_dir}"
            )

        zip_paths = sorted(self.zip_dir.glob("*.zip"))
        if not zip_paths:
            raise InmetHistoricalDataError(
                f"Nenhum ZIP historico do INMET encontrado em: {self.zip_dir}"
            )

        return zip_paths

    def filter_zips_by_year(
        self,
        zip_paths: list[Path],
        start_year: int,
        end_year: int,
    ) -> list[Path]:
        """Filtra ZIPs anuais pelo intervalo informado."""
        if start_year > end_year:
            raise InmetHistoricalDataError("Ano inicial nao pode ser maior que o final.")

        filtered_paths = [
            zip_path
            for zip_path in zip_paths
            if self._zip_year(zip_path) is not None
            and start_year <= self._zip_year(zip_path) <= end_year
        ]
        if not filtered_paths:
            raise InmetHistoricalDataError(
                "Nenhum ZIP historico do INMET encontrado no intervalo informado."
            )

        return filtered_paths

    def find_station_csv(self, zip_path: Path, station_code: str) -> str | None:
        """Localiza o CSV da estacao dentro de um ZIP anual."""
        station_code = station_code.strip().upper()
        with ZipFile(zip_path) as zip_file:
            for entry_name in zip_file.namelist():
                name_parts = Path(entry_name).name.upper().split("_")
                if station_code in name_parts and entry_name.upper().endswith(".CSV"):
                    return entry_name

        return None

    def read_station_metadata(
        self,
        zip_path: Path,
        csv_name: str,
    ) -> dict[str, str]:
        """Le os metadados da estacao no inicio do CSV."""
        metadata: dict[str, str] = {}
        with ZipFile(zip_path) as zip_file:
            with zip_file.open(csv_name) as csv_file:
                for index, raw_line in enumerate(csv_file):
                    if index >= METADATA_ROWS:
                        break
                    line = raw_line.decode("latin1").strip()
                    key, _, value = line.partition(";")
                    metadata[key.replace(":", "").strip().upper()] = value.strip()

        return {
            "station_code": metadata.get("CODIGO (WMO)", ""),
            "station_name": metadata.get("ESTACAO", ""),
            "state": metadata.get("UF", ""),
        }

    def read_hourly_data(self, zip_path: Path, csv_name: str) -> pd.DataFrame:
        """Le os dados horarios de um CSV historico do INMET."""
        with ZipFile(zip_path) as zip_file:
            with zip_file.open(csv_name) as csv_file:
                data = pd.read_csv(
                    csv_file,
                    sep=";",
                    encoding="latin1",
                    decimal=",",
                    skiprows=METADATA_ROWS,
                )

        data = data.dropna(axis=1, how="all")
        data.columns = [self._normalize_column_name(column) for column in data.columns]
        return data

    def normalize_historical_data(
        self,
        data: pd.DataFrame,
        metadata: dict[str, str],
    ) -> pd.DataFrame:
        """Normaliza dados historicos do INMET para o padrao do sistema."""
        normalized_data = pd.DataFrame(index=data.index)
        normalized_data["station_code"] = metadata["station_code"]
        normalized_data["station_name"] = metadata["station_name"]
        normalized_data["state"] = metadata["state"]
        normalized_data["date"] = data.get("Data")
        normalized_data["hour"] = data.get("Hora UTC")
        normalized_data["datetime"] = self._build_datetime(
            normalized_data["date"],
            normalized_data["hour"],
        )
        normalized_data["temperature"] = self._numeric_column(data, "temperature")
        normalized_data["feels_like"] = pd.NA
        normalized_data["humidity"] = self._numeric_column(data, "humidity")
        normalized_data["precipitation"] = self._numeric_column(data, "precipitation")
        normalized_data["wind_speed"] = self._numeric_column(data, "wind_speed") * 3.6
        normalized_data["pressure"] = self._numeric_column(data, "pressure")
        normalized_data["source"] = HISTORICAL_SOURCE_NAME

        return normalized_data[STANDARD_COLUMNS]

    def load_station_history(
        self,
        station_code: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """Retorna o historico padronizado da estacao em multiplos anos."""
        start_year = start_year or self.settings.inmet_historical_start_year
        end_year = end_year or self.settings.inmet_historical_end_year
        zip_paths = self.filter_zips_by_year(
            self.list_available_zips(),
            start_year,
            end_year,
        )

        station_frames = []
        for zip_path in zip_paths:
            csv_name = self.find_station_csv(zip_path, station_code)
            if csv_name is None:
                continue

            metadata = self.read_station_metadata(zip_path, csv_name)
            hourly_data = self.read_hourly_data(zip_path, csv_name)
            station_frames.append(
                self.normalize_historical_data(hourly_data, metadata)
            )

        if not station_frames:
            raise InmetHistoricalDataError(
                f"Estacao {station_code.upper()} nao encontrada nos ZIPs informados."
            )

        history = pd.concat(station_frames, ignore_index=True)
        return history.sort_values("datetime").reset_index(drop=True)

    def _zip_year(self, zip_path: Path) -> int | None:
        try:
            return int(zip_path.stem)
        except ValueError:
            return None

    def _normalize_column_name(self, column: Any) -> str:
        column_name = str(column).strip()
        column_name = column_name.replace("\N{DEGREE SIGN}", "deg")
        ascii_name = "".join(
            char
            for char in unicodedata.normalize("NFKD", column_name)
            if not unicodedata.combining(char)
        )
        return INMET_COLUMN_MAPPING.get(ascii_name, column_name)

    def _numeric_column(self, data: pd.DataFrame, column: str) -> pd.Series:
        if column not in data:
            return pd.Series([pd.NA] * len(data), index=data.index)

        values = data[column].astype(str).str.replace(",", ".", regex=False)
        values = values.str.replace("^\\s*$", "", regex=True)
        return pd.to_numeric(values, errors="coerce")

    def _build_datetime(
        self,
        dates: pd.Series,
        hours: pd.Series,
    ) -> pd.Series:
        hour_text = hours.fillna("").astype(str).str.extract(r"(\d{2})", expand=False)
        date_text = dates.fillna("").astype(str).str.replace("/", "-", regex=False)
        return pd.to_datetime(
            date_text + " " + hour_text.fillna("00") + ":00:00",
            errors="coerce",
        )
