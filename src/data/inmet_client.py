import re
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
    "latitude",
    "longitude",
    "altitude_m",
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
STATION_CATALOG_COLUMNS = [
    "station_code",
    "station_name",
    "station_name_normalized",
    "city",
    "city_normalized",
    "state",
    "region",
    "latitude",
    "longitude",
    "altitude",
    "altitude_m",
    "first_available_year",
    "last_available_year",
    "station_label",
]
INMET_CSV_NAME_PATTERN = re.compile(
    r"^INMET_(?P<region>[^_]+)_(?P<state>[A-Z]{2})_"
    r"(?P<station_code>[A-Z]\d{3})_(?P<station_name>.+?)_"
    r"\d{2}-\d{2}-\d{4}_A_\d{2}-\d{2}-\d{4}\.CSV$",
    re.IGNORECASE,
)
INMET_COLUMN_MAPPING = {
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (degC)": "temperature",
    "UMIDADE RELATIVA DO AR, HORARIA (%)": "humidity",
    "PRECIPITACAO TOTAL, HORARIO (mm)": "precipitation",
    "VENTO, VELOCIDADE HORARIA (m/s)": "wind_speed",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "pressure",
}


class InmetHistoricalDataError(Exception):
    """Erro esperado ao carregar a base historica local do INMET."""


def normalize_city_name(city: str) -> str:
    """Normaliza nome de cidade para busca simples no catalogo INMET."""
    ascii_city = "".join(
        char
        for char in unicodedata.normalize("NFKD", city.strip())
        if not unicodedata.combining(char)
    )
    return " ".join(ascii_city.upper().split())


def parse_inmet_station_from_csv_name(csv_name: str) -> dict[str, str] | None:
    """Extrai identificacao da estacao pelo nome do CSV historico do INMET."""
    file_name = Path(csv_name).name
    match = INMET_CSV_NAME_PATTERN.match(file_name)
    if match is None:
        return None

    station_name = match.group("station_name").replace("_", " ").strip().upper()
    return {
        "region": match.group("region").strip().upper(),
        "state": match.group("state").strip().upper(),
        "station_code": match.group("station_code").strip().upper(),
        "station_name": station_name,
        "city": station_name,
    }


def format_station_catalog_label(station: dict[str, Any] | pd.Series) -> str:
    """Formata label pesquisavel para selecao de estacao INMET."""
    return (
        f"{station['station_name']} - {station['state']} | "
        f"{station['station_code']}"
    )


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
            "city": metadata.get("ESTACAO", ""),
            "state": metadata.get("UF", ""),
            "region": metadata.get("REGIAO", ""),
            "latitude": metadata.get("LATITUDE", ""),
            "longitude": metadata.get("LONGITUDE", ""),
            "altitude": metadata.get("ALTITUDE", ""),
        }

    def build_station_catalog(self) -> pd.DataFrame:
        """Constroi catalogo local de estacoes lendo metadados dos ZIPs."""
        stations: dict[str, dict[str, Any]] = {}
        zip_paths = self.filter_zips_by_year(
            self.list_available_zips(),
            self.settings.inmet_historical_start_year,
            self.settings.inmet_historical_end_year,
        )

        for zip_path in zip_paths:
            year = self._zip_year(zip_path)
            if year is None:
                continue

            with ZipFile(zip_path) as zip_file:
                csv_names = [
                    entry_name
                    for entry_name in zip_file.namelist()
                    if entry_name.upper().endswith(".CSV")
                ]

            for csv_name in csv_names:
                parsed_station = parse_inmet_station_from_csv_name(csv_name)
                if parsed_station is None:
                    continue

                station_code = parsed_station["station_code"]
                if not station_code:
                    continue

                station = stations.get(station_code)
                if station is None:
                    metadata = self.read_station_metadata(zip_path, csv_name)
                    station_name = parsed_station["station_name"]
                    city = parsed_station["city"]
                    station = {
                        "station_code": station_code,
                        "station_name": station_name,
                        "station_name_normalized": normalize_city_name(station_name),
                        "city": city,
                        "city_normalized": normalize_city_name(city),
                        "state": parsed_station["state"],
                        "region": parsed_station["region"],
                        "latitude": self._decimal_text_to_float(metadata["latitude"]),
                        "longitude": self._decimal_text_to_float(metadata["longitude"]),
                        "altitude": self._decimal_text_to_float(metadata["altitude"]),
                        "altitude_m": self._decimal_text_to_float(metadata["altitude"]),
                        "first_available_year": year,
                        "last_available_year": year,
                        "station_label": "",
                    }
                    station["station_label"] = format_station_catalog_label(station)
                    stations[station_code] = station
                else:
                    station["first_available_year"] = min(
                        station["first_available_year"],
                        year,
                    )
                    station["last_available_year"] = max(
                        station["last_available_year"],
                        year,
                    )

        catalog = pd.DataFrame(stations.values(), columns=STATION_CATALOG_COLUMNS)
        if catalog.empty:
            raise InmetHistoricalDataError(
                "Nenhuma estacao INMET encontrada nos ZIPs historicos."
            )

        return catalog.sort_values(["state", "city", "station_code"]).reset_index(
            drop=True
        )

    def list_station_options(self) -> list[dict[str, Any]]:
        """Lista estacoes disponiveis para selecao no dashboard."""
        catalog = self.build_station_catalog()
        return [station.to_dict() for _, station in catalog.iterrows()]

    def find_station_by_label(
        self,
        station_label: str,
        catalog: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Recupera uma estacao do catalogo pelo label exibido ao usuario."""
        if catalog is None:
            station_catalog = self.build_station_catalog()
        else:
            station_catalog = catalog

        matches = station_catalog[
            station_catalog["station_label"] == station_label
        ].reset_index(drop=True)
        if matches.empty:
            raise InmetHistoricalDataError(
                f"Estacao INMET nao encontrada para a opcao: {station_label}."
            )

        return matches.iloc[0].to_dict()

    def find_stations_by_city_state(
        self,
        city: str,
        state: str,
        catalog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Busca estacoes do catalogo por cidade e UF."""
        if not city.strip() or not state.strip():
            raise InmetHistoricalDataError("Informe cidade e UF para buscar a estacao.")

        if catalog is None:
            station_catalog = self.build_station_catalog()
        else:
            station_catalog = catalog
        city_normalized = normalize_city_name(city)
        state_normalized = state.strip().upper()

        matches = station_catalog[
            (station_catalog["city_normalized"] == city_normalized)
            & (station_catalog["state"].str.upper() == state_normalized)
        ].reset_index(drop=True)

        if matches.empty:
            raise InmetHistoricalDataError(
                "Nenhuma estacao INMET encontrada para "
                f"{city.strip()}/{state_normalized}."
            )

        return matches

    def find_station_by_city_state(
        self,
        city: str,
        state: str,
        catalog: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Retorna uma unica estacao por cidade/UF quando nao ha ambiguidade."""
        matches = self.find_stations_by_city_state(city, state, catalog)
        if len(matches) > 1:
            raise InmetHistoricalDataError(
                f"Mais de uma estacao encontrada para {city.strip()}/{state.upper()}."
            )

        return matches.iloc[0].to_dict()

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
        normalized_data["latitude"] = self._decimal_text_to_float(
            metadata.get("latitude", "")
        )
        normalized_data["longitude"] = self._decimal_text_to_float(
            metadata.get("longitude", "")
        )
        normalized_data["altitude_m"] = self._decimal_text_to_float(
            metadata.get("altitude", "")
        )
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

    def _decimal_text_to_float(self, value: str) -> float | None:
        text = str(value).strip().replace(",", ".")
        numeric_value = pd.to_numeric(text, errors="coerce")
        if pd.isna(numeric_value):
            return None
        return float(numeric_value)

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
