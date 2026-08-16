from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pandas as pd
import pytest

from scripts.build_inmet_dataset import build_processed_inmet_dataset
from scripts.build_inmet_station_catalog import build_inmet_station_catalog
from src.config.settings import Settings
from src.data.inmet_client import format_station_catalog_label
from src.data.inmet_client import HISTORICAL_SOURCE_NAME
from src.data.inmet_client import InmetClient
from src.data.inmet_client import InmetHistoricalDataError
from src.data.inmet_client import parse_inmet_station_from_csv_name
from src.data.inmet_client import STANDARD_COLUMNS
from src.data.inmet_client import STATION_CATALOG_COLUMNS
from src.data.inmet_client import normalize_city_name


def make_settings(zip_dir: Path) -> Settings:
    settings = Mock(spec=Settings)
    settings.inmet_historical_zip_dir = str(zip_dir)
    settings.inmet_historical_start_year = 2020
    settings.inmet_historical_end_year = 2026
    settings.inmet_processed_data_path = "data/processed/inmet_hourly.parquet"
    settings.inmet_station_catalog_path = "data/processed/inmet_station_catalog.csv"
    settings.inmet_database_path = "data/processed/inmet_historical.duckdb"
    settings.inmet_database_url = ""
    return settings


def write_inmet_zip(
    zip_dir: Path,
    year: int,
    station_code: str = "A001",
    station_name: str = "BRASILIA",
    state: str = "DF",
    region: str = "CO",
) -> Path:
    zip_path = zip_dir / f"{year}.zip"
    station_name_for_file = station_name.replace(" ", "_")
    csv_name = (
        f"INMET_{region}_{state}_{station_code}_{station_name_for_file}_"
        f"01-01-{year}_A_31-12-{year}.CSV"
    )
    csv_text = "\n".join(
        [
            f"REGIAO:;{region}",
            f"UF:;{state}",
            f"ESTACAO:;{station_name}",
            f"CODIGO (WMO):;{station_code}",
            "LATITUDE:;-15,78944444",
            "LONGITUDE:;-47,92583332",
            "ALTITUDE:;1160,96",
            "DATA DE FUNDACAO:;07/05/00",
            (
                "Data;Hora UTC;PRECIPITACAO TOTAL, HORARIO (mm);"
                "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB);"
                "TEMPERATURA DO AR - BULBO SECO, HORARIA (\N{DEGREE SIGN}C);"
                "UMIDADE RELATIVA DO AR, HORARIA (%);"
                "VENTO, VELOCIDADE HORARIA (m/s);"
            ),
            f"{year}/01/01;0000 UTC;0;887,7;19,9;92;1,1;",
            f"{year}/01/01;0100 UTC;1,5;888,1;20,5;90;2,5;",
        ]
    )

    mode = "a" if zip_path.exists() else "w"
    with ZipFile(zip_path, mode) as zip_file:
        zip_file.writestr(csv_name, csv_text.encode("latin1"))

    return zip_path


def test_list_available_zips_missing_folder(tmp_path):
    client = InmetClient(make_settings(tmp_path / "inexistente"))

    with pytest.raises(InmetHistoricalDataError, match="nao encontrada"):
        client.list_available_zips()


def test_list_available_zips_empty_folder(tmp_path):
    client = InmetClient(make_settings(tmp_path))

    with pytest.raises(InmetHistoricalDataError, match="Nenhum ZIP"):
        client.list_available_zips()


def test_find_station_csv_by_station_code(tmp_path):
    zip_path = write_inmet_zip(tmp_path, 2020, "A001")
    client = InmetClient(make_settings(tmp_path))

    csv_name = client.find_station_csv(zip_path, "A001")

    assert csv_name is not None
    assert "A001" in csv_name


def test_load_station_history_missing_station(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A002")
    client = InmetClient(make_settings(tmp_path))

    with pytest.raises(InmetHistoricalDataError, match="A001"):
        client.load_station_history("A001", 2020, 2020)


def test_read_station_metadata_and_hourly_csv_sample(tmp_path):
    zip_path = write_inmet_zip(tmp_path, 2020, "A001")
    client = InmetClient(make_settings(tmp_path))
    csv_name = client.find_station_csv(zip_path, "A001")

    metadata = client.read_station_metadata(zip_path, csv_name)
    data = client.read_hourly_data(zip_path, csv_name)

    assert metadata == {
        "station_code": "A001",
        "station_name": "BRASILIA",
        "city": "BRASILIA",
        "state": "DF",
        "region": "CO",
        "latitude": "-15,78944444",
        "longitude": "-47,92583332",
        "altitude": "1160,96",
    }
    assert len(data) == 2
    assert "temperature" in data.columns


def test_load_station_history_normalizes_decimal_and_wind_speed(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001")
    client = InmetClient(make_settings(tmp_path))

    history = client.load_station_history("A001", 2020, 2020)

    assert history.loc[0, "temperature"] == 19.9
    assert history.loc[1, "precipitation"] == 1.5
    assert history.loc[1, "wind_speed"] == 9.0
    assert history.loc[1, "pressure"] == 888.1
    assert history.loc[1, "latitude"] == -15.78944444
    assert history.loc[1, "longitude"] == -47.92583332
    assert history.loc[1, "altitude_m"] == 1160.96
    assert pd.isna(history.loc[1, "feels_like"])


def test_load_station_history_combines_multiple_zips(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001")
    write_inmet_zip(tmp_path, 2021, "A001")
    client = InmetClient(make_settings(tmp_path))

    history = client.load_station_history("A001", 2020, 2021)

    assert len(history) == 4
    assert history["date"].min() == "2020/01/01"
    assert history["date"].max() == "2021/01/01"


def test_load_station_history_returns_standard_columns(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001")
    client = InmetClient(make_settings(tmp_path))

    history = client.load_station_history("A001", 2020, 2020)

    assert list(history.columns) == STANDARD_COLUMNS
    assert history.loc[0, "source"] == HISTORICAL_SOURCE_NAME
    assert history.loc[0, "station_code"] == "A001"


def test_filter_zips_by_year_interval(tmp_path):
    write_inmet_zip(tmp_path, 2019, "A001")
    write_inmet_zip(tmp_path, 2020, "A001")
    write_inmet_zip(tmp_path, 2021, "A001")
    client = InmetClient(make_settings(tmp_path))

    filtered_paths = client.filter_zips_by_year(
        client.list_available_zips(),
        2020,
        2020,
    )

    assert [path.name for path in filtered_paths] == ["2020.zip"]


def test_parse_inmet_station_from_manaus_csv_name():
    station = parse_inmet_station_from_csv_name(
        "INMET_N_AM_A101_MANAUS_01-01-2026_A_31-07-2026.CSV"
    )

    assert station == {
        "region": "N",
        "state": "AM",
        "station_code": "A101",
        "station_name": "MANAUS",
        "city": "MANAUS",
    }


def test_parse_inmet_station_from_brasilia_csv_name():
    station = parse_inmet_station_from_csv_name(
        "INMET_CO_DF_A001_BRASILIA_01-01-2026_A_31-07-2026.CSV"
    )

    assert station == {
        "region": "CO",
        "state": "DF",
        "station_code": "A001",
        "station_name": "BRASILIA",
        "city": "BRASILIA",
    }


def test_normalize_city_name_handles_accent_and_case():
    assert normalize_city_name("  Bras\u00edlia  ") == "BRASILIA"
    assert normalize_city_name("BRASILIA") == "BRASILIA"
    assert normalize_city_name("brasilia") == "BRASILIA"
    assert normalize_city_name("Manaus") == "MANAUS"


def test_build_station_catalog_from_zip_names_and_metadata(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001", "BRASILIA", "DF", "CO")
    client = InmetClient(make_settings(tmp_path))

    catalog = client.build_station_catalog()

    assert list(catalog.columns) == STATION_CATALOG_COLUMNS
    assert catalog.loc[0, "station_code"] == "A001"
    assert catalog.loc[0, "city"] == "BRASILIA"
    assert catalog.loc[0, "station_name_normalized"] == "BRASILIA"
    assert catalog.loc[0, "city_normalized"] == "BRASILIA"
    assert catalog.loc[0, "state"] == "DF"
    assert catalog.loc[0, "region"] == "CO"
    assert catalog.loc[0, "latitude"] == -15.78944444
    assert catalog.loc[0, "altitude_m"] == 1160.96
    assert catalog.loc[0, "station_label"] == "BRASILIA - DF | A001"


def test_find_station_by_city_state_returns_brasilia_station(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001", "BRASILIA", "DF", "CO")
    client = InmetClient(make_settings(tmp_path))

    station = client.find_station_by_city_state("Bras\u00edlia", "df")

    assert station["station_code"] == "A001"
    assert station["city_normalized"] == "BRASILIA"


def test_find_stations_by_city_state_missing_city_raises_friendly_error(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001", "BRASILIA", "DF", "CO")
    client = InmetClient(make_settings(tmp_path))

    with pytest.raises(InmetHistoricalDataError, match="Nenhuma estacao"):
        client.find_stations_by_city_state("Cidade Inexistente", "DF")


def test_find_stations_by_city_state_returns_multiple_options(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001", "BRASILIA", "DF", "CO")
    write_inmet_zip(tmp_path, 2020, "A002", "BRASILIA", "DF", "CO")
    client = InmetClient(make_settings(tmp_path))

    matches = client.find_stations_by_city_state("brasilia", "DF")

    assert list(matches["station_code"]) == ["A001", "A002"]
    with pytest.raises(InmetHistoricalDataError, match="Mais de uma estacao"):
        client.find_station_by_city_state("brasilia", "DF")


def test_station_catalog_contains_searchable_option_label(tmp_path):
    write_inmet_zip(tmp_path, 2026, "A101", "MANAUS", "AM", "N")
    client = InmetClient(make_settings(tmp_path))

    catalog = client.build_station_catalog()

    assert catalog.loc[0, "station_label"] == "MANAUS - AM | A101"
    assert format_station_catalog_label(catalog.loc[0]) == "MANAUS - AM | A101"


def test_find_station_by_label_returns_station_code(tmp_path):
    write_inmet_zip(tmp_path, 2026, "A101", "MANAUS", "AM", "N")
    client = InmetClient(make_settings(tmp_path))

    station = client.find_station_by_label("MANAUS - AM | A101")

    assert station["station_code"] == "A101"
    assert station["state"] == "AM"


def test_list_station_options_keeps_duplicate_city_labels_distinguishable(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001", "BRASILIA", "DF", "CO")
    write_inmet_zip(tmp_path, 2020, "A099", "BRASILIA", "DF", "CO")
    client = InmetClient(make_settings(tmp_path))

    labels = [
        str(station["station_label"])
        for station in client.list_station_options()
    ]

    assert labels == ["BRASILIA - DF | A001", "BRASILIA - DF | A099"]
    assert len(set(labels)) == 2


def test_build_station_catalog_updates_available_year_range(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001", "BRASILIA", "DF", "CO")
    write_inmet_zip(tmp_path, 2022, "A001", "BRASILIA", "DF", "CO")
    client = InmetClient(make_settings(tmp_path))

    catalog = client.build_station_catalog()

    assert catalog.loc[0, "first_available_year"] == 2020
    assert catalog.loc[0, "last_available_year"] == 2022


def test_build_inmet_station_catalog_script_with_simulated_zips(tmp_path):
    write_inmet_zip(tmp_path, 2020, "A001", "BRASILIA", "DF", "CO")
    output_path = tmp_path / "inmet_station_catalog.csv"

    saved_path = build_inmet_station_catalog(
        output_path=output_path,
        settings=make_settings(tmp_path),
    )

    saved_catalog = pd.read_csv(saved_path)
    assert saved_path == output_path
    assert saved_catalog.loc[0, "station_code"] == "A001"
    assert saved_catalog.loc[0, "city_normalized"] == "BRASILIA"


def test_build_processed_inmet_dataset_with_simulated_client(tmp_path, monkeypatch):
    class FakeInmetClient:
        def __init__(self, settings):
            self.settings = settings

        def load_station_history(self, station_code, start_year=None, end_year=None):
            assert station_code == "A001"
            assert start_year == 2020
            assert end_year == 2020
            return pd.DataFrame(
                [
                    {
                        "station_code": "A001",
                        "datetime": "2020-01-01 00:00:00",
                        "temperature": "20,5",
                        "feels_like": None,
                        "humidity": "80",
                        "precipitation": "0",
                        "wind_speed": "3,6",
                        "pressure": "1009,5",
                        "source": "INMET Historico",
                    }
                ]
            )

    monkeypatch.setattr("scripts.build_inmet_dataset.InmetClient", FakeInmetClient)
    output_path = tmp_path / "inmet_hourly.csv"

    saved_path = build_processed_inmet_dataset(
        station_code="A001",
        start_year=2020,
        end_year=2020,
        output_path=output_path,
        settings=make_settings(tmp_path),
    )

    saved_data = pd.read_csv(saved_path)
    assert saved_path == output_path
    assert saved_data.loc[0, "temperature"] == 20.5
    assert saved_data.loc[0, "quality_flag"] == "incomplete"
