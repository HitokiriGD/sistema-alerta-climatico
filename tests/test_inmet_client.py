from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from src.config.settings import Settings
from src.data.inmet_client import InmetClient
from src.data.inmet_client import InmetHistoricalDataError
from src.data.inmet_client import STANDARD_COLUMNS


def make_settings(zip_dir: Path) -> Settings:
    settings = Mock(spec=Settings)
    settings.inmet_historical_zip_dir = str(zip_dir)
    settings.inmet_historical_start_year = 2020
    settings.inmet_historical_end_year = 2021
    return settings


def write_inmet_zip(zip_dir: Path, year: int, station_code: str = "A001") -> Path:
    zip_path = zip_dir / f"{year}.zip"
    csv_name = (
        f"INMET_CO_DF_{station_code}_BRASILIA_"
        f"01-01-{year}_A_31-12-{year}.CSV"
    )
    csv_text = "\n".join([
        "REGIAO:;CO",
        "UF:;DF",
        "ESTACAO:;BRASILIA",
        f"CODIGO (WMO):;{station_code}",
        "LATITUDE:;-15,78944444",
        "LONGITUDE:;-47,92583332",
        "ALTITUDE:;1160,96",
        "DATA DE FUNDACAO:;07/05/00",
        (
            "Data;Hora UTC;PRECIPITAÇÃO TOTAL, HORÁRIO (mm);"
            "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB);"
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C);"
            "UMIDADE RELATIVA DO AR, HORARIA (%);"
            "VENTO, VELOCIDADE HORARIA (m/s);"
        ),
        f"{year}/01/01;0000 UTC;0;887,7;19,9;92;1,1;",
        f"{year}/01/01;0100 UTC;1,5;888,1;20,5;90;2,5;",
    ])

    with ZipFile(zip_path, "w") as zip_file:
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
        "state": "DF",
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
    assert history.loc[1, "feels_like"] == 0.0


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
    assert history.loc[0, "source"] == "INMET Histórico"
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
