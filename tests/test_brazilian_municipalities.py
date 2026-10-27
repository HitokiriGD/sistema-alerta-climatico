from pathlib import Path
import unicodedata

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.data.brazilian_municipalities import DEFAULT_MUNICIPALITY_CATALOG_PATH
from src.data.brazilian_municipalities import MunicipalityCatalogError
from src.data.brazilian_municipalities import MunicipalityLocationError
from src.data.brazilian_municipalities import VALID_STATE_CODES
from src.data.brazilian_municipalities import filter_municipalities
from src.data.brazilian_municipalities import find_municipality_by_name
from src.data.brazilian_municipalities import format_municipality_option
from src.data.brazilian_municipalities import load_local_municipality_catalog
from src.data.brazilian_municipalities import validate_municipality_coordinates


EXPECTED_MUNICIPALITY_COUNT = 5_571


@pytest.fixture(scope="module")
def municipalities():
    return load_local_municipality_catalog()


def test_catalog_loads_locally_without_internet(monkeypatch) -> None:
    def fail_request(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", fail_request)

    municipalities = load_local_municipality_catalog()

    assert len(municipalities) == EXPECTED_MUNICIPALITY_COUNT


def test_catalog_has_expected_count_and_unique_real_codes(municipalities) -> None:
    codes = [municipality.ibge_code for municipality in municipalities]

    assert len(municipalities) == EXPECTED_MUNICIPALITY_COUNT
    assert len(codes) == len(set(codes))
    assert all(code > 0 and len(str(code)) == 7 for code in codes)


def test_catalog_contains_only_valid_states(municipalities) -> None:
    state_codes = {municipality.state_code for municipality in municipalities}

    assert state_codes == VALID_STATE_CODES


def test_catalog_coordinates_are_valid(municipalities) -> None:
    for municipality in municipalities:
        assert validate_municipality_coordinates(
            municipality.latitude,
            municipality.longitude,
        ) == (municipality.latitude, municipality.longitude)


def test_search_and_option_format_accept_text_without_accents(
    municipalities,
) -> None:
    maceio = find_municipality_by_name(municipalities, "Maceió", "AL")
    brasilia = find_municipality_by_name(municipalities, "Brasília", "DF")

    assert maceio is not None
    assert brasilia is not None
    assert filter_municipalities(municipalities, "maceio/al") == (maceio,)
    assert filter_municipalities(municipalities, "BRASILIA/DF") == (brasilia,)
    option_label = format_municipality_option(maceio)
    assert unicodedata.normalize("NFC", option_label) == "Maceió/AL"
    assert "maceio" in option_label.casefold()


def test_catalog_keeps_homonymous_municipalities_separated_by_state(
    municipalities,
) -> None:
    homonyms = [item for item in municipalities if item.name == "Bom Jesus"]

    assert len(homonyms) > 1
    assert len({item.state_code for item in homonyms}) == len(homonyms)
    assert find_municipality_by_name(municipalities, "Bom Jesus") is None
    assert find_municipality_by_name(municipalities, "Bom Jesus", "PI") is not None


def test_missing_catalog_has_clear_configuration_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(MunicipalityCatalogError, match="não foi encontrado"):
        load_local_municipality_catalog(missing_path)


def test_invalid_catalog_has_clear_configuration_error(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.csv"
    invalid_path.write_text(
        "ibge_code,name,uf,latitude,longitude\n"
        "0,Município inválido,XX,90,180\n",
        encoding="utf-8",
    )

    with pytest.raises(MunicipalityCatalogError, match="código IBGE inválido"):
        load_local_municipality_catalog(invalid_path)


def test_invalid_coordinates_are_rejected() -> None:
    with pytest.raises(MunicipalityLocationError, match="coordenadas locais"):
        validate_municipality_coordinates(90, 180)


def test_versioned_catalog_path_is_project_relative() -> None:
    assert DEFAULT_MUNICIPALITY_CATALOG_PATH.is_file()
    assert DEFAULT_MUNICIPALITY_CATALOG_PATH.name == "brazilian_municipalities.csv"
    assert "data" in DEFAULT_MUNICIPALITY_CATALOG_PATH.parts
    assert "reference" in DEFAULT_MUNICIPALITY_CATALOG_PATH.parts


def test_form_initializes_when_ibge_api_is_unavailable(monkeypatch) -> None:
    def fail_ibge_request(*args, **kwargs):
        raise requests.ConnectionError("IBGE offline")

    monkeypatch.setattr(requests, "get", fail_ibge_request)
    app = AppTest.from_string(
        """
import streamlit as st
from app.streamlit_app import build_query_form
from src.config.settings import load_settings

query, submitted = build_query_form(load_settings(load_dotenv_file=False))
st.write(query.get("ibge_code"))
"""
    ).run(timeout=20)

    assert not app.exception
    assert app.selectbox[0].label == "Município/UF"
    assert len(app.selectbox[0].options) == EXPECTED_MUNICIPALITY_COUNT
