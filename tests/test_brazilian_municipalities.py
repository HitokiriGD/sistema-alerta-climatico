import unicodedata

import pytest
import requests

from src.data.brazilian_municipalities import BrazilianMunicipality
from src.data.brazilian_municipalities import MunicipalityCatalogError
from src.data.brazilian_municipalities import MunicipalityLocationError
from src.data.brazilian_municipalities import fetch_ibge_municipalities
from src.data.brazilian_municipalities import fetch_ibge_municipality_coordinates
from src.data.brazilian_municipalities import filter_municipalities
from src.data.brazilian_municipalities import find_municipality_by_name
from src.data.brazilian_municipalities import format_municipality_option


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        return self.payload


def make_ibge_item(code, name, state_code, state_name):
    return {
        "id": code,
        "nome": name,
        "regiao-imediata": {
            "regiao-intermediaria": {
                "UF": {"sigla": state_code, "nome": state_name}
            }
        },
    }


def test_catalog_keeps_homonymous_municipalities_separated_by_state() -> None:
    payload = [
        make_ibge_item(index + 10_000, f"Município {index}", "SP", "São Paulo")
        for index in range(5_000)
    ]
    payload[0] = make_ibge_item(1, "Bom Jesus", "PI", "Piauí")
    payload[1] = make_ibge_item(2, "Bom Jesus", "RS", "Rio Grande do Sul")

    municipalities = fetch_ibge_municipalities(
        request_get=lambda *args, **kwargs: FakeResponse(payload)
    )

    homonyms = [item for item in municipalities if item.name == "Bom Jesus"]
    assert {item.label for item in homonyms} == {"Bom Jesus/PI", "Bom Jesus/RS"}
    assert find_municipality_by_name(municipalities, "Bom Jesus") is None
    assert find_municipality_by_name(municipalities, "Bom Jesus", "PI").ibge_code == 1


def test_search_and_option_format_accept_text_without_accents() -> None:
    municipalities = (
        BrazilianMunicipality(2704302, "Maceió", "AL", "Alagoas"),
        BrazilianMunicipality(5300108, "Brasília", "DF", "Distrito Federal"),
    )

    assert filter_municipalities(municipalities, "maceio") == (municipalities[0],)
    assert filter_municipalities(municipalities, "BRASILIA/DF") == (
        municipalities[1],
    )
    option_label = format_municipality_option(municipalities[0])
    assert unicodedata.normalize("NFC", option_label) == "Maceió/AL"
    assert "maceio" in option_label.casefold()


def test_catalog_failure_is_reported_with_controlled_error() -> None:
    def fail_request(*args, **kwargs):
        raise requests.ConnectionError("offline")

    with pytest.raises(MunicipalityCatalogError, match="catálogo"):
        fetch_ibge_municipalities(request_get=fail_request)


def test_coordinates_use_point_inside_ibge_municipality_mesh() -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"codarea": "2704302"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-36.0, -10.0],
                            [-35.0, -10.0],
                            [-35.0, -9.0],
                            [-36.0, -9.0],
                            [-36.0, -10.0],
                        ]
                    ],
                },
            }
        ],
    }

    latitude, longitude = fetch_ibge_municipality_coordinates(
        2704302,
        request_get=lambda *args, **kwargs: FakeResponse(payload),
    )

    assert latitude == pytest.approx(-9.5)
    assert longitude == pytest.approx(-35.5)


def test_coordinates_reject_unvalidated_fallback_option() -> None:
    with pytest.raises(MunicipalityLocationError, match="código IBGE"):
        fetch_ibge_municipality_coordinates(0)
