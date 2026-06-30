import pytest
from unittest.mock import patch, Mock
from src.data.inmet_client import InmetClient
from src.config.settings import Settings

@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    settings.inmet_base_url = "https://apitempo.inmet.gov.br"
    return settings

@pytest.fixture
def inmet_client(mock_settings):
    return InmetClient(settings=mock_settings)

@patch("src.data.inmet_client.requests.get")
def test_get_current_weather_success(mock_get, inmet_client):
    mock_response = Mock()
    mock_response.json.return_value = [{
        "TEM_INS": "25.5",
        "UMD_INS": "60",
        "CHUVA": "10.2",
        "VEN_VEL": "3.5",
        "PRE_INS": "1012.1"
    }]
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    data = inmet_client.get_current_weather("A001")

    assert data is not None
    assert data["temperature"] == 25.5
    assert data["feels_like"] == 0.0
    assert data["humidity"] == 60.0
    assert data["precipitation"] == 10.2
    assert data["wind_speed"] == 3.5
    assert data["pressure"] == 1012.1

@patch("src.data.inmet_client.requests.get")
def test_get_current_weather_empty(mock_get, inmet_client):
    mock_response = Mock()
    mock_response.json.return_value = []
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    data = inmet_client.get_current_weather("A001")
    assert data is None