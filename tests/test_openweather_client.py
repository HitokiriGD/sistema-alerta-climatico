from dataclasses import replace

from src.config.settings import load_settings
from src.data.openweather_client import OpenWeatherClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"coord": {"lat": -9.65, "lon": -35.73}}


def test_current_weather_by_coordinates_does_not_use_city_geocoding(monkeypatch) -> None:
    captured = {}

    def fake_get(url, params, timeout):
        captured.update({"url": url, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("src.data.openweather_client.requests.get", fake_get)
    settings = replace(
        load_settings(load_dotenv_file=False),
        openweather_api_key="test-key",
    )

    payload = OpenWeatherClient(settings).fetch_current_weather_by_coordinates(
        -9.65,
        -35.73,
    )

    assert payload["coord"] == {"lat": -9.65, "lon": -35.73}
    assert captured["params"]["lat"] == -9.65
    assert captured["params"]["lon"] == -35.73
    assert captured["params"]["lang"] == "pt_br"
    assert "q" not in captured["params"]
    assert "country" not in captured["params"]
