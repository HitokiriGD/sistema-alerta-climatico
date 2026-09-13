from dataclasses import replace
from pathlib import Path

from app import streamlit_app
from app.streamlit_app import fetch_openweather_weather_data
from app.streamlit_app import persist_query_snapshot
from src.config.settings import load_settings


DATABASE_URL = "postgresql://dashboard:private-value@example.test/weather"


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeRepository:
    def __init__(self, observation_id: int | None = 42):
        self.observation_id = observation_id
        self.connection_calls = []
        self.observations = []

    def connect_database(self, database_url):
        self.connection_calls.append(database_url)
        return FakeConnection()

    def insert_weather_observation(self, connection, observation):
        self.observations.append(observation)
        return self.observation_id


class FailingRepository(FakeRepository):
    def connect_database(self, database_url):
        raise RuntimeError(f"connection failed for {database_url}")


class FakeOpenWeatherClient:
    def __init__(self, settings):
        self.settings = settings

    def fetch_current_weather(self, city):
        return make_payload(city)


def make_settings(**overrides):
    settings = load_settings(load_dotenv_file=False)
    values = {
        "database_url": DATABASE_URL,
        "openweather_api_key": "test-api-key",
        "openweather_store_raw_payload": False,
    }
    values.update(overrides)
    return replace(settings, **values)


def make_payload(city: str = "Manaus") -> dict[str, object]:
    return {
        "name": city,
        "dt": 1_700_000_000,
        "coord": {"lat": -3.10, "lon": -60.02},
        "main": {
            "temp": 34.0,
            "feels_like": 38.0,
            "humidity": 48,
            "pressure": 1008,
            "grnd_level": 1005,
        },
        "rain": {"1h": 0.0},
        "wind": {"speed": 4.0},
        "clouds": {"all": 30},
        "weather": [{"description": "nuvens dispersas"}],
    }


def test_workflow_has_manual_dispatch_without_automatic_schedule() -> None:
    workflow = Path(".github/workflows/collect-openweather.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "cron:" not in workflow


def test_empty_database_url_does_not_try_to_save() -> None:
    repository = FakeRepository()

    result = persist_query_snapshot(
        make_settings(database_url=""),
        make_payload(),
        "Manaus",
        "BR",
        repository=repository,
    )

    assert result == {
        "saved": False,
        "enabled": False,
        "observation_id": None,
        "message": "Persistência operacional não configurada.",
        "error_message": None,
    }
    assert repository.connection_calls == []
    assert repository.observations == []


def test_successful_insert_returns_id_and_respects_raw_payload_setting() -> None:
    repository = FakeRepository(observation_id=73)
    payload = make_payload()

    result = persist_query_snapshot(
        make_settings(openweather_store_raw_payload=False),
        payload,
        "Manaus",
        "BR",
        repository=repository,
    )

    assert result["enabled"] is True
    assert result["saved"] is True
    assert result["observation_id"] == 73
    assert result["message"] == "Consulta salva na base operacional."
    assert result["error_message"] is None
    assert repository.observations[0]["city"] == "Manaus"
    assert repository.observations[0]["raw_payload"] is None


def test_raw_payload_is_saved_only_when_enabled() -> None:
    repository = FakeRepository()
    payload = make_payload()

    persist_query_snapshot(
        make_settings(openweather_store_raw_payload=True),
        payload,
        "Manaus",
        "BR",
        repository=repository,
    )

    assert repository.observations[0]["raw_payload"] == payload


def test_database_failure_is_controlled_and_does_not_expose_url() -> None:
    result = persist_query_snapshot(
        make_settings(),
        make_payload(),
        "Manaus",
        "BR",
        repository=FailingRepository(),
    )

    combined = " ".join(str(value) for value in result.values())
    assert result["enabled"] is True
    assert result["saved"] is False
    assert result["observation_id"] is None
    assert "não foi possível salvar" in result["message"]
    assert result["error_message"]
    assert DATABASE_URL not in combined
    assert "private-value" not in combined


def test_dashboard_fetch_continues_without_supabase(monkeypatch) -> None:
    monkeypatch.setattr(streamlit_app, "OpenWeatherClient", FakeOpenWeatherClient)
    settings = make_settings(database_url="")

    weather_data = fetch_openweather_weather_data(settings, "Manaus", "BR")

    assert weather_data is not None
    assert weather_data["city"] == "Manaus"
    assert weather_data["data_source"] == "OpenWeather"


def test_dashboard_fetch_continues_when_database_insert_fails(monkeypatch) -> None:
    statuses = []

    def fail_connection(database_url):
        raise RuntimeError(f"database unavailable: {database_url}")

    monkeypatch.setattr(streamlit_app, "OpenWeatherClient", FakeOpenWeatherClient)
    monkeypatch.setattr(
        streamlit_app.weather_repository,
        "connect_database",
        fail_connection,
    )
    monkeypatch.setattr(
        streamlit_app,
        "show_query_persistence_status",
        statuses.append,
    )

    weather_data = fetch_openweather_weather_data(
        make_settings(),
        "Manaus",
        "BR",
    )

    assert weather_data is not None
    assert weather_data["city"] == "Manaus"
    assert statuses[0]["enabled"] is True
    assert statuses[0]["saved"] is False
    assert DATABASE_URL not in str(statuses[0])


def test_successful_openweather_fetch_triggers_query_persistence(monkeypatch) -> None:
    persistence_calls = []

    def fake_persist(settings, payload, requested_city, country):
        persistence_calls.append((payload, requested_city, country))
        return {
            "saved": True,
            "enabled": True,
            "observation_id": 10,
            "message": "Consulta salva na base operacional.",
            "error_message": None,
        }

    monkeypatch.setattr(streamlit_app, "OpenWeatherClient", FakeOpenWeatherClient)
    monkeypatch.setattr(streamlit_app, "persist_query_snapshot", fake_persist)
    monkeypatch.setattr(
        streamlit_app,
        "show_query_persistence_status",
        lambda status: None,
    )

    weather_data = fetch_openweather_weather_data(
        make_settings(),
        "Manaus",
        "BR",
    )

    assert weather_data is not None
    assert len(persistence_calls) == 1
    assert persistence_calls[0][1:] == ("Manaus", "BR")
