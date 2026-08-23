from dataclasses import replace

import requests

from scripts.collect_openweather_snapshot import build_observation_from_payload
from scripts.collect_openweather_snapshot import collect_openweather_snapshots
from scripts.collect_openweather_snapshot import resolve_collection_cities
from scripts.init_weather_database import initialize_weather_database
from src.config.settings import load_settings
from src.data.weather_repository import WeatherRepositoryError


DATABASE_URL = "postgresql://user:secret@example/db"


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeRepository:
    WeatherRepositoryError = WeatherRepositoryError

    def __init__(self):
        self.connection = FakeConnection()
        self.tables_created = False
        self.upserted_cities = ()
        self.inserted_observations = []
        self.registered_runs = []
        self.cleanup_calls = []

    def connect_database(self, database_url):
        assert database_url == DATABASE_URL
        return self.connection

    def create_tables(self, connection):
        self.tables_created = True

    def upsert_monitored_cities(self, connection, cities):
        self.upserted_cities = cities
        return len(cities)

    def insert_weather_observation(self, connection, observation):
        self.inserted_observations.append(observation)
        return len(self.inserted_observations)

    def cleanup_old_observations(self, connection, retention_days, max_rows):
        self.cleanup_calls.append((retention_days, max_rows))
        return {
            "deleted_by_retention": 2,
            "deleted_by_max_rows": 1,
            "remaining_rows": 5,
        }

    def register_ingestion_run(
        self,
        connection,
        status,
        records_inserted,
        records_failed,
        error_message=None,
    ):
        self.registered_runs.append(
            (status, records_inserted, records_failed, error_message)
        )
        return 1


class BadConnectionRepository(FakeRepository):
    def connect_database(self, database_url):
        raise RuntimeError(f"could not parse connection string: {database_url}")


class FakeOpenWeatherClient:
    calls = []
    failing_cities = set()

    def __init__(self, settings):
        self.settings = settings

    def fetch_current_weather(self, city):
        self.calls.append((city, self.settings.default_country))
        if city in self.failing_cities:
            raise requests.HTTPError("HTTPError with appid=test-key")
        return make_payload(city)


def make_settings(**overrides):
    settings = load_settings(load_dotenv_file=False)
    values = {
        "database_url": DATABASE_URL,
        "openweather_api_key": "test-key",
        "openweather_collection_cities": (("Brasilia", "BR"), ("Manaus", "BR")),
        "openweather_retention_days": 180,
        "openweather_max_rows": 100000,
        "openweather_store_raw_payload": False,
    }
    values.update(overrides)
    return replace(settings, **values)


def make_payload(city="Brasilia"):
    return {
        "name": city,
        "dt": 1_700_000_000,
        "coord": {"lat": -15.79, "lon": -47.88},
        "main": {
            "temp": 28.0,
            "feels_like": 29.0,
            "humidity": 50,
            "pressure": 1012,
            "grnd_level": 890,
        },
        "rain": {"1h": 0.5},
        "wind": {"speed": 4.0},
        "clouds": {"all": 20},
        "weather": [{"description": "ceu limpo"}],
    }


def test_initialize_weather_database_requires_database_url(capsys) -> None:
    settings = make_settings(database_url="")
    repository = FakeRepository()

    exit_code = initialize_weather_database(settings, repository)
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Configure DATABASE_URL no .env." in output
    assert DATABASE_URL not in output


def test_initialize_weather_database_hides_bad_connection_url(capsys) -> None:
    settings = make_settings()

    exit_code = initialize_weather_database(settings, BadConnectionRepository())
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Erro ao conectar ao banco Postgres. Verifique DATABASE_URL no .env." in output
    assert DATABASE_URL not in output
    assert "secret" not in output


def test_initialize_weather_database_creates_tables_and_cities(capsys) -> None:
    settings = make_settings()
    repository = FakeRepository()

    exit_code = initialize_weather_database(settings, repository)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert repository.tables_created is True
    assert repository.upserted_cities == settings.openweather_collection_cities
    assert "Banco inicializado com sucesso." in output
    assert "Cidades monitoradas configuradas: 2" in output
    assert DATABASE_URL not in output


def test_resolve_collection_cities_uses_manual_city() -> None:
    settings = make_settings()

    cities = resolve_collection_cities(settings, "Brasilia", "BR")

    assert cities == (("Brasilia", "BR"),)


def test_resolve_collection_cities_uses_configured_capitals() -> None:
    settings = make_settings()

    cities = resolve_collection_cities(settings)

    assert cities == (("Brasilia", "BR"), ("Manaus", "BR"))


def test_build_observation_without_raw_payload() -> None:
    observation = build_observation_from_payload(
        make_payload("Brasilia"),
        "Brasilia",
        "BR",
        store_raw_payload=False,
    )

    assert observation["city"] == "Brasilia"
    assert observation["country"] == "BR"
    assert observation["temperature"] == 28.0
    assert observation["wind_speed"] == 14.4
    assert observation["pressure_sea_level_hpa"] == 1012.0
    assert observation["pressure_station_hpa"] == 890.0
    assert observation["clouds"] == 20
    assert observation["weather_description"] == "ceu limpo"
    assert observation["raw_payload"] is None


def test_build_observation_with_raw_payload() -> None:
    payload = make_payload("Brasilia")

    observation = build_observation_from_payload(
        payload,
        "Brasilia",
        "BR",
        store_raw_payload=True,
    )

    assert observation["raw_payload"] == payload


def test_build_observation_without_optional_raw_fields() -> None:
    payload = {
        "name": "Brasilia",
        "main": {"temp": 28.0, "feels_like": 29.0, "humidity": 50, "pressure": 1012},
        "wind": {"speed": 4.0},
    }

    observation = build_observation_from_payload(
        payload,
        "Brasilia",
        "BR",
        store_raw_payload=True,
    )

    assert observation["raw_payload"] == payload
    assert observation["weather_datetime"] is None
    assert observation["clouds"] is None
    assert observation["weather_description"] is None


def test_collect_manual_city_calls_openweather_and_repository(capsys) -> None:
    settings = make_settings(openweather_store_raw_payload=False)
    repository = FakeRepository()
    FakeOpenWeatherClient.calls = []
    FakeOpenWeatherClient.failing_cities = set()

    result = collect_openweather_snapshots(
        settings,
        (("Brasilia", "BR"),),
        repository=repository,
        client_class=FakeOpenWeatherClient,
    )
    output = capsys.readouterr().out

    assert result["exit_code"] == 0
    assert FakeOpenWeatherClient.calls == [("Brasilia", "BR")]
    assert len(repository.inserted_observations) == 1
    assert repository.inserted_observations[0]["raw_payload"] is None
    assert repository.registered_runs[0] == ("success", 1, 0, None)
    assert "Registros inseridos: 1" in output
    assert DATABASE_URL not in output


def test_collect_configured_cities_continues_after_one_failure(capsys) -> None:
    settings = make_settings()
    repository = FakeRepository()
    FakeOpenWeatherClient.calls = []
    FakeOpenWeatherClient.failing_cities = {"Manaus"}

    result = collect_openweather_snapshots(
        settings,
        settings.openweather_collection_cities,
        repository=repository,
        client_class=FakeOpenWeatherClient,
    )
    output = capsys.readouterr().out

    assert result["exit_code"] == 0
    assert result["records_inserted"] == 1
    assert result["records_failed"] == 1
    assert len(repository.inserted_observations) == 1
    assert repository.registered_runs[0][0] == "partial_failed"
    assert "Manaus/BR (HTTPError)" in output
    assert DATABASE_URL not in output
    assert "test-key" not in output


def test_collect_without_database_url_is_safe(capsys) -> None:
    settings = make_settings(database_url="")
    repository = FakeRepository()

    result = collect_openweather_snapshots(
        settings,
        (("Brasilia", "BR"),),
        repository=repository,
        client_class=FakeOpenWeatherClient,
    )
    output = capsys.readouterr().out

    assert result["exit_code"] == 1
    assert "Configure DATABASE_URL no .env." in output
    assert DATABASE_URL not in output


def test_collect_hides_bad_connection_url(capsys) -> None:
    settings = make_settings()

    result = collect_openweather_snapshots(
        settings,
        (("Brasilia", "BR"),),
        repository=BadConnectionRepository(),
        client_class=FakeOpenWeatherClient,
    )
    output = capsys.readouterr().out

    assert result["exit_code"] == 1
    assert "Erro ao conectar ao banco Postgres. Verifique DATABASE_URL no .env." in output
    assert DATABASE_URL not in output
    assert "secret" not in output


def test_collect_runs_cleanup_after_collection() -> None:
    settings = make_settings(openweather_retention_days=90, openweather_max_rows=10)
    repository = FakeRepository()
    FakeOpenWeatherClient.calls = []
    FakeOpenWeatherClient.failing_cities = set()

    result = collect_openweather_snapshots(
        settings,
        (("Brasilia", "BR"),),
        repository=repository,
        client_class=FakeOpenWeatherClient,
    )

    assert repository.cleanup_calls == [(90, 10)]
    assert result["cleanup"] == {
        "deleted_by_retention": 2,
        "deleted_by_max_rows": 1,
        "remaining_rows": 5,
    }
