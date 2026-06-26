from src.config.settings import load_settings


def test_load_settings_uses_defaults_without_dotenv(monkeypatch) -> None:
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_CITY", raising=False)
    monkeypatch.delenv("DEFAULT_COUNTRY", raising=False)
    monkeypatch.delenv("INMET_BASE_URL", raising=False)
    monkeypatch.delenv("OPENWEATHER_BASE_URL", raising=False)

    settings = load_settings(load_dotenv_file=False)

    assert settings.default_city == "Brasilia"
    assert settings.default_country == "BR"
    assert settings.openweather_api_key == ""
    assert settings.inmet_base_url == "https://apitempo.inmet.gov.br"
    assert settings.openweather_base_url == "https://api.openweathermap.org/data/2.5"


def test_load_settings_reads_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("OPENWEATHER_API_KEY", "fake_key_for_test")
    monkeypatch.setenv("DEFAULT_CITY", "Sao Paulo")
    monkeypatch.setenv("DEFAULT_COUNTRY", "BR")
    monkeypatch.setenv("INMET_BASE_URL", "https://inmet.example.test")
    monkeypatch.setenv("OPENWEATHER_BASE_URL", "https://openweather.example.test")

    settings = load_settings(load_dotenv_file=False)

    assert settings.openweather_api_key == "fake_key_for_test"
    assert settings.default_city == "Sao Paulo"
    assert settings.default_country == "BR"
    assert settings.inmet_base_url == "https://inmet.example.test"
    assert settings.openweather_base_url == "https://openweather.example.test"
