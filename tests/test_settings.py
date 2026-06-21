from src.config.settings import load_settings


def test_load_settings_uses_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_CITY", raising=False)
    monkeypatch.delenv("DEFAULT_COUNTRY", raising=False)

    settings = load_settings()

    assert settings.default_city == "Brasilia"
    assert settings.default_country == "BR"
    assert settings.openweather_api_key == ""
