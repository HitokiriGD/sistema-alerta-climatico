from src.config.settings import load_settings


def test_load_settings_uses_defaults_without_dotenv(monkeypatch) -> None:
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_CITY", raising=False)
    monkeypatch.delenv("DEFAULT_COUNTRY", raising=False)
    monkeypatch.delenv("OPENWEATHER_BASE_URL", raising=False)
    monkeypatch.delenv("INMET_HISTORICAL_ZIP_DIR", raising=False)
    monkeypatch.delenv("INMET_HISTORICAL_START_YEAR", raising=False)
    monkeypatch.delenv("INMET_HISTORICAL_END_YEAR", raising=False)
    monkeypatch.delenv("INMET_PROCESSED_DATA_PATH", raising=False)
    monkeypatch.delenv("INMET_STATION_CATALOG_PATH", raising=False)
    monkeypatch.delenv("INMET_DATABASE_PATH", raising=False)
    monkeypatch.delenv("INMET_DATABASE_URL", raising=False)

    settings = load_settings(load_dotenv_file=False)

    assert settings.default_city == "Brasilia"
    assert settings.default_country == "BR"
    assert settings.openweather_api_key == ""
    assert settings.openweather_base_url == "https://api.openweathermap.org/data/2.5"
    assert settings.inmet_historical_zip_dir == "data/raw/inmet/zips"
    assert settings.inmet_historical_start_year == 2020
    assert settings.inmet_historical_end_year == 2026
    assert settings.inmet_processed_data_path == "data/processed/inmet_hourly.parquet"
    assert (
        settings.inmet_station_catalog_path
        == "data/processed/inmet_station_catalog.csv"
    )
    assert settings.inmet_database_path == "data/processed/inmet_historical.duckdb"
    assert settings.inmet_database_url == ""


def test_load_settings_reads_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("OPENWEATHER_API_KEY", "fake_key_for_test")
    monkeypatch.setenv("DEFAULT_CITY", "Sao Paulo")
    monkeypatch.setenv("DEFAULT_COUNTRY", "BR")
    monkeypatch.setenv("OPENWEATHER_BASE_URL", "https://openweather.example.test")
    monkeypatch.setenv("INMET_HISTORICAL_ZIP_DIR", "data/raw/teste")
    monkeypatch.setenv("INMET_HISTORICAL_START_YEAR", "2018")
    monkeypatch.setenv("INMET_HISTORICAL_END_YEAR", "2022")
    monkeypatch.setenv("INMET_PROCESSED_DATA_PATH", "data/processed/teste.csv")
    monkeypatch.setenv(
        "INMET_STATION_CATALOG_PATH",
        "data/processed/catalogo.csv",
    )
    monkeypatch.setenv(
        "INMET_DATABASE_PATH",
        "data/processed/teste.duckdb",
    )
    monkeypatch.setenv(
        "INMET_DATABASE_URL",
        "https://example.test/inmet_historical.duckdb",
    )

    settings = load_settings(load_dotenv_file=False)

    assert settings.openweather_api_key == "fake_key_for_test"
    assert settings.default_city == "Sao Paulo"
    assert settings.default_country == "BR"
    assert settings.openweather_base_url == "https://openweather.example.test"
    assert settings.inmet_historical_zip_dir == "data/raw/teste"
    assert settings.inmet_historical_start_year == 2018
    assert settings.inmet_historical_end_year == 2022
    assert settings.inmet_processed_data_path == "data/processed/teste.csv"
    assert settings.inmet_station_catalog_path == "data/processed/catalogo.csv"
    assert settings.inmet_database_path == "data/processed/teste.duckdb"
    assert settings.inmet_database_url == "https://example.test/inmet_historical.duckdb"
