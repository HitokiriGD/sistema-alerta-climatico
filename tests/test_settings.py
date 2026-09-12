from src.config.settings import BRAZILIAN_CAPITALS
from src.config.settings import load_settings
from src.config.settings import parse_openweather_collection_cities
from src.config.settings import sanitize_sensitive_text


def test_load_settings_uses_defaults_without_dotenv(monkeypatch) -> None:
    monkeypatch.setattr("src.config.settings._read_streamlit_secret", lambda key: "")
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
    monkeypatch.delenv("INMET_DATABASE_RELEASE_REPO", raising=False)
    monkeypatch.delenv("INMET_DATABASE_RELEASE_TAG", raising=False)
    monkeypatch.delenv("INMET_DATABASE_ASSET_NAME", raising=False)
    monkeypatch.delenv("INMET_DATABASE_SHA256", raising=False)
    monkeypatch.delenv("ML_ARTIFACTS_RELEASE_REPO", raising=False)
    monkeypatch.delenv("ML_ARTIFACTS_RELEASE_TAG", raising=False)
    monkeypatch.delenv("ML_MODEL_ASSET_NAME", raising=False)
    monkeypatch.delenv("ML_MODEL_METADATA_ASSET_NAME", raising=False)
    monkeypatch.delenv("ML_EVALUATION_REPORT_ASSET_NAME", raising=False)
    monkeypatch.delenv("ML_MODEL_SHA256", raising=False)
    monkeypatch.delenv("ML_MODEL_METADATA_SHA256", raising=False)
    monkeypatch.delenv("ML_EVALUATION_REPORT_SHA256", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENWEATHER_COLLECTION_CITIES", raising=False)
    monkeypatch.delenv("OPENWEATHER_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("OPENWEATHER_MAX_ROWS", raising=False)
    monkeypatch.delenv("OPENWEATHER_STORE_RAW_PAYLOAD", raising=False)

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
    assert (
        settings.inmet_database_release_repo
        == "HitokiriGD/sistema-alerta-climatico"
    )
    assert settings.inmet_database_release_tag == "inmet-db-v1"
    assert settings.inmet_database_asset_name == "inmet_historical.duckdb"
    assert (
        settings.inmet_database_sha256
        == "2621a5ada2f5b1d2f598690a3868639a406c4efe13fd36dd076f0a08eaa6edbe"
    )
    assert settings.ml_artifacts_release_repo == "HitokiriGD/sistema-alerta-climatico"
    assert settings.ml_artifacts_release_tag == "ml-artifacts-v1"
    assert settings.ml_model_asset_name == "risk_level_model.joblib"
    assert settings.ml_model_metadata_asset_name == "risk_level_model_metadata.json"
    assert (
        settings.ml_evaluation_report_asset_name
        == "risk_level_robust_evaluation_temporal_report.json"
    )
    assert settings.ml_model_sha256 == ""
    assert settings.ml_model_metadata_sha256 == ""
    assert settings.ml_evaluation_report_sha256 == ""
    assert settings.github_token == ""
    assert settings.database_url == ""
    assert settings.openweather_collection_cities == BRAZILIAN_CAPITALS
    assert settings.openweather_retention_days == 180
    assert settings.openweather_max_rows == 100000
    assert settings.openweather_store_raw_payload is False


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
    monkeypatch.setenv(
        "INMET_DATABASE_RELEASE_REPO",
        "owner/repo",
    )
    monkeypatch.setenv("INMET_DATABASE_RELEASE_TAG", "db-v2")
    monkeypatch.setenv("INMET_DATABASE_ASSET_NAME", "test.duckdb")
    monkeypatch.setenv("INMET_DATABASE_SHA256", "abc123")
    monkeypatch.setenv("ML_ARTIFACTS_RELEASE_REPO", "owner/ml-repo")
    monkeypatch.setenv("ML_ARTIFACTS_RELEASE_TAG", "ml-v2")
    monkeypatch.setenv("ML_MODEL_ASSET_NAME", "model.joblib")
    monkeypatch.setenv("ML_MODEL_METADATA_ASSET_NAME", "metadata.json")
    monkeypatch.setenv("ML_EVALUATION_REPORT_ASSET_NAME", "report.json")
    monkeypatch.setenv("ML_MODEL_SHA256", "model-sha")
    monkeypatch.setenv("ML_MODEL_METADATA_SHA256", "metadata-sha")
    monkeypatch.setenv("ML_EVALUATION_REPORT_SHA256", "report-sha")
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@example/db")
    monkeypatch.setenv(
        "OPENWEATHER_COLLECTION_CITIES",
        "Rio Branco:BR,Belo Horizonte:BR,Teste:UY",
    )
    monkeypatch.setenv("OPENWEATHER_RETENTION_DAYS", "90")
    monkeypatch.setenv("OPENWEATHER_MAX_ROWS", "5000")
    monkeypatch.setenv("OPENWEATHER_STORE_RAW_PAYLOAD", "true")

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
    assert settings.inmet_database_release_repo == "owner/repo"
    assert settings.inmet_database_release_tag == "db-v2"
    assert settings.inmet_database_asset_name == "test.duckdb"
    assert settings.inmet_database_sha256 == "abc123"
    assert settings.ml_artifacts_release_repo == "owner/ml-repo"
    assert settings.ml_artifacts_release_tag == "ml-v2"
    assert settings.ml_model_asset_name == "model.joblib"
    assert settings.ml_model_metadata_asset_name == "metadata.json"
    assert settings.ml_evaluation_report_asset_name == "report.json"
    assert settings.ml_model_sha256 == "model-sha"
    assert settings.ml_model_metadata_sha256 == "metadata-sha"
    assert settings.ml_evaluation_report_sha256 == "report-sha"
    assert settings.github_token == "secret-token"
    assert settings.database_url == "postgresql://user:secret@example/db"
    assert settings.openweather_collection_cities == (
        ("Rio Branco", "BR"),
        ("Belo Horizonte", "BR"),
        ("Teste", "UY"),
    )
    assert settings.openweather_retention_days == 90
    assert settings.openweather_max_rows == 5000
    assert settings.openweather_store_raw_payload is True


def test_load_settings_reads_streamlit_secrets_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    secret_values = {
        "OPENWEATHER_API_KEY": "secret-openweather",
        "DATABASE_URL": "postgresql://user:secret@example/db",
    }
    monkeypatch.setattr(
        "src.config.settings._read_streamlit_secret",
        lambda key: secret_values.get(key, ""),
    )

    settings = load_settings(load_dotenv_file=False)

    assert settings.openweather_api_key == "secret-openweather"
    assert settings.database_url == "postgresql://user:secret@example/db"


def test_sanitize_sensitive_text_hides_configured_values(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret_token")

    message = sanitize_sensitive_text(
        "Falha usando ghp_secret_token e postgresql://user:secret@example/db",
        sensitive_values=["postgresql://user:secret@example/db"],
    )

    assert "ghp_secret_token" not in message
    assert "postgresql://user:secret@example/db" not in message
    assert "[valor sensivel oculto]" in message


def test_parse_openweather_collection_cities_handles_spaces() -> None:
    cities = parse_openweather_collection_cities(
        "Rio Branco:BR, Belo Horizonte:BR, Montevideo:uy"
    )

    assert cities == (
        ("Rio Branco", "BR"),
        ("Belo Horizonte", "BR"),
        ("Montevideo", "UY"),
    )
