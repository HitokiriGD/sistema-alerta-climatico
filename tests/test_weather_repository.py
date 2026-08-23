from src.data import weather_repository


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, sql, params=None):
        self.connection.executed.append((sql, params))
        normalized_sql = " ".join(str(sql).split()).upper()
        if normalized_sql.startswith("DELETE"):
            self.rowcount = self.connection.delete_rowcounts.pop(0)
        elif "RETURNING ID" in normalized_sql:
            self.connection.fetchone_results.append((1,))
            self.rowcount = 1
        else:
            self.rowcount = 0

    def fetchone(self):
        if self.connection.fetchone_results:
            return self.connection.fetchone_results.pop(0)
        return None


class FakeConnection:
    def __init__(self):
        self.executed = []
        self.fetchone_results = []
        self.delete_rowcounts = []

    def cursor(self):
        return FakeCursor(self)


def test_connect_database_requires_database_url() -> None:
    try:
        weather_repository.connect_database("")
    except weather_repository.WeatherRepositoryError as error:
        assert str(error) == "Configure DATABASE_URL no .env."
    else:
        raise AssertionError("WeatherRepositoryError was not raised")


def test_create_tables_executes_schema_sql() -> None:
    connection = FakeConnection()

    weather_repository.create_tables(connection)

    executed_sql = "\n".join(sql for sql, _ in connection.executed)
    assert "CREATE TABLE IF NOT EXISTS weather_observations" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS ingestion_runs" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS monitored_cities" in executed_sql
    assert "idx_weather_observations_city_country_collected" in executed_sql


def test_upsert_monitored_cities_inserts_capitals() -> None:
    connection = FakeConnection()

    count = weather_repository.upsert_monitored_cities(
        connection,
        (("Rio Branco", "BR"), ("Belo Horizonte", "BR")),
    )

    assert count == 2
    assert len(connection.executed) == 2
    assert connection.executed[0][1] == ("Rio Branco", "BR")
    assert connection.executed[1][1] == ("Belo Horizonte", "BR")


def test_insert_weather_observation_maps_fields_and_null_raw_payload() -> None:
    connection = FakeConnection()
    observation = {
        "source": "OpenWeather",
        "city": "Brasilia",
        "country": "BR",
        "temperature": 28.0,
        "feels_like": 29.0,
        "humidity": 50.0,
        "precipitation": 0.0,
        "wind_speed": 12.0,
        "pressure_sea_level_hpa": 1012.0,
        "pressure_station_hpa": None,
        "raw_payload": None,
    }

    inserted_id = weather_repository.insert_weather_observation(
        connection,
        observation,
    )

    _, params = connection.executed[0]
    assert inserted_id == 1
    assert params[0] == "OpenWeather"
    assert params[1] == "Brasilia"
    assert params[2] == "BR"
    assert params[-1] is None


def test_register_ingestion_run_records_success() -> None:
    connection = FakeConnection()

    run_id = weather_repository.register_ingestion_run(
        connection,
        status="success",
        records_inserted=27,
        records_failed=0,
        error_message=None,
    )

    _, params = connection.executed[0]
    assert run_id == 1
    assert params == ("success", 27, 0, None)


def test_cleanup_old_observations_removes_by_days_and_max_rows() -> None:
    connection = FakeConnection()
    connection.delete_rowcounts = [3, 5]
    connection.fetchone_results = [(10,), (5,)]

    result = weather_repository.cleanup_old_observations(
        connection,
        retention_days=180,
        max_rows=5,
    )

    assert result == {
        "deleted_by_retention": 3,
        "deleted_by_max_rows": 5,
        "remaining_rows": 5,
    }


def test_cleanup_old_observations_skips_max_rows_when_under_limit() -> None:
    connection = FakeConnection()
    connection.delete_rowcounts = [2]
    connection.fetchone_results = [(4,), (4,)]

    result = weather_repository.cleanup_old_observations(
        connection,
        retention_days=180,
        max_rows=10,
    )

    assert result == {
        "deleted_by_retention": 2,
        "deleted_by_max_rows": 0,
        "remaining_rows": 4,
    }
