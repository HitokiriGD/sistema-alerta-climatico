from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd

from scripts.build_ml_dataset import build_ml_dataset_file
from scripts.build_ml_dataset import save_dataset
from src.ml.dataset import build_labeled_dataset
from src.ml.dataset import parse_station_codes
from src.ml.dataset import summarize_labeled_dataset


def sample_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "station_code": "A001",
                "station_name": "BRASILIA",
                "state": "DF",
                "datetime": "2025-01-02 03:00:00",
                "temperature": 25.0,
                "humidity": 80.0,
                "precipitation": 0.0,
                "wind_speed": 8.0,
                "pressure": 900.0,
            },
            {
                "station_code": "A101",
                "station_name": "MANAUS",
                "state": "AM",
                "datetime": "2025-09-10 14:00:00",
                "temperature": 36.0,
                "humidity": 18.0,
                "precipitation": 0.0,
                "wind_speed": 12.0,
                "pressure": 1005.0,
            },
        ]
    )


def test_build_labeled_dataset_creates_datetime_features() -> None:
    dataset = build_labeled_dataset(sample_history())

    first = dataset.iloc[0]
    assert first["year"] == 2025
    assert first["month"] == 1
    assert first["day"] == 2
    assert first["hour"] == 3
    assert first["day_of_year"] == 2


def test_build_labeled_dataset_uses_temperature_as_feels_like_fallback() -> None:
    dataset = build_labeled_dataset(sample_history())

    assert list(dataset["feels_like"]) == [25.0, 36.0]


def test_build_labeled_dataset_removes_rows_without_minimum_data() -> None:
    history = sample_history()
    history.loc[0, "humidity"] = None

    dataset = build_labeled_dataset(history)

    assert len(dataset) == 1
    assert dataset.loc[0, "station_code"] == "A101"


def test_build_labeled_dataset_generates_risk_level() -> None:
    dataset = build_labeled_dataset(sample_history())

    assert "risk_level" in dataset.columns
    assert set(dataset["risk_level"]) == {"baixo", "alto"}


def test_build_labeled_dataset_generates_event_type() -> None:
    dataset = build_labeled_dataset(sample_history())

    assert "event_type" in dataset.columns
    assert set(dataset["event_type"]) == {
        "sem_risco_relevante",
        "baixa_umidade",
    }


def test_summarize_labeled_dataset_returns_basic_counts() -> None:
    dataset = build_labeled_dataset(sample_history())

    summary = summarize_labeled_dataset(dataset)

    assert summary["total_records"] == 2
    assert summary["station_count"] == 2
    assert summary["risk_level_distribution"] == {"baixo": 1, "alto": 1}
    assert summary["event_type_distribution"]["baixa_umidade"] == 1


def test_build_labeled_dataset_does_not_mutate_original_dataframe() -> None:
    history = sample_history()
    original_columns = list(history.columns)

    build_labeled_dataset(history)

    assert list(history.columns) == original_columns
    assert "year" not in history.columns
    assert "risk_level" not in history.columns


def test_parse_station_codes_strips_and_uppercases_values() -> None:
    assert parse_station_codes(" a001, A101 ,,a312 ") == [
        "A001",
        "A101",
        "A312",
    ]


def test_build_ml_dataset_file_passes_limit_to_loader(tmp_path) -> None:
    db_path = tmp_path / "inmet.duckdb"
    db_path.write_text("fake")
    saved: dict[str, object] = {}

    def fake_loader(db_path_arg, start_year, end_year, stations, limit):
        saved["loader_args"] = (
            db_path_arg,
            start_year,
            end_year,
            stations,
            limit,
        )
        return sample_history()

    def fake_saver(dataset, output_path, output_format):
        saved["rows"] = len(dataset)
        saved["output_path"] = output_path
        saved["format"] = output_format

    summary = build_ml_dataset_file(
        settings=SimpleNamespace(inmet_database_path=str(db_path)),
        start_year=2025,
        end_year=2026,
        output_path=tmp_path / "dataset.csv",
        output_format="csv",
        limit=10000,
        loader=fake_loader,
        saver=fake_saver,
    )

    assert saved["loader_args"] == (db_path, 2025, 2026, None, 10000)
    assert saved["rows"] == 2
    assert summary["total_records"] == 2


def test_build_ml_dataset_file_passes_station_filter_to_loader(tmp_path) -> None:
    db_path = tmp_path / "inmet.duckdb"
    db_path.write_text("fake")
    captured: dict[str, object] = {}

    def fake_loader(db_path_arg, start_year, end_year, stations, limit):
        captured["stations"] = stations
        return sample_history()

    build_ml_dataset_file(
        settings=SimpleNamespace(inmet_database_path=str(db_path)),
        start_year=2020,
        end_year=2026,
        stations="A001,a101",
        output_path=tmp_path / "dataset.csv",
        output_format="csv",
        loader=fake_loader,
        saver=lambda dataset, output_path, output_format: None,
    )

    assert captured["stations"] == ["A001", "A101"]


def test_save_dataset_writes_csv_file(tmp_path) -> None:
    dataset = build_labeled_dataset(sample_history())
    output_path = tmp_path / "ml_training_dataset.csv"

    save_dataset(dataset, output_path, "csv")

    assert output_path.exists()
    saved_dataset = pd.read_csv(output_path)
    assert len(saved_dataset) == 2


def test_save_dataset_writes_parquet_file_with_duckdb(tmp_path) -> None:
    dataset = build_labeled_dataset(sample_history())
    output_path = tmp_path / "ml_training_dataset.parquet"

    save_dataset(dataset, output_path, "parquet")

    assert output_path.exists()
    with duckdb.connect() as connection:
        count = connection.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(output_path)],
        ).fetchone()[0]
    assert count == 2


def test_generated_datasets_are_ignored_by_git() -> None:
    gitignore = Path(".gitignore").read_text()

    assert "data/processed/*" in gitignore
    assert "data/processed/*.parquet" in gitignore
    assert "data/processed/*.csv" in gitignore
