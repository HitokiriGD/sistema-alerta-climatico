import argparse
import sys
from pathlib import Path
from typing import Any
from typing import Callable

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.config.settings import load_settings
from src.data.inmet_database import database_exists
from src.ml.dataset import build_labeled_dataset
from src.ml.dataset import load_inmet_history_from_duckdb
from src.ml.dataset import parse_station_codes
from src.ml.dataset import summarize_labeled_dataset


DEFAULT_OUTPUT_PATH = Path("data/processed/ml_training_dataset.parquet")


def parse_args() -> argparse.Namespace:
    """Le parametros para gerar o dataset supervisionado inicial."""
    parser = argparse.ArgumentParser(
        description="Gera dataset rotulado de ML a partir do DuckDB INMET.",
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--stations",
        default="",
        help="Codigos separados por virgula, por exemplo A001,A101,A312.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Caminho do dataset gerado.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de registros para amostras rapidas.",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Formato de saida.",
    )
    return parser.parse_args()


def build_ml_dataset_file(
    settings: Settings,
    start_year: int,
    end_year: int,
    stations: str = "",
    output_path: Path = DEFAULT_OUTPUT_PATH,
    output_format: str = "parquet",
    limit: int | None = None,
    loader: Callable[..., pd.DataFrame] = load_inmet_history_from_duckdb,
    saver: Callable[[pd.DataFrame, Path, str], None] | None = None,
) -> dict[str, Any]:
    """Gera, salva e resume o dataset rotulado de treinamento inicial."""
    if start_year > end_year:
        raise ValueError("Ano inicial nao pode ser maior que ano final.")

    db_path = Path(settings.inmet_database_path)
    if not database_exists(db_path):
        raise FileNotFoundError(
            "Base DuckDB INMET nao encontrada. Execute "
            "python scripts/download_inmet_database.py ou gere a base local.",
        )

    station_codes = parse_station_codes(stations)
    history = loader(
        db_path,
        start_year,
        end_year,
        station_codes or None,
        limit,
    )
    labeled_dataset = build_labeled_dataset(history)

    destination = _resolve_output_path(output_path, output_format)
    save_function = saver or save_dataset
    save_function(labeled_dataset, destination, output_format)

    summary = summarize_labeled_dataset(labeled_dataset)
    summary["saved_path"] = str(destination)
    summary["requested_period"] = f"{start_year}-{end_year}"
    summary["station_filter"] = station_codes
    return summary


def save_dataset(dataset: pd.DataFrame, output_path: Path, output_format: str) -> None:
    """Salva o dataset em CSV ou Parquet sem versionar o arquivo gerado."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        dataset.to_csv(output_path, index=False)
        return

    with duckdb.connect() as connection:
        connection.register("ml_training_dataset", dataset)
        safe_path = str(output_path).replace("'", "''")
        connection.execute(
            f"COPY ml_training_dataset TO '{safe_path}' (FORMAT PARQUET)"
        )


def print_summary(summary: dict[str, Any]) -> None:
    """Imprime resumo seguro do dataset gerado."""
    print(f"Total de registros: {summary['total_records']}")
    print(f"Periodo solicitado: {summary['requested_period']}")
    print(f"Primeiro registro: {_format_datetime(summary['start_datetime'])}")
    print(f"Ultimo registro: {_format_datetime(summary['end_datetime'])}")
    print(f"Quantidade de estacoes: {summary['station_count']}")
    print("Distribuicao por risk_level:")
    _print_distribution(summary["risk_level_distribution"])
    print("Distribuicao por event_type:")
    _print_distribution(summary["event_type_distribution"])
    print(f"Dataset salvo em: {summary['saved_path']}")


def main() -> int:
    args = parse_args()
    settings = load_settings()

    try:
        summary = build_ml_dataset_file(
            settings=settings,
            start_year=args.start_year,
            end_year=args.end_year,
            stations=args.stations,
            output_path=args.output,
            output_format=args.format,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print_summary(summary)
    return 0


def _resolve_output_path(output_path: Path, output_format: str) -> Path:
    if output_path.suffix:
        return output_path
    return output_path.with_suffix(f".{output_format}")


def _format_datetime(value: Any) -> str:
    if value is None or pd.isna(value):
        return "sem registros"
    return str(value)


def _print_distribution(distribution: dict[str, int]) -> None:
    if not distribution:
        print("- sem registros")
        return

    for label, count in distribution.items():
        print(f"- {label}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
