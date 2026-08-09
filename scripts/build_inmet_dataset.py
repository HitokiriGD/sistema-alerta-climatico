import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.config.settings import load_settings
from src.data.inmet_client import InmetClient
from src.processing.preprocessing import preprocess_weather_dataframe


def build_processed_inmet_dataset(
    station_code: str,
    start_year: int | None = None,
    end_year: int | None = None,
    output_path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Carrega historico INMET, pre-processa e salva o dataset tratado."""
    current_settings = settings or load_settings()
    destination = output_path or Path(current_settings.inmet_processed_data_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    historical_data = InmetClient(current_settings).load_station_history(
        station_code=station_code,
        start_year=start_year,
        end_year=end_year,
    )
    processed_data = preprocess_weather_dataframe(historical_data)

    if destination.suffix.lower() == ".parquet":
        try:
            processed_data.to_parquet(destination, index=False)
            return destination
        except ImportError:
            destination = destination.with_suffix(".csv")

    processed_data.to_csv(destination, index=False)
    return destination


def parse_args() -> argparse.Namespace:
    """Le parametros de linha de comando do pre-processamento INMET."""
    parser = argparse.ArgumentParser(
        description="Gera dataset tratado a partir dos ZIPs historicos do INMET.",
    )
    parser.add_argument(
        "--station",
        required=True,
        help="Codigo da estacao INMET, por exemplo A001.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Ano inicial. Usa o .env quando omitido.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Ano final. Usa o .env quando omitido.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Caminho de saida. Padrao: INMET_PROCESSED_DATA_PATH.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa a geracao local do dataset tratado do INMET."""
    args = parse_args()
    saved_path = build_processed_inmet_dataset(
        station_code=args.station,
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output,
    )
    print(f"Dataset tratado salvo em: {saved_path}")


if __name__ == "__main__":
    main()
