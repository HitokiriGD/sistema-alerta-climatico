import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.config.settings import load_settings
from src.data.inmet_client import InmetClient


def build_inmet_station_catalog(
    output_path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Gera CSV local com o catalogo de estacoes INMET."""
    current_settings = settings or load_settings()
    destination = output_path or Path(current_settings.inmet_station_catalog_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    catalog = InmetClient(current_settings).build_station_catalog()
    catalog.to_csv(destination, index=False)
    return destination


def main() -> None:
    """Executa a geracao local do catalogo de estacoes."""
    saved_path = build_inmet_station_catalog()
    print(f"Catalogo de estacoes INMET salvo em: {saved_path}")


if __name__ == "__main__":
    main()
