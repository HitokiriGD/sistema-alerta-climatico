import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_settings


def download_inmet_database(
    url: str,
    destination: Path,
    timeout: int = 60,
) -> Path:
    """Baixa uma base DuckDB historica pronta para uso local."""
    if not url.strip():
        raise ValueError("INMET_DATABASE_URL nao foi configurada.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)

    return destination


def main() -> None:
    """Executa download da base DuckDB configurada no .env."""
    settings = load_settings()
    saved_path = download_inmet_database(
        settings.inmet_database_url,
        Path(settings.inmet_database_path),
    )
    print(f"Base historica DuckDB baixada em: {saved_path}")


if __name__ == "__main__":
    main()
