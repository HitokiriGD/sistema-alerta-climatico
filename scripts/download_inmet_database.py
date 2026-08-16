import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_settings


DEFAULT_RELEASE_REPO = "HitokiriGD/sistema-alerta-climatico"
DEFAULT_RELEASE_TAG = "inmet-db-v1"
DEFAULT_ASSET_NAME = "inmet_historical.duckdb"
DEFAULT_SHA256 = (
    "2621a5ada2f5b1d2f598690a3868639a406c4efe13fd36dd076f0a08eaa6edbe"
)
GITHUB_API_BASE_URL = "https://api.github.com"
CHUNK_SIZE = 1024 * 1024


class InmetDatabaseDownloadError(RuntimeError):
    """Erro controlado no download da base historica INMET."""


def build_github_headers(
    github_token: str = "",
    accept: str = "application/vnd.github+json",
) -> dict[str, str]:
    """Monta headers para chamadas ao GitHub sem expor o token em logs."""
    headers = {
        "Accept": accept,
        "User-Agent": "sistema-alerta-climatico",
    }
    if github_token.strip():
        headers["Authorization"] = f"Bearer {github_token.strip()}"
    return headers


def raise_friendly_http_error(error: requests.HTTPError) -> None:
    """Converte erros HTTP comuns do GitHub em mensagens acionaveis."""
    response = error.response
    status_code = response.status_code if response is not None else None
    if status_code == 404:
        raise InmetDatabaseDownloadError(
            "Release nao encontrada ou sem permissao. Se o repositorio for "
            "privado, configure GITHUB_TOKEN no .env."
        ) from error
    if status_code in {401, 403}:
        raise InmetDatabaseDownloadError(
            "Token ausente, invalido ou sem permissao de leitura no "
            "repositorio."
        ) from error
    raise InmetDatabaseDownloadError(
        f"Erro HTTP ao baixar a base historica INMET ({status_code})."
    ) from error


def find_asset_by_name(release: dict[str, Any], asset_name: str) -> dict[str, Any]:
    """Seleciona um asset de release pelo nome exato."""
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            return asset
    raise ValueError(
        f"Asset '{asset_name}' nao encontrado na release informada."
    )


def fetch_release_asset(
    release_repo: str,
    release_tag: str,
    asset_name: str,
    github_token: str = "",
    timeout: int = 60,
    session: Any = requests,
) -> dict[str, Any]:
    """Busca a release no GitHub e retorna o asset configurado."""
    if not release_repo.strip() or not release_tag.strip():
        raise ValueError(
            "INMET_DATABASE_RELEASE_REPO e INMET_DATABASE_RELEASE_TAG "
            "devem estar configurados."
        )

    release_url = (
        f"{GITHUB_API_BASE_URL}/repos/{release_repo}/releases/tags/"
        f"{release_tag}"
    )
    response = session.get(
        release_url,
        headers=build_github_headers(github_token),
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise_friendly_http_error(error)
    return find_asset_by_name(response.json(), asset_name)


def resolve_download_source(
    url: str,
    release_repo: str,
    release_tag: str,
    asset_name: str,
    github_token: str = "",
    timeout: int = 60,
    session: Any = requests,
) -> tuple[str, dict[str, str]]:
    """Resolve a URL final de download, priorizando URL direta."""
    headers = build_github_headers(
        github_token,
        accept="application/octet-stream",
    )
    if url.strip():
        return url.strip(), headers

    asset = fetch_release_asset(
        release_repo=release_repo,
        release_tag=release_tag,
        asset_name=asset_name,
        github_token=github_token,
        timeout=timeout,
        session=session,
    )
    download_url = asset.get("url") or asset.get("browser_download_url")
    if not download_url:
        raise ValueError(
            f"Asset '{asset_name}' nao possui URL de download disponivel."
        )
    return str(download_url), headers


def normalize_sha256_digest(expected_sha256: str) -> str:
    """Remove prefixo opcional sha256: do digest configurado."""
    expected_sha256 = expected_sha256.strip()
    if expected_sha256.lower().startswith("sha256:"):
        return expected_sha256.split(":", maxsplit=1)[1].strip()
    return expected_sha256


def calculate_sha256(path: Path) -> str:
    """Calcula SHA256 de um arquivo local."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(path: Path, expected_sha256: str) -> None:
    """Valida SHA256 quando um digest esperado foi configurado."""
    normalized_expected = normalize_sha256_digest(expected_sha256)
    if not normalized_expected:
        return

    actual_sha256 = calculate_sha256(path)
    if actual_sha256.lower() != normalized_expected.lower():
        raise ValueError(
            "SHA256 invalido para a base baixada. "
            f"Esperado: {normalized_expected}; obtido: {actual_sha256}."
        )


def _format_mb(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MB"


def download_file(
    url: str,
    destination: Path,
    headers: dict[str, str],
    force: bool = False,
    expected_sha256: str = "",
    timeout: int = 60,
    session: Any = requests,
) -> Path:
    """Baixa arquivo para destino final usando arquivo temporario."""
    destination = Path(destination)
    if destination.exists() and not force:
        print(
            "Base historica ja existe. Use --force para baixar novamente. "
            f"Caminho: {destination}"
        )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    downloaded = 0
    total_size = 0
    try:
        print(f"Baixando base historica para: {destination}")
        with session.get(
            url,
            stream=True,
            headers=headers,
            timeout=timeout,
        ) as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                raise_friendly_http_error(error)
            total_size = int(response.headers.get("content-length") or 0)
            with temporary_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    output_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        print(
                            "Baixados "
                            f"{_format_mb(downloaded)} de "
                            f"{_format_mb(total_size)}"
                        )
                    else:
                        print(f"Baixados {_format_mb(downloaded)}")

        if not temporary_path.exists() or temporary_path.stat().st_size <= 0:
            raise ValueError("Arquivo temporario nao foi criado corretamente.")

        validate_sha256(temporary_path, expected_sha256)
        temporary_path.replace(destination)
        if not destination.exists() or destination.stat().st_size <= 0:
            raise ValueError("Arquivo final nao foi criado corretamente.")

        print(f"Download concluido: {destination}")
        print(f"Tamanho final: {_format_mb(destination.stat().st_size)}")
        return destination
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def download_inmet_database(
    url: str,
    destination: Path,
    timeout: int = 60,
    force: bool = False,
    expected_sha256: str = DEFAULT_SHA256,
    release_repo: str = DEFAULT_RELEASE_REPO,
    release_tag: str = DEFAULT_RELEASE_TAG,
    asset_name: str = DEFAULT_ASSET_NAME,
    github_token: str = "",
    session: Any = requests,
) -> Path:
    """Baixa a base DuckDB historica por URL direta ou GitHub Release."""
    destination = Path(destination)
    if destination.exists() and not force:
        print(
            "Base historica ja existe. Use --force para baixar novamente. "
            f"Caminho: {destination}"
        )
        return destination

    download_url, headers = resolve_download_source(
        url=url,
        release_repo=release_repo,
        release_tag=release_tag,
        asset_name=asset_name,
        github_token=github_token,
        timeout=timeout,
        session=session,
    )
    return download_file(
        url=download_url,
        destination=destination,
        headers=headers,
        force=force,
        expected_sha256=expected_sha256,
        timeout=timeout,
        session=session,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa a base historica INMET processada em DuckDB."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Baixa novamente mesmo se o arquivo final ja existir.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa download da base DuckDB configurada no .env."""
    args = parse_args()
    settings = load_settings()
    try:
        saved_path = download_inmet_database(
            url=settings.inmet_database_url,
            destination=Path(settings.inmet_database_path),
            force=args.force,
            expected_sha256=settings.inmet_database_sha256,
            release_repo=settings.inmet_database_release_repo,
            release_tag=settings.inmet_database_release_tag,
            asset_name=settings.inmet_database_asset_name,
            github_token=settings.github_token,
        )
        print(f"Base historica DuckDB disponivel em: {saved_path}")
    except (InmetDatabaseDownloadError, requests.RequestException, ValueError) as error:
        print(f"Erro ao baixar a base historica INMET: {error}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
