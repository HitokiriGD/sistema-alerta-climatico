from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from scripts.download_inmet_database import InmetDatabaseDownloadError
from scripts.download_inmet_database import download_inmet_database
from src.config.settings import Settings
from src.config.settings import sanitize_sensitive_text
from src.ml.predict import DEFAULT_METADATA_PATH
from src.ml.predict import DEFAULT_MODEL_PATH


DEFAULT_EVALUATION_REPORT_PATH = Path(
    "data/reports/risk_level_robust_evaluation_temporal_report.json"
)


@dataclass(frozen=True)
class MlArtifact:
    """Representa um artefato necessario para a camada ML do dashboard."""

    path: Path
    asset_name: str
    expected_sha256: str


def ensure_ml_artifacts_available(
    settings: Settings,
    downloader: Callable[..., Path] = download_inmet_database,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    evaluation_report_path: str | Path = DEFAULT_EVALUATION_REPORT_PATH,
) -> dict[str, object]:
    """Garante artefatos ML locais, baixando da GitHub Release se faltarem."""
    artifacts = _build_ml_artifact_list(
        settings,
        model_path=model_path,
        metadata_path=metadata_path,
        evaluation_report_path=evaluation_report_path,
    )
    missing_files = _missing_files(artifacts)
    sensitive_values = [
        settings.github_token,
        settings.database_url,
    ]

    if not missing_files:
        return {
            "available": True,
            "downloaded": False,
            "missing_files": [],
            "message": "Artefatos de ML ja disponiveis localmente.",
            "error_message": "",
        }

    for artifact in artifacts:
        artifact.path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_any = False
    try:
        for artifact in artifacts:
            if artifact.path.exists():
                continue
            downloader(
                url="",
                destination=artifact.path,
                force=False,
                expected_sha256=artifact.expected_sha256,
                release_repo=settings.ml_artifacts_release_repo,
                release_tag=settings.ml_artifacts_release_tag,
                asset_name=artifact.asset_name,
                github_token=settings.github_token,
            )
            downloaded_any = True
    except (
        InmetDatabaseDownloadError,
        requests.RequestException,
        ValueError,
        OSError,
    ) as error:
        return {
            "available": False,
            "downloaded": downloaded_any,
            "missing_files": _missing_files(artifacts),
            "message": "",
            "error_message": sanitize_sensitive_text(error, sensitive_values),
        }

    missing_after_download = _missing_files(artifacts)
    if missing_after_download:
        return {
            "available": False,
            "downloaded": downloaded_any,
            "missing_files": missing_after_download,
            "message": "",
            "error_message": (
                "Download finalizado, mas alguns artefatos de ML ainda nao "
                "foram encontrados no caminho configurado."
            ),
        }

    return {
        "available": True,
        "downloaded": downloaded_any,
        "missing_files": [],
        "message": "Artefatos de ML carregados com sucesso.",
        "error_message": "",
    }


def _build_ml_artifact_list(
    settings: Settings,
    model_path: str | Path,
    metadata_path: str | Path,
    evaluation_report_path: str | Path,
) -> list[MlArtifact]:
    return [
        MlArtifact(
            path=Path(model_path),
            asset_name=settings.ml_model_asset_name,
            expected_sha256=_configured_sha256(settings.ml_model_sha256),
        ),
        MlArtifact(
            path=Path(metadata_path),
            asset_name=settings.ml_model_metadata_asset_name,
            expected_sha256=_configured_sha256(settings.ml_model_metadata_sha256),
        ),
        MlArtifact(
            path=Path(evaluation_report_path),
            asset_name=settings.ml_evaluation_report_asset_name,
            expected_sha256=_configured_sha256(settings.ml_evaluation_report_sha256),
        ),
    ]


def _missing_files(artifacts: list[MlArtifact]) -> list[str]:
    return [
        str(artifact.path)
        for artifact in artifacts
        if not artifact.path.exists()
    ]


def _configured_sha256(value: str) -> str:
    value = value.strip()
    if not value or value.upper().startswith("COLE_AQUI"):
        return ""
    return value
