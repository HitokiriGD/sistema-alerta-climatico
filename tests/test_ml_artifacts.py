from pathlib import Path

from src.ml.artifacts import ensure_ml_artifacts_available


class FakeSettings:
    ml_artifacts_release_repo = "owner/repo"
    ml_artifacts_release_tag = "ml-v1"
    ml_model_asset_name = "risk_level_model.joblib"
    ml_model_metadata_asset_name = "risk_level_model_metadata.json"
    ml_evaluation_report_asset_name = "risk_level_report.json"
    ml_model_sha256 = "model-sha"
    ml_model_metadata_sha256 = "metadata-sha"
    ml_evaluation_report_sha256 = "report-sha"
    github_token = "secret-token"
    database_url = "postgresql://user:secret@example/db"


def artifact_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "data" / "models" / "risk_level_model.joblib",
        tmp_path / "data" / "models" / "risk_level_model_metadata.json",
        tmp_path / "data" / "reports" / "risk_level_report.json",
    )


def test_ensure_ml_artifacts_available_skips_download_when_all_exist(
    tmp_path,
) -> None:
    model_path, metadata_path, report_path = artifact_paths(tmp_path)
    for path in [model_path, metadata_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing")
    calls = []

    def fake_downloader(**kwargs):
        calls.append(kwargs)
        raise AssertionError("download nao deveria ser chamado")

    status = ensure_ml_artifacts_available(
        FakeSettings(),
        downloader=fake_downloader,
        model_path=model_path,
        metadata_path=metadata_path,
        evaluation_report_path=report_path,
    )

    assert status["available"] is True
    assert status["downloaded"] is False
    assert status["missing_files"] == []
    assert calls == []


def test_ensure_ml_artifacts_available_downloads_missing_files(tmp_path) -> None:
    model_path, metadata_path, report_path = artifact_paths(tmp_path)
    calls = []

    def fake_downloader(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["destination"])
        destination.write_bytes(b"downloaded")
        return destination

    status = ensure_ml_artifacts_available(
        FakeSettings(),
        downloader=fake_downloader,
        model_path=model_path,
        metadata_path=metadata_path,
        evaluation_report_path=report_path,
    )

    assert status["available"] is True
    assert status["downloaded"] is True
    assert status["missing_files"] == []
    assert model_path.exists()
    assert metadata_path.exists()
    assert report_path.exists()
    assert model_path.parent.is_dir()
    assert report_path.parent.is_dir()
    assert [call["asset_name"] for call in calls] == [
        "risk_level_model.joblib",
        "risk_level_model_metadata.json",
        "risk_level_report.json",
    ]
    assert [call["expected_sha256"] for call in calls] == [
        "model-sha",
        "metadata-sha",
        "report-sha",
    ]
    assert all(call["github_token"] == "secret-token" for call in calls)


def test_ensure_ml_artifacts_available_downloads_only_missing_files(tmp_path) -> None:
    model_path, metadata_path, report_path = artifact_paths(tmp_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"existing")
    calls = []

    def fake_downloader(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["destination"])
        destination.write_bytes(b"downloaded")
        return destination

    status = ensure_ml_artifacts_available(
        FakeSettings(),
        downloader=fake_downloader,
        model_path=model_path,
        metadata_path=metadata_path,
        evaluation_report_path=report_path,
    )

    assert status["available"] is True
    assert status["downloaded"] is True
    assert [call["asset_name"] for call in calls] == [
        "risk_level_model_metadata.json",
        "risk_level_report.json",
    ]


def test_ensure_ml_artifacts_available_returns_safe_error_on_download_failure(
    tmp_path,
) -> None:
    model_path, metadata_path, report_path = artifact_paths(tmp_path)

    def fake_downloader(**kwargs):
        raise ValueError(
            "falha usando secret-token e postgresql://user:secret@example/db"
        )

    status = ensure_ml_artifacts_available(
        FakeSettings(),
        downloader=fake_downloader,
        model_path=model_path,
        metadata_path=metadata_path,
        evaluation_report_path=report_path,
    )

    assert status["available"] is False
    assert status["downloaded"] is False
    assert str(model_path) in status["missing_files"]
    assert model_path.parent.is_dir()
    assert report_path.parent.is_dir()
    assert "secret-token" not in str(status["error_message"])
    assert "postgresql://user:secret@example/db" not in str(status["error_message"])
    assert "[valor sensivel oculto]" in str(status["error_message"])


def test_ensure_ml_artifacts_available_reports_invalid_sha_as_controlled_error(
    tmp_path,
) -> None:
    model_path, metadata_path, report_path = artifact_paths(tmp_path)

    def fake_downloader(**kwargs):
        raise ValueError("SHA256 invalido para secret-token")

    status = ensure_ml_artifacts_available(
        FakeSettings(),
        downloader=fake_downloader,
        model_path=model_path,
        metadata_path=metadata_path,
        evaluation_report_path=report_path,
    )

    assert status["available"] is False
    assert "SHA256 invalido" in str(status["error_message"])
    assert "secret-token" not in str(status["error_message"])
