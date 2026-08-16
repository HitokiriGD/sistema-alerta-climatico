import hashlib
from pathlib import Path

import pytest
import requests

from scripts.download_inmet_database import build_github_headers
from scripts.download_inmet_database import download_inmet_database
from scripts.download_inmet_database import find_asset_by_name
from scripts.download_inmet_database import InmetDatabaseDownloadError
from scripts.download_inmet_database import validate_sha256


class FakeHttpErrorResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeResponse:
    def __init__(
        self,
        content: bytes = b"",
        json_data: dict | None = None,
        headers: dict[str, str] | None = None,
        iter_error: Exception | None = None,
        status_code: int = 200,
    ) -> None:
        self.content = content
        self.json_data = json_data or {}
        self.headers = headers or {"content-length": str(len(content))}
        self.iter_error = iter_error
        self.status_code = status_code

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = FakeHttpErrorResponse(self.status_code)
            raise requests.HTTPError(
                f"{self.status_code} Client Error",
                response=response,
            )
        return None

    def json(self) -> dict:
        return self.json_data

    def iter_content(self, chunk_size: int):
        if self.iter_error:
            raise self.iter_error
        yield self.content


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("Chamada HTTP inesperada.")
        return self.responses.pop(0)


def test_download_skips_when_database_already_exists(tmp_path, capsys) -> None:
    destination = tmp_path / "inmet_historical.duckdb"
    destination.write_bytes(b"existing")
    session = FakeSession([])

    result = download_inmet_database(
        url="https://example.test/inmet.duckdb",
        destination=destination,
        expected_sha256="",
        session=session,
    )

    assert result == destination
    assert destination.read_bytes() == b"existing"
    assert session.calls == []
    assert "Base historica ja existe. Use --force" in capsys.readouterr().out


def test_download_force_replaces_existing_database(tmp_path) -> None:
    destination = tmp_path / "inmet_historical.duckdb"
    destination.write_bytes(b"old")
    session = FakeSession([FakeResponse(content=b"new")])

    download_inmet_database(
        url="https://example.test/inmet.duckdb",
        destination=destination,
        force=True,
        expected_sha256="",
        session=session,
    )

    assert destination.read_bytes() == b"new"
    assert len(session.calls) == 1


def test_download_creates_processed_directory(tmp_path) -> None:
    destination = tmp_path / "data" / "processed" / "inmet_historical.duckdb"
    session = FakeSession([FakeResponse(content=b"duckdb")])

    download_inmet_database(
        url="https://example.test/inmet.duckdb",
        destination=destination,
        expected_sha256="",
        session=session,
    )

    assert destination.exists()
    assert destination.read_bytes() == b"duckdb"


def test_find_asset_by_name_selects_exact_asset() -> None:
    release = {
        "assets": [
            {"name": "other.duckdb", "browser_download_url": "https://x.test/a"},
            {
                "name": "inmet_historical.duckdb",
                "browser_download_url": "https://x.test/db",
            },
        ]
    }

    asset = find_asset_by_name(release, "inmet_historical.duckdb")

    assert asset["browser_download_url"] == "https://x.test/db"


def test_download_uses_release_asset_when_direct_url_is_empty(tmp_path) -> None:
    release_response = FakeResponse(
        json_data={
            "assets": [
                {
                    "name": "inmet_historical.duckdb",
                    "browser_download_url": "https://download.test/db",
                    "url": "https://api.github.test/assets/1",
                }
            ]
        }
    )
    session = FakeSession([release_response, FakeResponse(content=b"db")])
    destination = tmp_path / "inmet.duckdb"

    download_inmet_database(
        url="",
        destination=destination,
        expected_sha256="",
        release_repo="owner/repo",
        release_tag="tag",
        asset_name="inmet_historical.duckdb",
        session=session,
    )

    assert session.calls[0]["url"].endswith("/repos/owner/repo/releases/tags/tag")
    assert session.calls[1]["url"] == "https://api.github.test/assets/1"
    assert session.calls[1]["headers"]["Accept"] == "application/octet-stream"
    assert destination.read_bytes() == b"db"


def test_download_uses_browser_download_url_when_asset_api_url_is_missing(
    tmp_path,
) -> None:
    release_response = FakeResponse(
        json_data={
            "assets": [
                {
                    "name": "inmet_historical.duckdb",
                    "browser_download_url": "https://download.test/db",
                }
            ]
        }
    )
    session = FakeSession([release_response, FakeResponse(content=b"db")])

    download_inmet_database(
        url="",
        destination=tmp_path / "inmet.duckdb",
        expected_sha256="",
        release_repo="owner/repo",
        release_tag="tag",
        asset_name="inmet_historical.duckdb",
        github_token="secret-token",
        session=session,
    )

    assert session.calls[1]["url"] == "https://download.test/db"
    assert session.calls[1]["headers"]["Authorization"] == "Bearer secret-token"


def test_download_release_api_404_returns_friendly_error(tmp_path) -> None:
    session = FakeSession([FakeResponse(status_code=404)])

    with pytest.raises(
        InmetDatabaseDownloadError,
        match="Release nao encontrada ou sem permissao",
    ):
        download_inmet_database(
            url="",
            destination=tmp_path / "inmet.duckdb",
            expected_sha256="",
            release_repo="owner/repo",
            release_tag="tag",
            asset_name="inmet_historical.duckdb",
            session=session,
        )


@pytest.mark.parametrize("status_code", [401, 403])
def test_download_release_api_auth_error_returns_friendly_error(
    tmp_path,
    status_code,
) -> None:
    session = FakeSession([FakeResponse(status_code=status_code)])

    with pytest.raises(
        InmetDatabaseDownloadError,
        match="Token ausente, invalido ou sem permissao",
    ):
        download_inmet_database(
            url="",
            destination=tmp_path / "inmet.duckdb",
            expected_sha256="",
            release_repo="owner/repo",
            release_tag="tag",
            asset_name="inmet_historical.duckdb",
            session=session,
        )


def test_download_asset_api_403_removes_tmp_and_returns_friendly_error(
    tmp_path,
) -> None:
    release_response = FakeResponse(
        json_data={
            "assets": [
                {
                    "name": "inmet_historical.duckdb",
                    "url": "https://api.github.test/assets/1",
                }
            ]
        }
    )
    session = FakeSession([release_response, FakeResponse(status_code=403)])
    destination = tmp_path / "inmet.duckdb"
    temporary_path = Path(f"{destination}.tmp")

    with pytest.raises(
        InmetDatabaseDownloadError,
        match="Token ausente, invalido ou sem permissao",
    ):
        download_inmet_database(
            url="",
            destination=destination,
            expected_sha256="",
            release_repo="owner/repo",
            release_tag="tag",
            asset_name="inmet_historical.duckdb",
            session=session,
        )

    assert not destination.exists()
    assert not temporary_path.exists()


def test_download_raises_friendly_error_when_asset_is_missing(tmp_path) -> None:
    session = FakeSession([FakeResponse(json_data={"assets": []})])

    with pytest.raises(ValueError, match="Asset 'missing.duckdb' nao encontrado"):
        download_inmet_database(
            url="",
            destination=tmp_path / "inmet.duckdb",
            expected_sha256="",
            release_repo="owner/repo",
            release_tag="tag",
            asset_name="missing.duckdb",
            session=session,
        )


def test_download_uses_token_header_without_printing_token(tmp_path, capsys) -> None:
    session = FakeSession([FakeResponse(content=b"db")])

    download_inmet_database(
        url="https://example.test/inmet.duckdb",
        destination=tmp_path / "inmet.duckdb",
        expected_sha256="",
        github_token="secret-token",
        session=session,
    )

    headers = session.calls[0]["headers"]
    assert headers["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in capsys.readouterr().out


def test_download_sends_token_to_release_api_without_printing_token(
    tmp_path,
    capsys,
) -> None:
    release_response = FakeResponse(
        json_data={
            "assets": [
                {
                    "name": "inmet_historical.duckdb",
                    "url": "https://api.github.test/assets/1",
                }
            ]
        }
    )
    session = FakeSession([release_response, FakeResponse(content=b"db")])

    download_inmet_database(
        url="",
        destination=tmp_path / "inmet.duckdb",
        expected_sha256="",
        release_repo="owner/repo",
        release_tag="tag",
        asset_name="inmet_historical.duckdb",
        github_token="secret-token",
        session=session,
    )

    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret-token"
    assert session.calls[1]["headers"]["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in capsys.readouterr().out


def test_build_github_headers_omits_authorization_without_token() -> None:
    headers = build_github_headers("")

    assert "Authorization" not in headers


def test_validate_sha256_accepts_small_file(tmp_path) -> None:
    destination = tmp_path / "small.duckdb"
    destination.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()

    validate_sha256(destination, expected)


def test_download_removes_tmp_file_when_sha256_is_invalid(tmp_path) -> None:
    destination = tmp_path / "inmet.duckdb"
    tmp_path_file = Path(f"{destination}.tmp")
    session = FakeSession([FakeResponse(content=b"abc")])

    with pytest.raises(ValueError, match="SHA256 invalido"):
        download_inmet_database(
            url="https://example.test/inmet.duckdb",
            destination=destination,
            expected_sha256="invalid",
            session=session,
        )

    assert not destination.exists()
    assert not tmp_path_file.exists()


def test_download_removes_tmp_file_when_stream_fails(tmp_path) -> None:
    destination = tmp_path / "inmet.duckdb"
    tmp_path_file = Path(f"{destination}.tmp")
    session = FakeSession(
        [FakeResponse(content=b"db", iter_error=RuntimeError("falha"))]
    )

    with pytest.raises(RuntimeError, match="falha"):
        download_inmet_database(
            url="https://example.test/inmet.duckdb",
            destination=destination,
            expected_sha256="",
            session=session,
        )

    assert not destination.exists()
    assert not tmp_path_file.exists()
