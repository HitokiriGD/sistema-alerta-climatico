from pathlib import Path


REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"


def _requirements() -> set[str]:
    return {
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_scikit_learn_matches_published_random_forest() -> None:
    assert "scikit-learn==1.9.0" in _requirements()
    assert "scikit-learn" not in _requirements()


def test_xgboost_uses_official_cpu_distribution() -> None:
    assert "xgboost-cpu==3.3.0" in _requirements()
    assert "xgboost" not in _requirements()
