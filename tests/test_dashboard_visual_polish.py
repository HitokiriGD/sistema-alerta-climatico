from pathlib import Path
import inspect
import tomllib

from app import streamlit_app
from app.streamlit_app import DASHBOARD_SUBTITLE
from app.streamlit_app import PAGE_ICON
from app.streamlit_app import PAGE_LAYOUT
from app.streamlit_app import PAGE_TITLE


def test_streamlit_theme_is_light() -> None:
    config = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf-8"))

    assert config["theme"]["base"] == "light"
    assert config["theme"]["backgroundColor"] == "#f8fafc"
    assert config["theme"]["secondaryBackgroundColor"] == "#ffffff"
    assert config["theme"]["textColor"] == "#0f172a"


def test_page_identity_constants_are_configured() -> None:
    main_source = inspect.getsource(streamlit_app.main)

    assert PAGE_TITLE == "Alerta Climático | TCC"
    assert PAGE_ICON.strip()
    assert PAGE_LAYOUT == "wide"
    assert "page_title=PAGE_TITLE" in main_source
    assert "page_icon=PAGE_ICON" in main_source
    assert "layout=PAGE_LAYOUT" in main_source


def test_banner_contains_refined_subtitle() -> None:
    header_source = inspect.getsource(streamlit_app.show_dashboard_header)

    assert DASHBOARD_SUBTITLE == (
        "Consulta dados meteorológicos atuais, compara com o histórico do INMET "
        "e usa Machine Learning supervisionado para apoiar a análise de risco "
        "climático."
    )
    assert "DASHBOARD_SUBTITLE" in header_source


def test_custom_css_does_not_force_dark_global_theme() -> None:
    style_source = inspect.getsource(streamlit_app.apply_dashboard_style)

    assert "#111827" not in style_source
    assert "#0f172a 100%" not in style_source
    app_rule = style_source.split(".stApp {", 1)[1].split("}", 1)[0]
    assert "background: #f8fafc" in app_rule
    assert "background: #ffffff" not in app_rule
    assert "background: #ffffff" in style_source
