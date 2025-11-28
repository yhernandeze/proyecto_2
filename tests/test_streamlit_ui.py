def test_streamlit_module_imports():
    """Smoke test: importar el módulo de Streamlit no debe lanzar excepciones."""
    import streamlit_ui.app  # noqa: F401


def test_streamlit_feature_names_contract():
    """El contrato de columnas de features debe coincidir con lo esperado."""
    from streamlit_ui.app import get_feature_names
    from shared.feature_contract import FEATURE_COLUMNS

    assert get_feature_names() == FEATURE_COLUMNS
