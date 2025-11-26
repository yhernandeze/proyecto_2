# tests/unit/streamlit/test_streamlit_app.py

def test_streamlit_module_imports():
    """Smoke test: importar el módulo de Streamlit no debe lanzar excepciones."""
    import streamlit_ui.app  # noqa: F401


def test_streamlit_feature_names_contract():
    """El contrato de columnas de features debe coincidir con lo esperado."""
    from streamlit_ui.app import get_feature_names

    cols = get_feature_names()
    assert cols == [
        "bed",
        "bath",
        "acre_lot",
        "house_size",
        "zip_code",
        "brokered_by",
        "street",
    ]
