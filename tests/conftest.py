# tests/conftest.py
import pytest

@pytest.fixture(autouse=True)
def no_requests(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: pytest.fail(
        "Unexpected call to requests.get — mock it in the test!"
    ))

