# tests/conftest.py
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for package imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def no_requests(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: pytest.fail(
        "Unexpected call to requests.get — mock it in the test!"
    ))
