import pytest
from pytest_socket import disable_socket


def pytest_runtest_setup():
    """Enforce a strict network ban for all unit tests."""
    disable_socket()


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Inject mandatory test environment variables."""
    monkeypatch.setenv("JUDGE_MODEL", "test-model")
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-key")
