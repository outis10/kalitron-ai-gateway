import base64

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

# Override API keys so tests can authenticate without real keys
settings.API_KEYS = "test-key-1,test-key-2"
settings.ANTHROPIC_API_KEY = "sk-test-placeholder"
settings.OPENAI_API_KEY = "sk-test-placeholder"


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Synchronous TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(scope="session")
def api_headers() -> dict[str, str]:
    """Valid API key headers for authenticated requests."""
    return {"X-API-Key": "test-key-1"}


@pytest.fixture(scope="session")
def dummy_png() -> bytes:
    """Minimal 1×1 white PNG image for upload tests."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhf"
        "DwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )


@pytest.fixture(scope="session")
def large_dummy_png(dummy_png: bytes) -> bytes:
    """PNG payload larger than the configured MAX_FILE_SIZE_MB for 413 checks."""
    return dummy_png * ((settings.MAX_FILE_SIZE_MB * 1024 * 1024 // len(dummy_png)) + 1)
