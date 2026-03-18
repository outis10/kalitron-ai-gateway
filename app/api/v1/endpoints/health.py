import time

from fastapi import APIRouter

from app.core.config import settings
from app.models.responses import HealthResponse

router = APIRouter()

_START_TIME = time.monotonic()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Return service status, version, environment, and uptime."""
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
    )
