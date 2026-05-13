"""Health check route."""

from fastapi import APIRouter


router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple liveness response."""
    return {"status": "ok"}
