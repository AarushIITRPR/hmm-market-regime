"""Optimization routes."""

from fastapi import APIRouter

from app.schemas.optimization import OptimizationRequest, OptimizationResponse
from app.services.optimization_service import optimize_allocations


router = APIRouter(tags=["optimization"])


@router.post("/optimize", response_model=OptimizationResponse)
def optimize(request: OptimizationRequest) -> OptimizationResponse:
    """Run a simple OR-Tools stock allocation optimization."""
    return optimize_allocations(request)
