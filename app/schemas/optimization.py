"""Pydantic schemas for simple OR-Tools optimization."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class OptimizationAsset(BaseModel):
    """Input asset for the allocation model."""

    ticker: str = Field(..., min_length=1, max_length=20)
    price: float = Field(..., gt=0)
    expected_profit: float = Field(..., ge=0)
    risk_score: float = Field(..., ge=0)
    regime_label: str = "Unknown"

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, ticker: str) -> str:
        return ticker.strip().upper()


class OptimizationScenario(BaseModel):
    """Simple optimization constraints for one scenario."""

    name: str = Field(default="Base Case", min_length=1, max_length=80)
    budget: float = Field(..., gt=0)
    max_risk: float = Field(..., ge=0)
    max_units_per_asset: int = Field(default=10, ge=0, le=10_000)


class OptimizationRequest(BaseModel):
    """Optimization request that can run multiple scenarios in parallel."""

    assets: list[OptimizationAsset] = Field(..., min_length=1, max_length=100)
    scenarios: list[OptimizationScenario] = Field(..., min_length=1, max_length=25)


class AllocationResult(BaseModel):
    """Selected allocation for one asset."""

    ticker: str
    units: int
    price: float
    cost: float
    risk: float
    expected_profit: float
    regime_label: str


class OptimizationResult(BaseModel):
    """Optimization result for one scenario."""

    scenario: str
    status: str
    objective_value: float
    total_cost: float
    total_risk: float
    allocations: list[AllocationResult]
    success: bool = True
    error: str | None = None


class OptimizationResponse(BaseModel):
    """Response returned by the optimization API."""

    results: list[OptimizationResult]
    succeeded: int
    failed: int
