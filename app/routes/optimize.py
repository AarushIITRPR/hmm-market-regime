from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator


router = APIRouter(tags=["allocation"])


class OptimizationAsset(BaseModel):
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
    name: str = Field(default="Base Case", min_length=1, max_length=80)
    budget: float = Field(..., gt=0)
    max_risk: float = Field(..., ge=0)
    max_units_per_asset: int = Field(default=10, ge=0, le=10_000)


class OptimizationRequest(BaseModel):
    assets: list[OptimizationAsset] = Field(..., min_length=1, max_length=100)
    scenarios: list[OptimizationScenario] = Field(..., min_length=1, max_length=25)


class AllocationResult(BaseModel):
    ticker: str
    units: int
    price: float
    cost: float
    risk: float
    expected_profit: float
    regime_label: str


class OptimizationResult(BaseModel):
    scenario: str
    status: str
    objective_value: float
    total_cost: float
    total_risk: float
    allocations: list[AllocationResult]
    success: bool = True
    error: str | None = None


class OptimizationResponse(BaseModel):
    results: list[OptimizationResult]
    succeeded: int
    failed: int


@router.post("/optimize", response_model=OptimizationResponse)
def optimize(request: OptimizationRequest) -> OptimizationResponse:
    results = [_allocate_greedily(request.assets, scenario) for scenario in request.scenarios]
    return OptimizationResponse(
        results=results,
        succeeded=len(results),
        failed=0,
    )


def _allocate_greedily(
    assets: list[OptimizationAsset],
    scenario: OptimizationScenario,
) -> OptimizationResult:
    units_by_index = [0 for _ in assets]
    total_cost = 0.0
    total_risk = 0.0

    ranked_indices = sorted(
        range(len(assets)),
        key=lambda index: _asset_score(assets[index]),
        reverse=True,
    )

    improved = True
    while improved:
        improved = False
        for index in ranked_indices:
            asset = assets[index]
            if units_by_index[index] >= scenario.max_units_per_asset:
                continue
            if total_cost + asset.price > scenario.budget:
                continue
            if total_risk + asset.risk_score > scenario.max_risk:
                continue
            units_by_index[index] += 1
            total_cost += asset.price
            total_risk += asset.risk_score
            improved = True

    allocation_rows = []
    objective_value = 0.0
    for asset, units in zip(assets, units_by_index):
        cost = units * asset.price
        risk = units * asset.risk_score
        profit = units * asset.expected_profit
        objective_value += profit
        allocation_rows.append(
            AllocationResult(
                ticker=asset.ticker,
                units=units,
                price=asset.price,
                cost=cost,
                risk=risk,
                expected_profit=profit,
                regime_label=asset.regime_label,
            )
        )

    return OptimizationResult(
        scenario=scenario.name,
        status="FEASIBLE",
        objective_value=objective_value,
        total_cost=total_cost,
        total_risk=total_risk,
        allocations=allocation_rows,
    )


def _asset_score(asset: OptimizationAsset) -> float:
    cost_penalty = asset.price if asset.price > 0 else 1.0
    risk_penalty = asset.risk_score if asset.risk_score > 0 else 1.0
    return asset.expected_profit / (cost_penalty * risk_penalty)
