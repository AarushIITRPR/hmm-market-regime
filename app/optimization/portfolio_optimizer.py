"""Simple OR-Tools optimization for regime-aware stock allocation."""

from __future__ import annotations

from ortools.linear_solver import pywraplp

from app.schemas.optimization import OptimizationAsset, OptimizationScenario


def solve_stock_allocation(
    assets: list[OptimizationAsset],
    scenario: OptimizationScenario,
) -> dict:
    """
    Maximize expected profit subject to budget, risk, and position limits.

    This is intentionally simple and educational: each decision variable is the
    integer number of units to allocate to one asset.
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is unavailable.")

    units = {
        asset.ticker: solver.IntVar(0, scenario.max_units_per_asset, f"units_{asset.ticker}")
        for asset in assets
    }

    solver.Add(
        sum(asset.price * units[asset.ticker] for asset in assets) <= scenario.budget
    )
    solver.Add(
        sum(asset.risk_score * units[asset.ticker] for asset in assets) <= scenario.max_risk
    )
    solver.Maximize(
        sum(asset.expected_profit * units[asset.ticker] for asset in assets)
    )

    status = solver.Solve()
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        return {
            "scenario": scenario.name,
            "status": _status_name(status),
            "objective_value": 0.0,
            "total_cost": 0.0,
            "total_risk": 0.0,
            "allocations": [],
        }

    allocations = []
    total_cost = 0.0
    total_risk = 0.0
    for asset in assets:
        quantity = int(round(units[asset.ticker].solution_value()))
        cost = quantity * asset.price
        risk = quantity * asset.risk_score
        total_cost += cost
        total_risk += risk
        allocations.append(
            {
                "ticker": asset.ticker,
                "units": quantity,
                "price": asset.price,
                "cost": cost,
                "risk": risk,
                "expected_profit": quantity * asset.expected_profit,
                "regime_label": asset.regime_label,
            }
        )

    return {
        "scenario": scenario.name,
        "status": _status_name(status),
        "objective_value": solver.Objective().Value(),
        "total_cost": total_cost,
        "total_risk": total_risk,
        "allocations": allocations,
    }


def _status_name(status: int) -> str:
    statuses = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    return statuses.get(status, "UNKNOWN")
