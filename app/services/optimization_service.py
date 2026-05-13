"""Optimization service orchestration with Ray fallback."""

from __future__ import annotations

import logging

from app.optimization.portfolio_optimizer import solve_stock_allocation
from app.ray_tasks.optimization_tasks import optimization_task
from app.schemas.optimization import (
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResult,
)
from app.services import ray_service

logger = logging.getLogger(__name__)


def optimize_allocations(request: OptimizationRequest) -> OptimizationResponse:
    """Run simple OR-Tools optimization scenarios, in parallel when Ray is ready."""
    logger.info("Optimization requested for %d scenario(s).", len(request.scenarios))
    if ray_service.is_ray_ready() and optimization_task is not None:
        logger.info("Dispatching optimization scenarios to Ray workers.")
        results = _run_with_ray(request)
    else:
        logger.warning("Ray is unavailable; running optimization scenarios sequentially.")
        results = [_run_single_scenario(request, scenario) for scenario in request.scenarios]

    succeeded = sum(1 for result in results if result.success)
    failed = len(results) - succeeded
    logger.info("Optimization completed: %d succeeded, %d failed.", succeeded, failed)
    return OptimizationResponse(results=results, succeeded=succeeded, failed=failed)


def _run_with_ray(request: OptimizationRequest) -> list[OptimizationResult]:
    assets_payload = [asset.model_dump() for asset in request.assets]
    futures = []
    for scenario in request.scenarios:
        logger.info("Submitting Ray optimization task for scenario %s.", scenario.name)
        futures.append((scenario.name, optimization_task.remote(assets_payload, scenario.model_dump())))

    results = []
    for scenario_name, future in futures:
        try:
            payload = ray_service.ray.get(future)
            results.append(OptimizationResult.model_validate(payload))
        except Exception as exc:
            logger.exception("Optimization task failed for scenario %s.", scenario_name)
            results.append(_failed_result(scenario_name, exc))
    return results


def _run_single_scenario(request: OptimizationRequest, scenario) -> OptimizationResult:
    try:
        logger.info("Running optimization scenario %s.", scenario.name)
        payload = solve_stock_allocation(request.assets, scenario)
        return OptimizationResult.model_validate(payload)
    except Exception as exc:
        logger.exception("Optimization failed for scenario %s.", scenario.name)
        return _failed_result(scenario.name, exc)


def _failed_result(scenario_name: str, exc: Exception) -> OptimizationResult:
    return OptimizationResult(
        scenario=scenario_name,
        status="FAILED",
        objective_value=0.0,
        total_cost=0.0,
        total_risk=0.0,
        allocations=[],
        success=False,
        error=str(exc) or "Optimization failed.",
    )
