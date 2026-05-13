"""Ray tasks for OR-Tools optimization workloads."""

from __future__ import annotations

import logging

from app.optimization.portfolio_optimizer import solve_stock_allocation
from app.schemas.optimization import OptimizationAsset, OptimizationScenario
from app.services import ray_service

logger = logging.getLogger(__name__)


def _optimization_task(assets_payload: list[dict], scenario_payload: dict) -> dict:
    scenario = OptimizationScenario.model_validate(scenario_payload)
    assets = [OptimizationAsset.model_validate(asset) for asset in assets_payload]
    logger.info("Ray task started: optimization scenario %s.", scenario.name)
    result = solve_stock_allocation(assets, scenario)
    logger.info("Ray task completed: optimization scenario %s.", scenario.name)
    return result


if ray_service.ray is not None:
    optimization_task = ray_service.ray.remote(_optimization_task)
else:
    optimization_task = None
