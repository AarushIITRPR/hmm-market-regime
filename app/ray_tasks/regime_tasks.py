"""Ray remote tasks for HMM regime prediction workloads."""

from __future__ import annotations

import logging

from app.schemas.predict import PredictionRequest, TickerPredictionRequest
from app.services import ray_service
from app.services.regime_service import predict_regimes, predict_ticker_regimes

logger = logging.getLogger(__name__)


def _market_regime_prediction(payload: dict) -> dict:
    request = PredictionRequest.model_validate(payload)
    logger.info("Ray task started: market regime prediction.")
    response = predict_regimes(request)
    logger.info("Ray task completed: market regime prediction.")
    return response.model_dump()


def _ticker_analysis(payload: dict) -> dict:
    request = TickerPredictionRequest.model_validate(payload)
    logger.info("Ray task started: ticker analysis for %s.", request.ticker)
    response = predict_ticker_regimes(request)
    logger.info("Ray task completed: ticker analysis for %s.", request.ticker)
    return response.model_dump()


def _batch_stock_processing(payloads: list[dict]) -> list[dict]:
    logger.info("Ray task started: batch stock processing for %d tickers.", len(payloads))
    results = [_ticker_analysis(payload) for payload in payloads]
    logger.info("Ray task completed: batch stock processing.")
    return results


if ray_service.ray is not None:
    market_regime_prediction_task = ray_service.ray.remote(_market_regime_prediction)
    ticker_analysis_task = ray_service.ray.remote(_ticker_analysis)
    batch_stock_processing_task = ray_service.ray.remote(_batch_stock_processing)
else:
    market_regime_prediction_task = None
    ticker_analysis_task = None
    batch_stock_processing_task = None
