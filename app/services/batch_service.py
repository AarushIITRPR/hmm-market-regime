"""Batch prediction orchestration using Ray when available."""

from __future__ import annotations

import logging

from fastapi import status

from app.ray_tasks.regime_tasks import ticker_analysis_task
from app.schemas.predict import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchTickerResult,
    TickerPredictionRequest,
    TickerPredictionResponse,
)
from app.services import ray_service
from app.services.regime_service import RegimeServiceError, predict_ticker_regimes

logger = logging.getLogger(__name__)


def batch_predict_tickers(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Run ticker analyses in parallel when Ray is available."""
    ticker_requests = [_build_ticker_request(request, ticker) for ticker in request.tickers]
    logger.info("Batch prediction requested for %d tickers.", len(ticker_requests))

    if ray_service.is_ray_ready() and ticker_analysis_task is not None:
        logger.info("Dispatching batch prediction to Ray workers.")
        results = _run_with_ray(ticker_requests)
    else:
        logger.warning("Ray is unavailable; running batch prediction sequentially.")
        results = [_run_single_ticker(ticker_request) for ticker_request in ticker_requests]

    succeeded = sum(1 for result in results if result.success)
    failed = len(results) - succeeded
    logger.info("Batch prediction completed: %d succeeded, %d failed.", succeeded, failed)

    return BatchPredictionResponse(results=results, succeeded=succeeded, failed=failed)


def _run_with_ray(ticker_requests: list[TickerPredictionRequest]) -> list[BatchTickerResult]:
    futures = []
    for ticker_request in ticker_requests:
        logger.info("Submitting Ray ticker task for %s.", ticker_request.ticker)
        futures.append((ticker_request.ticker.upper(), ticker_analysis_task.remote(ticker_request.model_dump(mode="json"))))

    results: list[BatchTickerResult] = []
    for ticker, future in futures:
        try:
            payload = ray_service.ray.get(future)
            prediction = TickerPredictionResponse.model_validate(payload)
            results.append(BatchTickerResult(ticker=ticker, success=True, data=prediction))
        except Exception as exc:
            logger.exception("Ray ticker task failed for %s.", ticker)
            results.append(_failed_result(ticker, exc))

    return results


def _run_single_ticker(ticker_request: TickerPredictionRequest) -> BatchTickerResult:
    ticker = ticker_request.ticker.upper()
    try:
        logger.info("Running ticker analysis for %s.", ticker)
        prediction = predict_ticker_regimes(ticker_request)
        return BatchTickerResult(ticker=ticker, success=True, data=prediction)
    except Exception as exc:
        logger.exception("Ticker analysis failed for %s.", ticker)
        return _failed_result(ticker, exc)


def _build_ticker_request(request: BatchPredictionRequest, ticker: str) -> TickerPredictionRequest:
    return TickerPredictionRequest(
        ticker=ticker,
        start_date=request.start_date,
        end_date=request.end_date,
        n_states=request.n_states,
        n_iter=request.n_iter,
        tol=request.tol,
        random_state=request.random_state,
    )


def _failed_result(ticker: str, exc: Exception) -> BatchTickerResult:
    status_code = exc.status_code if isinstance(exc, RegimeServiceError) else status.HTTP_500_INTERNAL_SERVER_ERROR
    message = exc.message if isinstance(exc, RegimeServiceError) else str(exc)
    return BatchTickerResult(
        ticker=ticker.upper(),
        success=False,
        error=message or "Ticker analysis failed.",
        status_code=status_code,
    )
