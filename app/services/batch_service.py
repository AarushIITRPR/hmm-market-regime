from __future__ import annotations

import logging

from fastapi import status

from app.schemas.predict import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchTickerResult,
    TickerPredictionRequest,
)
from app.services.regime_service import RegimeServiceError, predict_ticker_regimes

logger = logging.getLogger(__name__)


def batch_predict_tickers(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Run ticker analyses one by one."""
    ticker_requests = [_build_ticker_request(request, ticker) for ticker in request.tickers]
    logger.info("Batch prediction requested for %d tickers.", len(ticker_requests))

    results = [_run_single_ticker(ticker_request) for ticker_request in ticker_requests]
    succeeded = sum(1 for result in results if result.success)
    failed = len(results) - succeeded
    logger.info("Batch prediction completed: %d succeeded, %d failed.", succeeded, failed)

    return BatchPredictionResponse(results=results, succeeded=succeeded, failed=failed)


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
