"""Ray lifecycle management for distributed backend tasks."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

try:
    import ray
except ImportError:  # pragma: no cover - exercised when Ray is intentionally absent.
    ray = None


def initialize_ray() -> bool:
    """Initialize Ray with a graceful fallback when it is unavailable."""
    if ray is None:
        logger.warning("Ray is not installed; distributed tasks will run sequentially.")
        return False

    if ray.is_initialized():
        logger.info("Ray is already initialized.")
        return True

    try:
        address = os.getenv("RAY_ADDRESS")
        ray.init(
            address=address,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=True,
        )
        logger.info("Ray initialized successfully%s.", f" at {address}" if address else "")
        return True
    except Exception:
        logger.exception("Ray initialization failed; falling back to sequential execution.")
        return False


def shutdown_ray() -> None:
    """Shutdown Ray if this process initialized it."""
    if ray is not None and ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete.")


def is_ray_ready() -> bool:
    """Return whether Ray can currently execute remote tasks."""
    return ray is not None and ray.is_initialized()
