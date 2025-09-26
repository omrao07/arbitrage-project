# bus/python/utils/retry.py
from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Any, Awaitable, Callable, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Backoff calculation
# --------------------------------------------------------------------
def _calc_backoff(
    attempt: int,
    base: float,
    cap: float,
    jitter: bool = True,
) -> float:
    """Exponential backoff with optional jitter."""
    delay = base * (2 ** (attempt - 1))
    if jitter:
        delay = delay * random.uniform(0.8, 1.2)
    return min(delay, cap)


# --------------------------------------------------------------------
# Sync retry
# --------------------------------------------------------------------
def retry(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    tries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter: bool = True,
    logger_: Optional[logging.Logger] = None,
) -> Callable:
    """
    Decorator for retrying sync functions with exponential backoff.

    Example:
        @retry(tries=5, base_delay=1, max_delay=10)
        def fragile_call():
            ...
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    delay = _calc_backoff(attempt, base_delay, max_delay, jitter)
                    (logger_ or logger).warning(
                        f"[retry] {func.__name__} failed (attempt {attempt}/{tries}): {e}. Retrying in {delay:.2f}s"
                    )
                    if attempt < tries:
                        time.sleep(delay)
            raise last_exc # type: ignore

        return wrapper

    return decorator


# --------------------------------------------------------------------
# Async retry
# --------------------------------------------------------------------
def aretry(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    tries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter: bool = True,
    logger_: Optional[logging.Logger] = None,
) -> Callable:
    """
    Decorator for retrying async functions with exponential backoff.

    Example:
        @aretry(tries=5, base_delay=1, max_delay=10)
        async def fragile_async_call():
            ...
    """

    def decorator(func: Callable[..., Awaitable[Any]]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(1, tries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    delay = _calc_backoff(attempt, base_delay, max_delay, jitter)
                    (logger_ or logger).warning(
                        f"[aretry] {func.__name__} failed (attempt {attempt}/{tries}): {e}. Retrying in {delay:.2f}s"
                    )
                    if attempt < tries:
                        await asyncio.sleep(delay)
            raise last_exc # type: ignore

        return wrapper

    return decorator


# --------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------
if __name__ == "__main__":

    @retry(tries=3, base_delay=0.2, max_delay=1.5)
    def flaky():
        if random.random() < 0.7:
            raise RuntimeError("boom")
        return "ok"

    @aretry(tries=3, base_delay=0.2, max_delay=1.5)
    async def flaky_async():
        if random.random() < 0.7:
            raise RuntimeError("boom async")
        return "ok-async"

    print("Sync demo:")
    try:
        print(flaky())
    except Exception as e:
        print("Failed:", e)

    print("Async demo:")
    asyncio.run(flaky_async())