"""Bridge between ragval's async-native core and synchronous callers.

``run_sync`` must work in three environments:
  * a plain script (no running loop) -> ``asyncio.run``
  * a Jupyter notebook (loop already running) -> ``nest_asyncio`` if available,
    otherwise a worker thread
  * inside FastAPI / an existing event loop -> a worker thread with its own loop
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Awaitable, Coroutine, List, TypeVar

T = TypeVar("T")


def _run_in_new_thread(coro: Coroutine) -> Any:
    """Run ``coro`` to completion on a fresh event loop in a worker thread."""

    def runner() -> Any:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(runner).result()


def run_sync(coro: Coroutine) -> Any:
    """Execute an awaitable from synchronous code, whatever the loop situation."""
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is None:
        # No loop running: the simple, fast path.
        return asyncio.run(coro)

    # A loop is already running (Jupyter, FastAPI, nested call).
    try:
        import nest_asyncio  # type: ignore

        nest_asyncio.apply()
        return running_loop.run_until_complete(coro)
    except ImportError:
        return _run_in_new_thread(coro)


async def gather_with_concurrency(n: int, *coros: Awaitable[T]) -> List[T]:
    """Like ``asyncio.gather`` but limits in-flight coroutines to ``n``.

    Results are returned in the same order as the input coroutines.
    """
    if n <= 0:
        n = 1
    semaphore = asyncio.Semaphore(n)

    async def _bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_bounded(c) for c in coros))
