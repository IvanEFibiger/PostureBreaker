from __future__ import annotations

import time
from typing import Any, Callable

DEFAULT_BACKOFF = (1.0, 2.0, 5.0, 10.0)


class CameraError(RuntimeError):
    pass


def open_camera(
    factory: Callable[[int], Any],
    index: int,
    *,
    attempts: int = 3,
    backoff: tuple[float, ...] = DEFAULT_BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, float], None] | None = None,
) -> Any:
    """Open a camera, retrying with backoff. Raises CameraError if all fail."""
    for attempt in range(1, attempts + 1):
        capture = factory(index)
        if capture is not None and capture.isOpened():
            return capture
        if capture is not None:
            capture.release()
        if attempt < attempts:
            delay = backoff[min(attempt - 1, len(backoff) - 1)]
            if on_retry is not None:
                on_retry(attempt, delay)
            sleep(delay)
    raise CameraError(f"No pude abrir la camara {index} tras {attempts} intentos.")


def list_cameras(factory: Callable[[int], Any], max_index: int = 5) -> list[int]:
    """Probe camera indices and return the ones that open successfully."""
    available: list[int] = []
    for index in range(max_index + 1):
        capture = factory(index)
        try:
            if capture is not None and capture.isOpened():
                available.append(index)
        finally:
            if capture is not None:
                capture.release()
    return available
