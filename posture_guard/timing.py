from __future__ import annotations


def is_session_gap(dt_seconds: float, max_gap_seconds: float) -> bool:
    """True when the elapsed time between frames is too large to be a normal frame.

    A gap this big usually means the machine was suspended, locked, or the
    process stalled. Callers should drop the delta instead of adding it as work
    or posture time.
    """
    return dt_seconds > max_gap_seconds
