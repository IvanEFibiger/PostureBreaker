from __future__ import annotations

import platform
import sys
from pathlib import Path

from . import __version__
from .config import Config


def build_report(
    config: Config,
    data_dir: Path,
    *,
    cameras: list[int] | None = None,
    model_path: Path | None = None,
    log_path: Path | None = None,
) -> dict[str, object]:
    model = model_path if model_path is not None else data_dir / config.model_path
    model_exists = Path(model).exists()
    return {
        "app_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "data_dir": str(data_dir),
        "database": str(data_dir / config.database_path),
        "model_path": str(model),
        "model_present": model_exists,
        "model_size_bytes": Path(model).stat().st_size if model_exists else 0,
        "cameras": list(cameras or []),
        "camera_index": config.camera_index,
        "min_visibility": config.min_visibility,
        "break_interval_minutes": config.break_interval_minutes,
        "snooze_minutes": config.snooze_minutes,
        "log_path": str(log_path) if log_path else "",
    }


def format_report(report: dict[str, object]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in report.items())
