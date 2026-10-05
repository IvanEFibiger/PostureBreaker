from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "PostureBreaker"


def user_data_dir(app_name: str = APP_NAME, env: dict[str, str] | None = None) -> Path:
    """Resolve a writable per-user data directory without third-party deps."""
    source = os.environ if env is None else env
    base = source.get("LOCALAPPDATA") or source.get("APPDATA") or source.get("XDG_DATA_HOME")
    if base:
        return Path(base) / app_name
    return Path.home() / f".{app_name.lower()}"


def resolve_dirs() -> tuple[Path, Path]:
    """Return (data_dir, resource_dir).

    Packaged builds keep writable data in the user profile (so an install under
    Program Files still works) and read bundled resources next to the executable.
    Running from source keeps both pointing at the repo for convenience.
    """
    if getattr(sys, "frozen", False):
        data_dir = user_data_dir()
        resource_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        source_dir = Path(__file__).resolve().parent.parent
        data_dir = source_dir
        resource_dir = source_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, resource_dir
