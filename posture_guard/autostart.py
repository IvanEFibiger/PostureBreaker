from __future__ import annotations

import sys

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


class Autostart:
    """Manage a Windows per-user autostart entry (HKCU ...\\Run)."""

    def __init__(self, app_name: str, command: str, backend: object | None = None) -> None:
        self.app_name = app_name
        self.command = command
        self._backend = backend if backend is not None else _default_backend()

    def supported(self) -> bool:
        return self._backend is not None

    def is_enabled(self) -> bool:
        if self._backend is None:
            return False
        return self._backend.get_value(self.app_name) is not None

    def enable(self) -> bool:
        if self._backend is None:
            return False
        self._backend.set_value(self.app_name, self.command)
        return True

    def disable(self) -> bool:
        if self._backend is None:
            return False
        self._backend.delete_value(self.app_name)
        return True

    def toggle(self) -> bool:
        if self.is_enabled():
            self.disable()
            return False
        self.enable()
        return True


class _WinRegBackend:
    def get_value(self, name: str) -> str | None:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
                value, _ = winreg.QueryValueEx(key, name)
                return str(value)
        except FileNotFoundError:
            return None

    def set_value(self, name: str, value: str) -> None:
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)

    def delete_value(self, name: str) -> None:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, name)
        except FileNotFoundError:
            pass


def _default_backend() -> object | None:
    if sys.platform.startswith("win"):
        return _WinRegBackend()
    return None
