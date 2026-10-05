from __future__ import annotations

import sys

ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    """Cross-process single-instance guard using a Windows named mutex.

    On non-Windows platforms it always allows startup (returns True).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._handle = None
        self._kernel32 = None

    def acquire(self) -> bool:
        if not sys.platform.startswith("win"):
            return True

        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        handle = kernel32.CreateMutexW(None, False, self._name)
        if not handle:
            return True
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        self._kernel32 = kernel32
        return True

    def release(self) -> None:
        if self._handle is not None and self._kernel32 is not None:
            self._kernel32.CloseHandle(self._handle)
        self._handle = None
        self._kernel32 = None
