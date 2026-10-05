from __future__ import annotations

import unittest

from posture_guard.autostart import Autostart


class _FakeBackend:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get_value(self, name: str) -> str | None:
        return self.values.get(name)

    def set_value(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete_value(self, name: str) -> None:
        self.values.pop(name, None)


class AutostartTests(unittest.TestCase):
    def _autostart(self, backend) -> Autostart:
        return Autostart("PostureBreaker", r"C:\app\PostureBreaker.exe", backend=backend)

    def test_enable_then_is_enabled(self) -> None:
        backend = _FakeBackend()
        autostart = self._autostart(backend)
        self.assertFalse(autostart.is_enabled())
        self.assertTrue(autostart.enable())
        self.assertTrue(autostart.is_enabled())
        self.assertEqual(backend.values["PostureBreaker"], r"C:\app\PostureBreaker.exe")

    def test_disable_removes_entry(self) -> None:
        backend = _FakeBackend()
        autostart = self._autostart(backend)
        autostart.enable()
        self.assertTrue(autostart.disable())
        self.assertFalse(autostart.is_enabled())

    def test_toggle_flips_state(self) -> None:
        backend = _FakeBackend()
        autostart = self._autostart(backend)
        self.assertTrue(autostart.toggle())
        self.assertTrue(autostart.is_enabled())
        self.assertFalse(autostart.toggle())
        self.assertFalse(autostart.is_enabled())

    def test_unsupported_backend_is_noop(self) -> None:
        autostart = Autostart("x", "y", backend=None)
        if autostart.supported():
            self.skipTest("platform provides a registry backend")
        self.assertFalse(autostart.is_enabled())
        self.assertFalse(autostart.enable())
        self.assertFalse(autostart.disable())


if __name__ == "__main__":
    unittest.main()
