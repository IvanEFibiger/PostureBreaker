from __future__ import annotations

import unittest
from pathlib import Path

from posture_guard import paths


class UserDataDirTests(unittest.TestCase):
    def test_prefers_localappdata(self) -> None:
        data_dir = paths.user_data_dir("MyApp", env={"LOCALAPPDATA": r"C:\Users\x\AppData\Local"})
        self.assertEqual(data_dir, Path(r"C:\Users\x\AppData\Local") / "MyApp")

    def test_falls_back_to_home_when_no_env(self) -> None:
        data_dir = paths.user_data_dir("MyApp", env={})
        self.assertEqual(data_dir, Path.home() / ".myapp")


if __name__ == "__main__":
    unittest.main()
