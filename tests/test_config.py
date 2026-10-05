from __future__ import annotations

import unittest
from pathlib import Path

from posture_guard.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "posture_break_guard.config.json"


class ShippedConfigTests(unittest.TestCase):
    def test_shipped_config_loads_into_dataclass(self) -> None:
        self.assertTrue(CONFIG_PATH.exists())
        config = load_config(CONFIG_PATH)
        self.assertIsInstance(config, Config)
        self.assertEqual(config.posture_min_bad_metrics, 2)

    def test_falls_back_to_defaults_when_file_is_missing(self) -> None:
        config = load_config(REPO_ROOT / "missing.config.json")
        self.assertEqual(config.posture_min_bad_metrics, 2)


if __name__ == "__main__":
    unittest.main()
