from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from posture_guard.config import Config
from posture_guard.diagnostics import build_report, format_report


class DiagnosticsTests(unittest.TestCase):
    def test_report_has_expected_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(Config(), Path(tmp), cameras=[0, 1])
            self.assertEqual(report["cameras"], [0, 1])
            self.assertEqual(report["app_version"], "0.1.0")
            self.assertIn("python", report)
            self.assertIn("platform", report)

    def test_missing_model_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(Config(), Path(tmp), model_path=Path(tmp) / "none.task")
            self.assertFalse(report["model_present"])
            self.assertEqual(report["model_size_bytes"], 0)

    def test_existing_model_size_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "pose.task"
            model.write_bytes(b"x" * 128)
            report = build_report(Config(), Path(tmp), model_path=model)
            self.assertTrue(report["model_present"])
            self.assertEqual(report["model_size_bytes"], 128)

    def test_format_report_mentions_keys(self) -> None:
        text = format_report({"app_version": "0.1.0", "python": "3.12"})
        self.assertIn("app_version: 0.1.0", text)
        self.assertIn("python: 3.12", text)


if __name__ == "__main__":
    unittest.main()
