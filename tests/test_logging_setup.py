from __future__ import annotations

import logging
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path

from posture_guard.logging_setup import setup_logging


class LoggingSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        self._tmp.cleanup()

    def test_writes_a_log_file(self) -> None:
        log_path = setup_logging(self.tmp / "logs")
        logging.getLogger("test").info("hola-log")
        for handler in logging.getLogger().handlers:
            handler.flush()
        self.assertTrue(log_path.exists())
        self.assertIn("hola-log", log_path.read_text(encoding="utf-8"))

    def test_installs_a_single_rotating_handler(self) -> None:
        setup_logging(self.tmp)
        handlers = logging.getLogger().handlers
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0], RotatingFileHandler)

    def test_repeated_setup_does_not_stack_handlers(self) -> None:
        setup_logging(self.tmp)
        setup_logging(self.tmp)
        self.assertEqual(len(logging.getLogger().handlers), 1)


if __name__ == "__main__":
    unittest.main()
