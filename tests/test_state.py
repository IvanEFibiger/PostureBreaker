from __future__ import annotations

import unittest

from posture_guard.state import SharedState


class ConsumeValueTests(unittest.TestCase):
    def test_consume_value_returns_then_resets_to_default(self) -> None:
        shared = SharedState()
        shared.update(pending_camera_index=3)
        self.assertEqual(shared.consume_value("pending_camera_index", -1), 3)
        self.assertEqual(shared.consume_value("pending_camera_index", -1), -1)

    def test_consume_command_returns_then_clears(self) -> None:
        shared = SharedState()
        shared.update(cmd_snooze=True)
        self.assertTrue(shared.consume_command("cmd_snooze"))
        self.assertFalse(shared.consume_command("cmd_snooze"))


if __name__ == "__main__":
    unittest.main()
