from __future__ import annotations

import unittest

from posture_guard.timing import is_session_gap


class IsSessionGapTests(unittest.TestCase):
    def test_normal_frame_is_not_a_gap(self) -> None:
        self.assertFalse(is_session_gap(1.0 / 15.0, 30.0))

    def test_suspension_scale_gap_is_detected(self) -> None:
        self.assertTrue(is_session_gap(2400.0, 30.0))

    def test_exact_threshold_is_not_a_gap(self) -> None:
        self.assertFalse(is_session_gap(30.0, 30.0))


if __name__ == "__main__":
    unittest.main()
