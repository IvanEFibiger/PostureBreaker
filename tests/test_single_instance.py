from __future__ import annotations

import sys
import unittest
import uuid

from posture_guard.single_instance import SingleInstance


@unittest.skipUnless(sys.platform.startswith("win"), "Windows named mutex")
class SingleInstanceTests(unittest.TestCase):
    def test_second_instance_is_rejected_and_release_frees_the_lock(self) -> None:
        name = f"PostureBreaker.test.{uuid.uuid4().hex}"
        first = SingleInstance(name)
        self.assertTrue(first.acquire())

        second = SingleInstance(name)
        self.assertFalse(second.acquire())

        first.release()

        third = SingleInstance(name)
        self.assertTrue(third.acquire())
        third.release()


if __name__ == "__main__":
    unittest.main()
