from __future__ import annotations

import unittest

from posture_guard.camera import CameraError, list_cameras, open_camera


class _FakeCapture:
    def __init__(self, opened: bool) -> None:
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True


class OpenCameraTests(unittest.TestCase):
    def test_returns_open_capture_on_first_try(self) -> None:
        capture = open_camera(lambda index: _FakeCapture(True), 0, attempts=3, sleep=lambda _: None)
        self.assertTrue(capture.isOpened())

    def test_retries_with_backoff_until_success(self) -> None:
        outcomes = iter([False, False, True])
        sleeps: list[float] = []
        capture = open_camera(
            lambda index: _FakeCapture(next(outcomes)),
            0,
            attempts=3,
            backoff=(1.0, 2.0),
            sleep=sleeps.append,
        )
        self.assertTrue(capture.isOpened())
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_raises_camera_error_after_all_attempts(self) -> None:
        with self.assertRaises(CameraError):
            open_camera(lambda index: _FakeCapture(False), 0, attempts=2, sleep=lambda _: None)

    def test_on_retry_reports_each_backoff(self) -> None:
        retries: list[tuple[int, float]] = []
        with self.assertRaises(CameraError):
            open_camera(
                lambda index: _FakeCapture(False),
                0,
                attempts=2,
                backoff=(3.0,),
                sleep=lambda _: None,
                on_retry=lambda attempt, delay: retries.append((attempt, delay)),
            )
        self.assertEqual(retries, [(1, 3.0)])


class ListCamerasTests(unittest.TestCase):
    def test_returns_only_openable_indices(self) -> None:
        self.assertEqual(list_cameras(lambda index: _FakeCapture(index in {0, 2}), max_index=3), [0, 2])

    def test_returns_empty_when_none_open(self) -> None:
        self.assertEqual(list_cameras(lambda index: _FakeCapture(False), max_index=2), [])


if __name__ == "__main__":
    unittest.main()
