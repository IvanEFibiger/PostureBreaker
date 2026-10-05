from __future__ import annotations

import unittest

from posture_guard.models import DebugLandmark, DetectionMetrics

try:  # cv2/tkinter (and numpy) may be absent in minimal CI images.
    import numpy as np

    from posture_guard.ui import draw_guides

    _HAS_UI = True
except Exception:  # pragma: no cover - environment dependent
    np = None
    draw_guides = None
    _HAS_UI = False


def frame():  # type: ignore[no-untyped-def]
    return np.zeros((120, 160, 3), dtype=np.uint8)


def metrics_with(landmarks: dict[str, DebugLandmark]) -> DetectionMetrics:
    return DetectionMetrics(side="right", values={}, points={}, debug_landmarks=landmarks)


@unittest.skipUnless(_HAS_UI, "cv2/tkinter not available")
class DrawGuidesTests(unittest.TestCase):
    def test_partial_landmarks_do_not_crash(self) -> None:
        metrics = metrics_with(
            {
                "nose": DebugLandmark(0.5, 0.4, 0.9),
                "left_eye": DebugLandmark(0.45, 0.38, 0.9),
                "right_shoulder": DebugLandmark(0.6, 0.6, 0.9),
            }
        )
        image = frame()
        draw_guides(image, metrics, 0.55)
        self.assertGreater(int(image.sum()), 0)

    def test_missing_eye_does_not_draw_the_eye_line(self) -> None:
        with_both = metrics_with(
            {
                "left_eye": DebugLandmark(0.4, 0.4, 0.9),
                "right_eye": DebugLandmark(0.6, 0.4, 0.9),
            }
        )
        with_one = metrics_with({"left_eye": DebugLandmark(0.4, 0.4, 0.9)})
        both_image, one_image = frame(), frame()
        draw_guides(both_image, with_both, 0.55)
        draw_guides(one_image, with_one, 0.55)
        # The connecting line makes the two-eye frame strictly brighter.
        self.assertGreater(int(both_image.sum()), int(one_image.sum()))

    def test_low_visibility_hip_is_not_drawn(self) -> None:
        metrics = metrics_with(
            {
                "left_shoulder": DebugLandmark(0.4, 0.6, 0.9),
                "right_shoulder": DebugLandmark(0.6, 0.6, 0.9),
                "left_hip": DebugLandmark(0.4, 0.95, 0.1),
                "right_hip": DebugLandmark(0.6, 0.95, 0.1),
            }
        )
        image = frame()
        draw_guides(image, metrics, 0.55)
        self.assertEqual(int(image[110:, :, :].sum()), 0)

    def test_legacy_fallback_without_debug_landmarks(self) -> None:
        metrics = DetectionMetrics(
            side="right",
            values={},
            points={
                "ear": (0.5, 0.3),
                "shoulder": (0.5, 0.6),
                "hip": (0.5, 0.95),
                "nose": (0.5, 0.25),
            },
        )
        image = frame()
        draw_guides(image, metrics, 0.55)
        self.assertGreater(int(image.sum()), 0)

    def test_mirror_flag_draws_on_the_flipped_side(self) -> None:
        metrics = metrics_with({"nose": DebugLandmark(0.2, 0.5, 0.9)})
        image = frame()
        draw_guides(image, metrics, 0.55, mirror=True)
        left_half = int(image[:, :80, :].sum())
        right_half = int(image[:, 80:, :].sum())
        self.assertEqual(left_half, 0)
        self.assertGreater(right_half, 0)

    def test_none_metrics_is_safe(self) -> None:
        image = frame()
        draw_guides(image, None)
        self.assertEqual(int(image.sum()), 0)


if __name__ == "__main__":
    unittest.main()
