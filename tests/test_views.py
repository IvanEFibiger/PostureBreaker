from __future__ import annotations

import unittest

from posture_guard.models import CalibrationSet, ViewProfile, ViewState
from posture_guard.views import ViewSelector, profile_distance


def make_set(*profiles: ViewProfile) -> CalibrationSet:
    return CalibrationSet(profiles=list(profiles), active_profile_id=profiles[0].id if profiles else None)


MONITOR_1 = ViewProfile(id="m1", name="Monitor 1", head_yaw_mean=0.4, torso_yaw_mean=0.05)
MONITOR_2 = ViewProfile(id="m2", name="Monitor 2", head_yaw_mean=0.0, torso_yaw_mean=0.0)


class ProfileDistanceTests(unittest.TestCase):
    def test_weighted_distance_uses_head_and_torso(self) -> None:
        distance = profile_distance(ViewState(head_yaw=0.4, torso_yaw=0.05), MONITOR_1)
        self.assertAlmostEqual(distance, 0.0)

    def test_distance_to_other_profile_is_larger(self) -> None:
        state = ViewState(head_yaw=0.4, torso_yaw=0.05)
        self.assertLess(profile_distance(state, MONITOR_1), profile_distance(state, MONITOR_2))

    def test_head_only_when_torso_missing(self) -> None:
        distance = profile_distance(ViewState(head_yaw=0.1), MONITOR_1)
        self.assertAlmostEqual(distance, abs(0.1 - 0.4) / 0.03 * 0.7)

    def test_normalizes_units_so_torso_does_not_dominate(self) -> None:
        profile = ViewProfile(id="p", name="P", head_yaw_mean=0.0, head_yaw_std=0.1, torso_yaw_mean=0.0, torso_yaw_std=5.0)
        distance = profile_distance(ViewState(head_yaw=0.3, torso_yaw=10.0), profile)
        # head: 3.0 * 0.7 = 2.1 ; torso: 2.0 * 0.3 = 0.6
        self.assertAlmostEqual(distance, 2.7)

    def test_none_when_head_unavailable(self) -> None:
        self.assertIsNone(profile_distance(ViewState(head_yaw=None), MONITOR_1))
        self.assertIsNone(profile_distance(ViewState(head_yaw=0.1), ViewProfile(id="x", name="X")))


class ViewSelectorTests(unittest.TestCase):
    def test_first_update_selects_closest_profile(self) -> None:
        selector = ViewSelector(stability_seconds=0.75)
        selection = selector.update(make_set(MONITOR_1, MONITOR_2), ViewState(head_yaw=0.4), dt=1.0)
        self.assertTrue(selection.changed)
        self.assertEqual(selection.profile.id, "m1")

    def test_stays_on_active_profile(self) -> None:
        selector = ViewSelector(stability_seconds=0.75, initial_id="m1")
        selection = selector.update(make_set(MONITOR_1, MONITOR_2), ViewState(head_yaw=0.4), dt=1.0)
        self.assertFalse(selection.changed)
        self.assertEqual(selection.profile.id, "m1")

    def test_switch_requires_stability_window(self) -> None:
        selector = ViewSelector(stability_seconds=0.75, initial_id="m1")
        calibration_set = make_set(MONITOR_1, MONITOR_2)
        first = selector.update(calibration_set, ViewState(head_yaw=0.0), dt=0.5)
        self.assertFalse(first.changed)
        self.assertEqual(first.profile.id, "m1")
        second = selector.update(calibration_set, ViewState(head_yaw=0.0), dt=0.8)
        self.assertTrue(second.changed)
        self.assertEqual(second.profile.id, "m2")

    def test_flapping_candidate_does_not_switch(self) -> None:
        selector = ViewSelector(stability_seconds=0.75, initial_id="m1")
        calibration_set = make_set(MONITOR_1, MONITOR_2)
        selector.update(calibration_set, ViewState(head_yaw=0.0), dt=0.4)
        selection = selector.update(calibration_set, ViewState(head_yaw=0.4), dt=0.4)
        self.assertFalse(selection.changed)
        self.assertEqual(selection.profile.id, "m1")

    def test_no_profiles_returns_none(self) -> None:
        selector = ViewSelector()
        selection = selector.update(make_set(), ViewState(head_yaw=0.4), dt=1.0)
        self.assertIsNone(selection.profile)
        self.assertFalse(selection.changed)

    def test_profile_without_baseline_is_ignored(self) -> None:
        selector = ViewSelector()
        selection = selector.update(make_set(ViewProfile(id="x", name="X")), ViewState(head_yaw=0.4), dt=1.0)
        self.assertIsNone(selection.profile)


if __name__ == "__main__":
    unittest.main()
