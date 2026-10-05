from __future__ import annotations

from dataclasses import dataclass

from .models import CalibrationSet, ViewProfile, ViewState

# Orientation weights for the profile distance (plan §51).
HEAD_YAW_WEIGHT = 0.7
TORSO_YAW_WEIGHT = 0.3

# Each signal is normalized by its calibrated spread; these floors keep an
# unusually stable calibration from making the distance explode.
HEAD_YAW_SCALE_FLOOR = 0.03
TORSO_YAW_SCALE_FLOOR = 3.0


@dataclass
class ViewSelection:
    profile: ViewProfile | None
    distance: float
    changed: bool


def profile_distance(state: ViewState, profile: ViewProfile) -> float | None:
    """Weighted distance between the current view and a profile's baseline.

    Both yaws are adimensionalized by their calibrated spread, so a signal in
    degrees (torso yaw) cannot dominate one that is a ratio (head yaw).
    """
    if state.head_yaw is None or profile.head_yaw_mean is None:
        return None
    head_distance = abs(state.head_yaw - profile.head_yaw_mean) / max(
        profile.head_yaw_std, HEAD_YAW_SCALE_FLOOR
    )
    distance = head_distance * HEAD_YAW_WEIGHT
    if state.torso_yaw is not None and profile.torso_yaw_mean is not None:
        torso_distance = abs(state.torso_yaw - profile.torso_yaw_mean) / max(
            profile.torso_yaw_std, TORSO_YAW_SCALE_FLOOR
        )
        distance += torso_distance * TORSO_YAW_WEIGHT
    return distance


class ViewSelector:
    """Picks the closest view profile, switching only after it stays winning."""

    def __init__(self, stability_seconds: float = 0.75, initial_id: str | None = None) -> None:
        self.stability_seconds = stability_seconds
        self.active_id = initial_id
        self._candidate_id: str | None = None
        self._candidate_seconds = 0.0

    def _active_profile(self, calibration_set: CalibrationSet) -> ViewProfile | None:
        for profile in calibration_set.profiles:
            if profile.id == self.active_id:
                return profile
        return None

    def update(self, calibration_set: CalibrationSet, state: ViewState | None, dt: float) -> ViewSelection:
        candidates = [profile for profile in calibration_set.profiles if profile.head_yaw_mean is not None]
        if not candidates or state is None or state.head_yaw is None:
            return ViewSelection(None, 0.0, False)

        scored = [(profile_distance(state, profile), profile) for profile in candidates]
        scored = [(distance, profile) for distance, profile in scored if distance is not None]
        if not scored:
            return ViewSelection(None, 0.0, False)

        best_distance, best = min(scored, key=lambda item: item[0])

        if self.active_id is None:
            self.active_id = best.id
            self._candidate_id = None
            self._candidate_seconds = 0.0
            return ViewSelection(best, best_distance, True)

        if best.id == self.active_id:
            self._candidate_id = None
            self._candidate_seconds = 0.0
            return ViewSelection(best, best_distance, False)

        if self._candidate_id != best.id:
            self._candidate_id = best.id
            self._candidate_seconds = 0.0
            return ViewSelection(self._active_profile(calibration_set), best_distance, False)

        self._candidate_seconds += dt
        if self._candidate_seconds >= self.stability_seconds:
            self.active_id = best.id
            self._candidate_id = None
            self._candidate_seconds = 0.0
            return ViewSelection(best, best_distance, True)

        return ViewSelection(self._active_profile(calibration_set), best_distance, False)
