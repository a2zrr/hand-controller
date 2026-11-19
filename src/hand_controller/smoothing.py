"""Debounce raw per-frame states so the output does not flicker."""

from __future__ import annotations

from hand_controller.core import State


class Smoother:
    """Accept a new state only after it has been seen in N consecutive frames.

    Switching to a direction needs `direction_frames` frames, switching to STOP needs
    `stop_frames` (fewer, because STOP is the safe state and should react quickly).
    """

    def __init__(self, direction_frames: int = 4, stop_frames: int = 2) -> None:
        if direction_frames < 1 or stop_frames < 1:
            raise ValueError("Frame counts must be at least 1")
        self.direction_frames = direction_frames
        self.stop_frames = stop_frames
        self.state = State.STOP
        self.confidence = 0.0
        self._candidate: State | None = None
        self._count = 0

    def update(self, raw: State, confidence: float) -> tuple[State, float]:
        """Feed one raw frame result and return the smoothed (state, confidence)."""
        if raw == self.state:
            # Still agreeing with the current state: refresh confidence, drop any pending switch.
            self._candidate = None
            self._count = 0
            self.confidence = confidence
            return self.state, self.confidence

        if raw != self._candidate:
            self._candidate = raw
            self._count = 0
        self._count += 1

        needed = self.stop_frames if raw == State.STOP else self.direction_frames
        if self._count >= needed:
            self.state = raw
            self.confidence = confidence
            self._candidate = None
            self._count = 0
        return self.state, self.confidence
