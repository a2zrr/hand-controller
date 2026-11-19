"""Debounce raw per-frame states so the output does not flicker.

Smoothing is time-based (milliseconds), not frame-based, so the feel does not change when
the frame rate changes. A new state is accepted only after it has been the raw result
continuously for the required time.
"""

from __future__ import annotations

from hand_controller.core import State


class Smoother:
    """Accept a new state only after it has held for `direction_ms` (or `stop_ms` for STOP)."""

    def __init__(self, direction_ms: float = 120.0, stop_ms: float = 60.0) -> None:
        if direction_ms < 0 or stop_ms < 0:
            raise ValueError("Smoothing times must not be negative")
        self.direction_ms = direction_ms
        self.stop_ms = stop_ms
        self.state = State.STOP
        self.confidence = 0.0
        self._candidate: State | None = None
        self._since_ms = 0.0

    def update(self, raw: State, confidence: float, now_ms: float) -> tuple[State, float]:
        """Feed one raw frame result taken at time `now_ms` and return the smoothed (state, confidence)."""
        if raw == self.state:
            # Still agreeing with the current state: refresh confidence, drop any pending switch.
            self._candidate = None
            self.confidence = confidence
            return self.state, self.confidence

        if raw != self._candidate:
            self._candidate = raw
            self._since_ms = now_ms

        required = self.stop_ms if raw == State.STOP else self.direction_ms
        if now_ms - self._since_ms >= required:
            self.state = raw
            self.confidence = confidence
            self._candidate = None
        return self.state, self.confidence
