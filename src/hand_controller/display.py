"""Debug window: mirrored camera, landmarks, smoothed and raw state, FPS.

If the OpenCV window cannot be created (for example on Wayland without XWayland),
the Preview switches itself off with a message and the app continues headless.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from hand_controller.core import Hand, State

WINDOW_NAME = "hand-controller"

# OpenCV's pip wheels ship only the X11 ("xcb") Qt platform plugin. Without this, Qt looks for a
# Wayland plugin that is not there and prints a warning. XWayland (xwayland-satellite) provides X11.
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
# Hand skeleton edges (MediaPipe hand topology).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
STATE_COLORS = {
    State.STOP: (0, 0, 255),
    State.RIGHT: (0, 200, 0),
    State.LEFT: (0, 200, 0),
    State.UP: (0, 200, 0),
    State.DOWN: (0, 200, 0),
}


class Preview:
    """Shows the debug window. `enabled` becomes False if the window cannot be opened."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            return
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            self.enabled = False
            print(window_help(exc), file=sys.stderr)
            print("Continuing without the debug window (headless mode).", file=sys.stderr)

    def show(
        self,
        frame: np.ndarray,
        hand: Hand | None,
        smoothed: State,
        raw: State,
        fps: float,
        hand_label: str | None,
    ) -> bool:
        """Draw and display one frame. Returns False if the user asked to quit (q or Esc)."""
        if not self.enabled:
            return True
        canvas = frame.copy()
        if hand is not None:
            draw_hand(canvas, hand)
        color = STATE_COLORS[smoothed]
        cv2.putText(canvas, smoothed.value, (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
        info = f"raw: {raw.value}   hand: {hand_label or '-'}   fps: {fps:4.1f}"
        cv2.putText(canvas, info, (16, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        try:
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self.enabled = False
            print(window_help(exc), file=sys.stderr)
            print("Continuing without the debug window (headless mode).", file=sys.stderr)
            return True
        return key not in (ord("q"), 27)

    def close(self) -> None:
        """Destroy the window if it was created."""
        if self.enabled:
            cv2.destroyAllWindows()


def draw_hand(canvas: np.ndarray, hand: Hand) -> None:
    """Draw the skeleton and landmark dots."""
    pts = hand.pixels().astype(int)
    for a, b in CONNECTIONS:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (255, 200, 0), 2)
    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 255), -1)


def window_help(exc: Exception) -> str:
    """Explain how to fix a failing OpenCV window."""
    return (
        f"Could not open the debug window: {exc}\n"
        "  - On Wayland (Niri, GNOME, ...) OpenCV's windows need X11/XWayland.\n"
        "    Install xwayland-satellite and make sure DISPLAY is set, then restart the app.\n"
        "  - Or run without the window: `uv run hand-controller --no-window`."
    )
