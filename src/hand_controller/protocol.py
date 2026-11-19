"""UDP message format. See docs/PROTOCOL.md for the full specification."""

from __future__ import annotations

import json
import time

from hand_controller.core import State

PROTOCOL_VERSION = 1


def now_ms() -> int:
    """Return the current Unix time in milliseconds."""
    return int(time.time() * 1000)


def encode_message(state: State, confidence: float, hand: str | None, ts_ms: int) -> bytes:
    """Encode one state message as compact UTF-8 JSON.

    Args:
        state: The smoothed state to report.
        confidence: Confidence 0..1, rounded to 2 decimals in the message.
        hand: "Left", "Right", or None when no hand is visible.
        ts_ms: Unix time in milliseconds.
    """
    payload = {
        "v": PROTOCOL_VERSION,
        "state": state.value,
        "confidence": round(float(confidence), 2),
        "hand": hand,
        "ts": int(ts_ms),
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")
