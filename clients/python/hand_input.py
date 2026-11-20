"""Receive hand-controller states over UDP. Standard library only, no engine dependency.

Example:
    from hand_input import HandInput

    hand = HandInput(port=5005)
    print(hand.state)        # "RIGHT", "LEFT", "UP", "DOWN" or "STOP"
    hand.close()

`hand.state` is "STOP" before the first message and whenever no message arrived for 300 ms.
A background thread receives the messages, so reading `state` never blocks.
"""

from __future__ import annotations

import json
import socket
import threading
import time

TIMEOUT_S = 0.3  # protocol rule: no message for 300 ms means STOP
VALID_STATES = {"RIGHT", "LEFT", "UP", "DOWN", "STOP"}


class HandInput:
    """Latest hand-controller state, received on a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.2)  # lets the thread check for close() regularly
        self._lock = threading.Lock()
        self._state = "STOP"
        self._last_received: float | None = None  # time.monotonic() of the last valid message
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    @property
    def state(self) -> str:
        """The latest direction, or "STOP" if nothing valid arrived in the last 300 ms."""
        with self._lock:
            if self._last_received is None or time.monotonic() - self._last_received > TIMEOUT_S:
                return "STOP"
            return self._state

    def close(self) -> None:
        """Stop the receive thread and close the socket."""
        self._running = False
        self._thread.join(timeout=1.0)
        self._sock.close()

    def __enter__(self) -> HandInput:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                break  # socket closed
            state = _parse(data)
            if state is not None:
                with self._lock:
                    self._state = state
                    self._last_received = time.monotonic()


def _parse(data: bytes) -> str | None:
    """Return the state from one datagram, or None if it is not a valid version-1 message."""
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(msg, dict) or msg.get("v") != 1:
        return None
    state = msg.get("state")
    return state if state in VALID_STATES else None
