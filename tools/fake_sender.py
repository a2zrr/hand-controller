"""Send hand-controller states from the keyboard, so you can test a game without a camera.

Controls (in this terminal window):
  Arrow keys or W/A/S/D  -> RIGHT / LEFT / UP / DOWN (WASD: W=UP, A=LEFT, S=DOWN, D=RIGHT)
  Space                  -> STOP
  Q                      -> quit

The current state is re-sent 20 times per second, like the real app does every frame.
No global keyboard hooks: keys are only read while this terminal has focus.
Works on Linux (terminal, raw mode) and Windows (msvcrt).

Usage:  uv run python tools/fake_sender.py [--port 5005]
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

STATE_KEYS = {
    "d": "RIGHT", "a": "LEFT", "w": "UP", "s": "DOWN", " ": "STOP",
}
ARROWS_POSIX = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}  # after ESC [
ARROWS_WINDOWS = {"H": "UP", "P": "DOWN", "M": "RIGHT", "K": "LEFT"}  # after 0xE0 / 0x00
SEND_INTERVAL = 0.05  # 20 messages per second


def read_key_posix(timeout: float) -> str | None:
    """Read one key on Linux/macOS. Returns a state name, 'QUIT', or None.

    Uses os.read on the raw file descriptor so that select() sees exactly what is pending.
    """
    import os
    import select

    fd = sys.stdin.fileno()

    def next_byte(wait: float) -> str | None:
        if not select.select([fd], [], [], wait)[0]:
            return None
        data = os.read(fd, 1)
        return data.decode("latin-1") if data else None

    ch = next_byte(timeout)
    if ch is None:
        return None
    if ch == "\x1b":
        # Arrow keys send ESC [ X. Wait briefly for the rest of the sequence.
        if next_byte(0.01) == "[":
            return ARROWS_POSIX.get(next_byte(0.01) or "")
        return None
    return _map_char(ch)


def read_key_windows(timeout: float) -> str | None:
    """Read one key on Windows. Returns a state name, 'QUIT', or None."""
    import msvcrt

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # arrow keys send a prefix, then a code
                return ARROWS_WINDOWS.get(msvcrt.getwch())
            return _map_char(ch)
        time.sleep(0.01)
    return None


def _map_char(ch: str) -> str | None:
    """Map one typed character to a state, or 'QUIT'."""
    if ch.lower() == "q":
        return "QUIT"
    return STATE_KEYS.get(ch.lower())


def main() -> None:
    """Loop: read a key if one is pressed, then send the current state."""
    parser = argparse.ArgumentParser(description="Send states from the keyboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = (args.host, args.port)
    state = "STOP"
    print(__doc__.split("Usage")[0])
    print(f"Sending to udp://{args.host}:{args.port}. Current state: {state}")

    if sys.platform == "win32":
        read_key = read_key_windows
        old_settings = None
    else:
        import termios
        import tty

        read_key = read_key_posix
        old_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())  # single keys without Enter, no echo

    try:
        while True:
            key = read_key(SEND_INTERVAL)
            if key == "QUIT":
                break
            if key is not None and key != state:
                state = key
                print(f"state -> {state}")
            msg = {"v": 1, "state": state, "confidence": 1.0, "hand": None,
                   "ts": int(time.time() * 1000)}
            sock.sendto(json.dumps(msg, separators=(",", ":")).encode("utf-8"), address)
    except KeyboardInterrupt:
        pass
    finally:
        if old_settings is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        sock.close()
        print("\nBye.")


if __name__ == "__main__":
    main()
