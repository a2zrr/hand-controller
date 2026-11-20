"""Example: print the hand-controller state whenever it changes.

Usage:  uv run python clients/python/listen.py [--port 5005]

Start hand-controller (or tools/fake_sender.py) in another terminal. Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import time

from hand_input import HandInput


def main() -> None:
    """Poll the HandInput client 20 times a second and print state changes."""
    parser = argparse.ArgumentParser(description="Print hand-controller states.")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()

    hand = HandInput(port=args.port)
    print(f"Listening on UDP port {args.port}. Ctrl+C to stop.")
    last = None
    try:
        while True:
            state = hand.state
            if state != last:
                print(f"{time.strftime('%H:%M:%S')}  {state}")
                last = state
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        hand.close()


if __name__ == "__main__":
    main()
