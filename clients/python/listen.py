"""Print every hand-controller message it receives. Useful to check the stream.

Usage:  uv run python clients/python/listen.py [--port 5005]

Run hand-controller in another terminal, or tools/fake_sender.py, and you will see the
messages here. Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

VALID_STATES = {"RIGHT", "LEFT", "UP", "DOWN", "STOP"}


def main() -> None:
    """Listen on UDP and print each decoded message, with a warning for timeouts."""
    parser = argparse.ArgumentParser(description="Listen for hand-controller UDP messages.")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5005, help="UDP port (default 5005)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.3)  # 300 ms, the same timeout the game uses
    print(f"Listening on udp://{args.host}:{args.port} (Ctrl+C to stop)")
    last_seen = time.monotonic()
    timed_out = False
    try:
        while True:
            try:
                data, _addr = sock.recvfrom(4096)
            except TimeoutError:
                if not timed_out:
                    print("-- no message for 300 ms: a game should treat the state as STOP --")
                    timed_out = True
                continue
            timed_out = False
            now = time.monotonic()
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                print(f"invalid datagram: {data!r}")
                continue
            if msg.get("v") != 1 or msg.get("state") not in VALID_STATES:
                print(f"unexpected message: {msg}")
                continue
            gap_ms = (now - last_seen) * 1000
            last_seen = now
            print(f"{msg['state']:<5} conf={msg.get('confidence')} hand={msg.get('hand')} "
                  f"ts={msg.get('ts')} gap={gap_ms:6.1f} ms")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
