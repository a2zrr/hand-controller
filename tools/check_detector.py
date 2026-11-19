"""Smoke test: run the MediaPipe detector on real webcam frames and print what it returns.

Use it after installing or upgrading mediapipe. Hold a hand up to the camera while it runs.

Usage:  uv run python tools/check_detector.py [--camera 0] [--frames 60]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from hand_controller.capture import Camera
from hand_controller.config import load_config
from hand_controller.detection import HandDetector

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Read frames, detect, print the first detected hand and a summary."""
    parser = argparse.ArgumentParser(description="Check the hand detector on the webcam.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera

    hits = 0
    printed = False
    with Camera(cfg.camera) as cam, HandDetector(cfg.detection) as det:
        print(f"Camera frame size: {cam.size}")
        start = time.monotonic()
        for i in range(args.frames):
            frame = cam.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            hand = det.detect(frame, int((time.monotonic() - start) * 1000))
            if hand is None:
                continue
            hits += 1
            if not printed:
                printed = True
                print(f"frame {i}: handedness={hand.handedness!r} score={hand.score:.3f} "
                      f"landmarks shape={hand.landmarks.shape} size={hand.width}x{hand.height}")
                print(f"  wrist (normalized) = {hand.landmarks[0].round(3).tolist()}")
                print(f"  index tip (pixels) = {hand.pixels()[8].round(1).tolist()}")
    print(f"Hand detected in {hits}/{args.frames} frames.")
    if hits == 0:
        print("No hand seen: show your hand to the camera with good light and run again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
