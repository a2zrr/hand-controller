"""Record labelled hand samples from the webcam, one CSV file per recording session.

Controls (click the window first so it has keyboard focus):
  1 RIGHT   2 LEFT   3 UP   4 DOWN   5 STOP
      Press a number to START recording that label; press the same number again to STOP.
      Pressing a different number switches the label.
  Q or Esc  quit (the file is already saved row by row)

Each recording session writes data/session-<timestamp>.csv. Use at least two sessions
(for example one per day or per lighting) so the train/test split by session works.

Usage:  uv run python tools/record.py [--camera 0] [--no-mirror]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from hand_controller.capture import Camera, CameraError
from hand_controller.config import load_config
from hand_controller.core import State
from hand_controller.dataset import LABELS, SampleWriter
from hand_controller.detection import HandDetector
from hand_controller.display import STATE_COLORS, draw_hand, window_help

ROOT = Path(__file__).resolve().parents[1]
KEY_TO_LABEL = {ord(str(i + 1)): label for i, label in enumerate(LABELS)}
WINDOW = "record - hand-controller"


def main() -> int:
    """Open the camera and record samples while the user presses the label keys."""
    parser = argparse.ArgumentParser(description="Record gesture samples.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--camera", type=int, help="camera index")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera
    session = time.strftime("%Y%m%d-%H%M%S")
    path = args.data_dir / f"session-{session}.csv"

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        print(window_help(exc), file=sys.stderr)
        return 1

    recording: str | None = None
    counts: Counter[str] = Counter()
    writer = SampleWriter(path, session)
    print(f"Writing samples to {path}")
    try:
        with Camera(cfg.camera) as camera, HandDetector(cfg.detection) as detector:
            while True:
                frame = camera.read()
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)  # same mirroring as the runtime
                hand = detector.detect(frame, int(time.monotonic() * 1000))
                if hand is not None and recording is not None:
                    writer.write(recording, hand)
                    counts[recording] += 1

                canvas = frame.copy()
                if hand is not None:
                    draw_hand(canvas, hand)
                _overlay(canvas, recording, counts, hand is not None)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in KEY_TO_LABEL:
                    label = KEY_TO_LABEL[key]
                    recording = None if recording == label else label
                    print(f"recording: {recording or 'off'}")
    except CameraError as exc:
        print(f"Error: {exc}")
        return 1
    except cv2.error as exc:
        print(window_help(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        cv2.destroyAllWindows()

    print("Samples in this session: " + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none"))
    return 0


def _overlay(canvas, recording: str | None, counts: Counter[str], hand_visible: bool) -> None:
    """Show what is being recorded and how many samples each class has."""
    if recording:
        color = (0, 0, 255)
        status = f"REC {recording}  (press {LABELS.index(recording) + 1} to stop)"
    else:
        color = (200, 200, 200)
        status = "NOT RECORDING  (press 1-5 to start)"
    cv2.putText(canvas, status, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(canvas, "" if hand_visible else "no hand detected", (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    for i, label in enumerate(LABELS):
        text = f"{i + 1} {label}: {counts[label]}"
        cv2.putText(canvas, text, (12, 110 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATE_COLORS[State(label)], 2)


if __name__ == "__main__":
    raise SystemExit(main())
