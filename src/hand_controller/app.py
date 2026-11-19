"""Main loop and command-line entry point: camera -> detector -> classifier -> smoothing -> output."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from hand_controller.capture import Camera, CameraError
from hand_controller.classifier import build_classifier
from hand_controller.config import Config, load_config
from hand_controller.core import State
from hand_controller.detection import HandDetector
from hand_controller.display import Preview
from hand_controller.output import Output, UdpOutput
from hand_controller.protocol import now_ms
from hand_controller.smoothing import Smoother


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line flags. Only flags that are given override config.toml."""
    p = argparse.ArgumentParser(prog="hand-controller", description="Hand gestures to game input over UDP.")
    p.add_argument("--config", type=Path, default=Path("config.toml"), help="config file (default: config.toml)")
    p.add_argument("--camera", type=int, dest="camera_index", help="camera index (default from config)")
    p.add_argument("--width", type=int, dest="camera_width", help="camera width in pixels")
    p.add_argument("--height", type=int, dest="camera_height", help="camera height in pixels")
    p.add_argument("--host", dest="output_host", help="UDP host to send to")
    p.add_argument("--port", type=int, dest="output_port", help="UDP port to send to")
    p.add_argument("--classifier", choices=["rules", "model"], dest="classifier_name", help="gesture classifier")
    p.add_argument("--model", dest="classifier_model_path", help="path to a trained .joblib model")
    p.add_argument("--no-window", action="store_true", help="do not open the debug window")
    return p.parse_args(argv)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Copy CLI flags that were actually given onto the config."""
    mapping = {
        "camera_index": ("camera", "index"),
        "camera_width": ("camera", "width"),
        "camera_height": ("camera", "height"),
        "output_host": ("output", "host"),
        "output_port": ("output", "port"),
        "classifier_name": ("classifier", "name"),
        "classifier_model_path": ("classifier", "model_path"),
    }
    for arg_name, (section, key) in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(getattr(cfg, section), key, value)
    if args.no_window:
        cfg.display.window = False
    return cfg


@dataclass
class _Monotonic:
    """Produces strictly increasing millisecond timestamps for MediaPipe VIDEO mode."""

    start: float = 0.0

    def __post_init__(self) -> None:
        self.start = time.monotonic()

    def next_ms(self) -> int:
        return int((time.monotonic() - self.start) * 1000)


def run(cfg: Config) -> int:
    """Run the capture loop until the user quits (q/Esc in the window, or Ctrl+C)."""
    classifier = build_classifier(cfg.classifier, cfg.features)
    smoother = Smoother(cfg.smoothing.direction_ms, cfg.smoothing.stop_ms)
    output: Output = UdpOutput(cfg.output.host, cfg.output.port)
    clock = _Monotonic()
    preview = Preview(cfg.display.window)
    last_state: State | None = None
    last_frame_time = time.monotonic()
    fps = 0.0

    print(f"Sending to udp://{cfg.output.host}:{cfg.output.port} using the '{cfg.classifier.name}' classifier")
    try:
        with Camera(cfg.camera) as camera, HandDetector(cfg.detection) as detector:
            print("Camera open. Show your hand. Press Ctrl+C (or q in the window) to quit.")
            while True:
                frame = camera.read()
                if frame is None:
                    print("Warning: no frame from camera, retrying...")
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)  # mirror, so pointing right on screen means right for the player

                hand = detector.detect(frame, clock.next_ms())
                raw, raw_conf = classifier.classify(hand)
                state, confidence = smoother.update(raw, raw_conf, clock.next_ms())
                hand_label = hand.handedness if hand is not None else None
                output.send(state, confidence, hand_label, now_ms())

                if state != last_state:
                    print(f"{time.strftime('%H:%M:%S')}  state -> {state.value}  (confidence {confidence:.2f})")
                    last_state = state

                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - last_frame_time, 1e-6) if fps else 1.0 / max(now - last_frame_time, 1e-6)
                last_frame_time = now
                if not preview.show(frame, hand, state, raw, fps, hand_label):
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    except CameraError as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        preview.close()
        output.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console entry point (`uv run hand-controller`)."""
    args = parse_args(argv)
    try:
        cfg = apply_overrides(load_config(args.config), args)
    except (ValueError, OSError) as exc:
        print(f"Config error: {exc}")
        return 2
    try:
        return run(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
