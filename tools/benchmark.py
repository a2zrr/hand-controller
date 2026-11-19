"""Measure the real pipeline's frame rate and per-stage timings (no window, no UDP).

Runs camera -> mirror -> MediaPipe -> classifier for a warm-up period, then measures
for --seconds and prints camera FPS, the time spent in each stage, and the loop FPS.

Usage:  uv run python tools/benchmark.py [--seconds 20] [--width 640 --height 480]
Show a hand to the camera while it runs, so the detector does real work.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2

from hand_controller.capture import Camera
from hand_controller.classifier import build_classifier
from hand_controller.config import load_config
from hand_controller.detection import HandDetector

ROOT = Path(__file__).resolve().parents[1]


def summarize(name: str, values_ms: list[float]) -> str:
    """Median and 95th percentile of a list of millisecond timings."""
    if not values_ms:
        return f"{name:<14} no samples"
    values_ms = sorted(values_ms)
    p95 = values_ms[min(len(values_ms) - 1, int(len(values_ms) * 0.95))]
    return f"{name:<14} median {statistics.median(values_ms):6.2f} ms   p95 {p95:6.2f} ms"


def main() -> int:
    """Run the benchmark and print the results."""
    parser = argparse.ArgumentParser(description="Benchmark the frame rate of the pipeline.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--camera", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.width:
        cfg.camera.width = args.width
    if args.height:
        cfg.camera.height = args.height
    if args.camera is not None:
        cfg.camera.index = args.camera

    classifier = build_classifier(cfg.classifier, cfg.features)
    read_ms: list[float] = []
    detect_ms: list[float] = []
    classify_ms: list[float] = []
    loop_times: list[float] = []

    with Camera(cfg.camera) as cam, HandDetector(cfg.detection) as det:
        print(f"Camera {cam.size[0]}x{cam.size[1]} requested {cfg.camera.width}x{cfg.camera.height}")
        start = time.monotonic()
        measuring_from: float | None = None
        last = None
        frames = 0
        while True:
            t0 = time.perf_counter()
            frame = cam.read()
            t1 = time.perf_counter()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            hand = det.detect(frame, int((time.monotonic() - start) * 1000))
            t2 = time.perf_counter()
            classifier.classify(hand)
            t3 = time.perf_counter()

            now = time.monotonic()
            if now - start < args.warmup:
                continue
            if measuring_from is None:
                measuring_from = now
                last = None
                continue
            frames += 1
            read_ms.append((t1 - t0) * 1000)
            detect_ms.append((t2 - t1) * 1000)
            classify_ms.append((t3 - t2) * 1000)
            if last is not None:
                loop_times.append(now - last)
            last = now
            if now - measuring_from >= args.seconds:
                break

    elapsed = (last or measuring_from or start) - (measuring_from or start)
    loop_fps = frames / elapsed if elapsed > 0 else 0.0
    cam_fps = 1000.0 / statistics.median(read_ms) if read_ms else 0.0
    print(f"\nMeasured {frames} frames over {elapsed:.1f} s")
    print(f"Loop FPS (frames per second):  {loop_fps:5.1f}")
    print(f"Camera read FPS (from median read time): {cam_fps:5.1f}")
    print(summarize("camera read", read_ms))
    print(summarize("MediaPipe", detect_ms))
    print(summarize("classifier", classify_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
