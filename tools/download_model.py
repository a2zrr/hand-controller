"""Download the MediaPipe hand_landmarker.task model into models/.

Usage:  uv run python tools/download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Official model URL from the MediaPipe Hand Landmarker guide:
# https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "hand_landmarker.task"


def main() -> int:
    """Download the model unless it already exists."""
    if TARGET.exists() and TARGET.stat().st_size > 0:
        print(f"Model already present: {TARGET}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_suffix(".task.part")
    print(f"Downloading {MODEL_URL}")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while chunk := resp.read(64 * 1024):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:6.1%}", end="", flush=True)
        print()
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    tmp.replace(TARGET)
    print(f"Saved to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
