"""Configuration: defaults, config.toml loading and CLI overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    """Webcam settings."""

    index: int = 0
    width: int = 640
    height: int = 480


@dataclass
class DetectionConfig:
    """MediaPipe HandLandmarker settings."""

    model_path: str = "models/hand_landmarker.task"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Only changes the reported "hand" label. Direction logic never uses it.
    swap_handedness: bool = False
    # "auto" tries the GPU delegate and falls back to CPU; "gpu" or "cpu" force one.
    delegate: str = "auto"


@dataclass
class FeaturesConfig:
    """Feature extraction settings shared by recorder, trainer and runtime."""

    use_z: bool = False


@dataclass
class ClassifierConfig:
    """Gesture classifier settings."""

    name: str = "rules"  # "rules" or "model"
    model_path: str = "models/gesture_clf.joblib"
    min_hand_score: float = 0.5  # hands below this handedness score count as "no hand"
    finger_extend_ratio: float = 1.15  # tip-to-wrist distance must exceed pip-to-wrist * ratio
    dead_zone_deg: float = 20.0  # width of the gap between two direction sectors
    dead_zone_state: str = "STOP"  # state used for points inside the dead zone
    min_model_confidence: float = 0.6  # model predictions below this become STOP


@dataclass
class SmoothingConfig:
    """How long (ms) a new state must hold before it is accepted."""

    direction_ms: float = 120.0
    stop_ms: float = 60.0


@dataclass
class OutputConfig:
    """Where UDP messages are sent."""

    host: str = "127.0.0.1"
    port: int = 5005


@dataclass
class DisplayConfig:
    """Debug window settings."""

    window: bool = True


@dataclass
class Config:
    """Complete application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


def _apply_table(target: Any, table: dict[str, Any], section: str) -> None:
    """Copy values from a parsed TOML table onto a dataclass, rejecting unknown keys."""
    known = {f.name for f in fields(target)}
    for key, value in table.items():
        if key not in known:
            raise ValueError(f"Unknown config key [{section}] {key}")
        setattr(target, key, value)


def load_config(path: Path | None) -> Config:
    """Load config.toml if it exists, otherwise return the built-in defaults."""
    cfg = Config()
    if path is None or not path.exists():
        return cfg
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    for section, table in data.items():
        target = getattr(cfg, section, None)
        if target is None or not is_dataclass(target):
            raise ValueError(f"Unknown config section [{section}] in {path}")
        _apply_table(target, table, section)
    return cfg
