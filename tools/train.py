"""Train a gesture classifier (MLP) on recorded data and save models/gesture_clf.joblib.

The train/test split is by SESSION, not by random frames: frames of the same recording
session are highly similar, so a random split would give over-optimistic scores.

Usage:  uv run python tools/train.py [--data-dir data] [--out models/gesture_clf.joblib]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hand_controller.config import load_config
from hand_controller.dataset import LABELS, load_data_dir, split_by_session, to_features
from hand_controller.features import FEATURE_VERSION

ROOT = Path(__file__).resolve().parents[1]


def build_pipeline() -> object:
    """Scaler + small MLP. No early stopping: on small recording sets it stops too early and gives low, poorly calibrated probabilities."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=0),
    )


def print_scores(y_true: list[str], y_pred: list[str]) -> None:
    """Accuracy, per-class report and confusion matrix."""
    labels = [label for label in LABELS if label in set(y_true) | set(y_pred)]
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print(f"Test accuracy: {acc:.3f} on {len(y_true)} samples")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print("labels:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))


def main() -> int:
    """Load data, split by session, train, evaluate on the held-out sessions, save the model."""
    parser = argparse.ArgumentParser(description="Train the gesture classifier.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "gesture_clf.joblib")
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_z = cfg.features.use_z
    samples = load_data_dir(args.data_dir)
    train, test = split_by_session(samples)
    sessions = lambda rows: sorted({s.session for s in rows})  # noqa: E731
    print(f"Train: {len(train)} samples from sessions {sessions(train)}")
    print(f"Test:  {len(test)} samples from sessions {sessions(test)}")

    x_train = to_features(train, use_z)
    y_train = [s.label for s in train]
    x_test = to_features(test, use_z)
    y_test = [s.label for s in test]

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    print_scores(y_test, list(pipeline.predict(x_test)))

    # Retrain on ALL sessions for the saved model, so the final model uses every recording.
    x_all = to_features(samples, use_z)
    y_all = [s.label for s in samples]
    final = build_pipeline()
    final.fit(x_all, y_all)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": final, "use_z": use_z, "feature_version": FEATURE_VERSION, "labels": LABELS},
        args.out,
    )
    print(f"Saved model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
