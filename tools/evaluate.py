"""Compare the rule classifier and the trained model on the same held-out recordings.

Uses the same session-based test split as tools/train.py.

Usage:  uv run python tools/evaluate.py [--data-dir data] [--model models/gesture_clf.joblib]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hand_controller.classifier import ModelClassifier, RuleClassifier
from hand_controller.config import ClassifierConfig, load_config
from hand_controller.dataset import load_data_dir, split_by_session

ROOT = Path(__file__).resolve().parents[1]


def accuracy(classifier, samples) -> float:
    """Fraction of samples whose predicted state equals the recorded label."""
    correct = sum(
        1 for s in samples if classifier.classify(s.to_hand())[0].value == s.label
    )
    return correct / len(samples) if samples else 0.0


def main() -> int:
    """Print the accuracy of both classifiers on the held-out sessions."""
    parser = argparse.ArgumentParser(description="Compare rules and model on recorded data.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "gesture_clf.joblib")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _train, test = split_by_session(load_data_dir(args.data_dir))
    rules = RuleClassifier(cfg.classifier)
    cfg_model = ClassifierConfig(
        model_path=str(args.model),
        min_hand_score=cfg.classifier.min_hand_score,
        min_model_confidence=cfg.classifier.min_model_confidence,
    )
    model = ModelClassifier(cfg_model, cfg.features)

    print(f"Test samples (held-out sessions): {len(test)}")
    print(f"RuleClassifier accuracy:  {accuracy(rules, test):.3f}")
    print(f"ModelClassifier accuracy: {accuracy(model, test):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
