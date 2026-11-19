"""ModelClassifier tests: save a tiny model, load it through the normal interface."""

import joblib
import pytest

from hand_controller.classifier import ModelClassifier
from hand_controller.config import ClassifierConfig, FeaturesConfig
from hand_controller.core import State
from hand_controller.features import hand_features
from tests.synthetic import fist, pointing

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


@pytest.fixture
def model_file(tmp_path):
    hands = [(pointing(0), "RIGHT"), (pointing(180), "LEFT"), (fist(), "STOP")] * 5
    x = [hand_features(h, False) for h, _ in hands]
    y = [label for _, label in hands]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(x, y)
    path = tmp_path / "m.joblib"
    joblib.dump({"pipeline": pipe, "use_z": False, "feature_version": 1, "labels": []}, path)
    return path


def test_model_classifier_returns_valid_state(model_file):
    clf = ModelClassifier(ClassifierConfig(model_path=str(model_file), min_model_confidence=0.0), FeaturesConfig())
    state, conf = clf.classify(pointing(0))
    assert isinstance(state, State)
    assert 0.0 <= conf <= 1.0


def test_no_hand_is_stop(model_file):
    clf = ModelClassifier(ClassifierConfig(model_path=str(model_file)), FeaturesConfig())
    assert clf.classify(None) == (State.STOP, 0.0)


def test_low_confidence_becomes_stop(model_file):
    clf = ModelClassifier(ClassifierConfig(model_path=str(model_file), min_model_confidence=1.01), FeaturesConfig())
    assert clf.classify(pointing(0))[0] == State.STOP


def test_use_z_mismatch_is_rejected(model_file):
    with pytest.raises(ValueError):
        ModelClassifier(ClassifierConfig(model_path=str(model_file)), FeaturesConfig(use_z=True))


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="tools/train.py"):
        ModelClassifier(ClassifierConfig(model_path=str(tmp_path / "none.joblib")), FeaturesConfig())
