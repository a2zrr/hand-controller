"""Recorded-data tests: CSV round trip and the split by session (no camera needed)."""

import pytest

from hand_controller.dataset import SampleWriter, load_data_dir, load_samples, split_by_session
from tests.synthetic import fist, pointing


def write_session(path, session, hands):
    writer = SampleWriter(path, session)
    for label, hand in hands:
        writer.write(label, hand)
    writer.close()


def test_round_trip_keeps_labels_sessions_and_landmarks(tmp_path):
    p = tmp_path / "session-a.csv"
    hand = pointing(0)
    write_session(p, "a", [("RIGHT", hand), ("STOP", fist())])
    samples = load_samples([p])
    assert [s.label for s in samples] == ["RIGHT", "STOP"]
    assert all(s.session == "a" for s in samples)
    assert samples[0].landmarks.shape == (21, 3)
    assert abs(samples[0].landmarks[8, 0] - hand.landmarks[8, 0]) < 1e-5


def test_split_is_by_session_never_mixed(tmp_path):
    for name in ("s1", "s2", "s3", "s4"):
        write_session(tmp_path / f"{name}.csv", name, [("UP", pointing(90)), ("STOP", fist())])
    samples = load_data_dir(tmp_path)
    train, test = split_by_session(samples)
    assert {s.session for s in train}.isdisjoint({s.session for s in test})
    assert {s.session for s in test} == {"s4"}  # 25% of 4 sessions = 1 test session
    assert len(train) + len(test) == len(samples)


def test_split_needs_two_sessions(tmp_path):
    write_session(tmp_path / "only.csv", "only", [("UP", pointing(90))])
    with pytest.raises(ValueError):
        split_by_session(load_data_dir(tmp_path))


def test_empty_data_dir_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data_dir(tmp_path)
