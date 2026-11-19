"""Message encoding tests."""

import json

from hand_controller.core import State
from hand_controller.protocol import PROTOCOL_VERSION, encode_message


def test_message_has_all_fields_with_expected_types():
    msg = json.loads(encode_message(State.LEFT, 0.9432, "Right", 1732057200123))
    assert msg == {"v": 1, "state": "LEFT", "confidence": 0.94, "hand": "Right", "ts": 1732057200123}
    assert PROTOCOL_VERSION == 1


def test_no_hand_is_null():
    msg = json.loads(encode_message(State.STOP, 0.0, None, 5))
    assert msg["hand"] is None and msg["state"] == "STOP"


def test_encoding_is_compact_single_line_utf8():
    raw = encode_message(State.UP, 1.0, "Left", 10)
    assert b"\n" not in raw and b" " not in raw
    assert raw.decode("utf-8").startswith('{"v":1,')


def test_timestamp_is_integer_milliseconds():
    msg = json.loads(encode_message(State.DOWN, 0.5, None, 1700000000000))
    assert isinstance(msg["ts"], int)
