"""Delegate -> image format choice. No camera and no landmarker are created here.

The format must match the delegate: a mismatch on the macOS GPU delegate is a native abort,
so these tests pin the pairing down.
"""

import cv2
import mediapipe as mp
import numpy as np
import pytest

from hand_controller.detection import frame_plan_for


def test_gpu_uses_srgba():
    plan = frame_plan_for("gpu")
    assert plan.image_format == mp.ImageFormat.SRGBA
    assert plan.conversion == cv2.COLOR_BGR2RGBA


def test_cpu_uses_srgb():
    plan = frame_plan_for("cpu")
    assert plan.image_format == mp.ImageFormat.SRGB
    assert plan.conversion == cv2.COLOR_BGR2RGB


@pytest.mark.parametrize("delegate", ["gpu", "cpu"])
def test_conversion_output_channels_match_image_format(delegate):
    # The channel count of the converted frame must equal the channels the image format
    # declares, otherwise MediaPipe receives a malformed image.
    expected_channels = {mp.ImageFormat.SRGB: 3, mp.ImageFormat.SRGBA: 4}
    plan = frame_plan_for(delegate)
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    converted = cv2.cvtColor(bgr, plan.conversion)
    assert converted.shape[2] == expected_channels[plan.image_format]


@pytest.mark.parametrize("delegate", ["auto", "", "GPU", "metal"])
def test_unknown_delegate_is_rejected(delegate):
    # "auto" is resolved to a concrete delegate before this is called, so it is not valid here.
    with pytest.raises(ValueError):
        frame_plan_for(delegate)
