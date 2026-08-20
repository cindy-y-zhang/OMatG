"""Focused tests for joint-loss calibration state and diagnostics."""

import pytest
import torch

from joint_geometry.lightning import (
    CALIBRATION_BATCHES,
    TARGET_GRADIENT_RATIO,
    JointGeometryLightning,
    _gradient_norm,
)


class CalibrationHarness:
    """Minimal receiver for testing Lightning's calibration methods in isolation."""

    def __init__(self, ratio: float = 2.0) -> None:
        self._geometry_weight = None
        self._calibrated = False
        self._calibration_ratios = []
        self.ratio = ratio
        self.device = torch.device("cpu")
        self.global_step = 1
        self.messages = []

    def _gradient_ratio(self, flow_loss, geometry_loss):
        return self.ratio

    def print(self, message: str) -> None:
        self.messages.append(message)

    def log(self, *args, **kwargs) -> None:
        raise AssertionError("The frozen-ratio diagnostic should not run at step 1.")


def test_geometry_weight_calibrates_once_and_freezes() -> None:
    harness = CalibrationHarness(ratio=2.0)
    loss = torch.tensor(1.0)
    for _ in range(CALIBRATION_BATCHES):
        weight = JointGeometryLightning._geometry_weight_for(harness, loss, loss)

    expected = TARGET_GRADIENT_RATIO * 2.0
    assert float(weight) == pytest.approx(expected)
    assert harness._geometry_weight == pytest.approx(expected)
    assert harness._calibrated
    assert len(harness._calibration_ratios) == CALIBRATION_BATCHES

    harness.ratio = 20.0
    frozen = JointGeometryLightning._geometry_weight_for(harness, loss, loss)
    assert float(frozen) == pytest.approx(expected)
    assert harness._geometry_weight == pytest.approx(expected)


def test_geometry_weight_targets_the_mean_weighted_gradient_share() -> None:
    harness = CalibrationHarness()
    loss = torch.tensor(1.0)
    ratios = [1.0, 3.0] * (CALIBRATION_BATCHES // 2)
    for ratio in ratios:
        harness.ratio = ratio
        weight = JointGeometryLightning._geometry_weight_for(harness, loss, loss)

    achieved = sum(float(weight) / ratio for ratio in ratios) / len(ratios)
    assert achieved == pytest.approx(TARGET_GRADIENT_RATIO)


def test_geometry_calibration_checkpoint_round_trip() -> None:
    source = CalibrationHarness(ratio=3.0)
    source._geometry_weight = 0.2121
    source._calibrated = True
    source._calibration_ratios = [3.0] * CALIBRATION_BATCHES
    checkpoint = {}
    JointGeometryLightning.on_save_checkpoint(source, checkpoint)

    restored = CalibrationHarness()
    JointGeometryLightning.on_load_checkpoint(restored, checkpoint)

    assert restored._geometry_weight == pytest.approx(0.2121)
    assert restored._calibrated
    assert restored._calibration_ratios == [3.0] * CALIBRATION_BATCHES


def test_gradient_norm_ignores_unused_parameters() -> None:
    used = torch.nn.Parameter(torch.tensor(2.0))
    unused = torch.nn.Parameter(torch.tensor(3.0))
    loss = (2.0 * used).square()
    assert _gradient_norm(loss, [used, unused]) == pytest.approx(16.0)
