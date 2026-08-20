"""Tests for the frozen geometry-readable backbone selection rule."""

from joint_geometry.scripts.probe_backbone import select_backbone


def _entry(mean: float, low: float, high: float) -> dict:
    return {
        "chemistry_relative_r2": {
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
        }
    }


def test_backbone_gate_prefers_cheapest_statistically_tied_graph() -> None:
    clean = {
        "fc": _entry(0.10, 0.05, 0.15),
        "fc_distance": _entry(0.14, 0.10, 0.18),
        "periodic_distance": _entry(0.16, 0.11, 0.21),
    }
    assert select_backbone(clean) == "fc"


def test_backbone_gate_promotes_periodic_graph_when_cheaper_intervals_are_worse() -> None:
    clean = {
        "fc": _entry(0.02, -0.01, 0.05),
        "fc_distance": _entry(0.07, 0.04, 0.10),
        "periodic_distance": _entry(0.18, 0.15, 0.21),
    }
    assert select_backbone(clean) == "periodic_distance"


def test_backbone_gate_requires_positive_held_out_gain() -> None:
    clean = {
        "fc": _entry(-0.10, -0.20, 0.00),
        "fc_distance": _entry(-0.05, -0.10, 0.00),
        "periodic_distance": _entry(0.0, -0.01, 0.01),
    }
    assert select_backbone(clean) is None
