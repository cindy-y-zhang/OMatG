"""Tests for paired local-screen statistics and stop gates."""

import pytest

from joint_geometry.scripts.check_memorization import memorization_statistics
from joint_geometry.scripts.local_report import (
    contrast,
    evidence_summary,
    paired_vectors,
    promotion_failures,
    sampler_jitter,
)


def report(arm: str, seed: int, matches: list[int]) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "draw": 0,
        "num_compositions": len(matches),
        "outcomes": [
            {"index": index, "match": bool(value)}
            for index, value in enumerate(matches)
        ],
    }


def test_paired_contrast_uses_structure_level_differences() -> None:
    reports = {
        ("J", 0, 0): report("J", 0, [1, 1, 0, 1]),
        ("D", 0, 0): report("D", 0, [0, 1, 0, 0]),
        ("J", 1, 0): report("J", 1, [1, 0, 1, 1]),
        ("D", 1, 0): report("D", 1, [0, 0, 0, 1]),
    }
    result = contrast(reports, "J", "D", bootstrap_draws=1000, seed=3)
    assert result is not None
    assert result["difference_points"] == 50.0
    assert result["mcnemar"]["better_only"] == 4
    assert result["mcnemar"]["worse_only"] == 0
    assert result["bootstrap_ci95"][0] >= 0.0


def test_paired_vectors_reject_different_structure_order() -> None:
    first = report("J", 0, [1, 0])
    second = report("D", 0, [0, 1])
    second["outcomes"][0]["index"] = 1
    second["outcomes"][1]["index"] = 0
    with pytest.raises(ValueError, match="different structure ordering"):
        paired_vectors(first, second)


def _passing_contrast(worse: str) -> dict:
    return {
        "better": "J",
        "worse": worse,
        "difference_points": 2.0,
        "seed_draw_differences_points": [2.0, 2.0, 2.0],
        "seed_differences_points": {"0": 2.0, "1": 2.0, "2": 2.0},
        "bootstrap_ci95": [0.2, 3.8],
    }


def test_promotion_failures_encode_all_preregistered_gates() -> None:
    contrasts = {
        arm: _passing_contrast(arm)
        for arm in ("D", "P", "H", "R")
    }
    assert promotion_failures(contrasts, [0, 1, 2], 0.25) == []

    contrasts["H"]["difference_points"] = 0.0
    failures = promotion_failures(contrasts, [0, 1], 0.20)
    assert "J is not directionally above H" in failures
    assert "fewer than three paired seeds are present" in failures
    assert "endpoint descriptor consistency improves by less than 25% over P" in failures


def test_sampler_jitter_keeps_draws_grouped_by_checkpoint() -> None:
    reports = {
        ("J", 0, 0): report("J", 0, [1, 0, 1, 0]),
        ("J", 0, 1): {
            **report("J", 0, [1, 1, 1, 0]),
            "draw": 1,
        },
    }
    result = sampler_jitter(reports)["J_seed0"]
    assert result["draws"] == [0, 1]
    assert result["match_rates"] == [0.5, 0.75]
    assert result["range"] == pytest.approx(0.25)


def test_memorization_gate_accepts_a_materially_falling_objective() -> None:
    result = memorization_statistics([1.0, 0.9, 0.7, 0.6], 0.20)
    assert result["passed"]
    assert result["relative_drop"] > 0.20


def test_memorization_gate_rejects_nonfinite_or_flat_traces() -> None:
    with pytest.raises(ValueError, match="fewer than two finite"):
        memorization_statistics([1.0, float("nan")], 0.20)
    assert not memorization_statistics([1.0, 1.0], 0.20)["passed"]


def test_local_report_binds_frozen_upstream_evidence(tmp_path) -> None:
    (tmp_path / "DESCRIPTOR-GATE.json").write_text(
        '{"verdict":{"passed":true,"promoted":"radial17"}}'
    )
    (tmp_path / "BACKBONE-GATE.json").write_text(
        '{"verdict":{"passed":true,"selected_backbone":"periodic_distance"}}'
    )
    (tmp_path / "GEOMETRY-WEIGHT.json").write_text('{"geometry_weight":0.123}')
    summary = evidence_summary(tmp_path)
    assert summary["descriptor"]["promoted"] == "radial17"
    assert summary["backbone"]["selected_backbone"] == "periodic_distance"
    assert summary["geometry_weight"]["geometry_weight"] == pytest.approx(0.123)
    assert len(summary["descriptor"]["sha256"]) == 64
