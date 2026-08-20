"""Tests for the preregistered gate arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from motif_conditioning.scripts.census_report import (
    ERROR_TOLERANCE,
    NOISE_FLOOR_POINTS,
    collect,
    contrast,
    error_regression,
    information_axis,
    load_outcomes,
)


def write_outcomes(path: Path, matches: list[bool], position: float = 0.03,
                   cell: float = 4.0, arm: str = "M") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "arm": arm,
                "seed": 0,
                "draw": 0,
                "num_compositions": len(matches),
                "match_rate": float(np.mean(matches)),
                "position_target_mse": position,
                "cell_target_mse": cell,
                "outcomes": [
                    {
                        "index": index,
                        "match": bool(value),
                        "position_target_mse": position,
                        "cell_target_mse": cell,
                    }
                    for index, value in enumerate(matches)
                ],
            }
        )
    )
    return path


def test_the_contrast_is_paired_not_a_difference_of_rates(tmp_path: Path) -> None:
    """Two arms with identical rates but different winners are still distinguishable."""
    treatment = load_outcomes(write_outcomes(tmp_path / "a.json", [True, True, False, False]))
    control = load_outcomes(write_outcomes(tmp_path / "b.json", [False, False, True, True]))
    result = contrast(treatment, control)

    assert result["difference_points"] == pytest.approx(0.0)
    assert result["won"] == 2 and result["lost"] == 2


def test_the_contrast_counts_wins_and_losses(tmp_path: Path) -> None:
    treatment = load_outcomes(write_outcomes(tmp_path / "a.json", [True, True, True, False]))
    control = load_outcomes(write_outcomes(tmp_path / "b.json", [True, False, False, False]))
    result = contrast(treatment, control)

    assert result["won"] == 2
    assert result["lost"] == 0
    assert result["difference_points"] == pytest.approx(50.0)


def test_unpaired_arms_are_refused(tmp_path: Path) -> None:
    treatment = load_outcomes(write_outcomes(tmp_path / "a.json", [True, False, True]))
    control = load_outcomes(write_outcomes(tmp_path / "b.json", [True, False]))
    with pytest.raises(ValueError, match="not paired"):
        contrast(treatment, control)


def test_identical_arms_report_no_discordant_pairs(tmp_path: Path) -> None:
    matches = [True, False, True, True]
    treatment = load_outcomes(write_outcomes(tmp_path / "a.json", matches))
    control = load_outcomes(write_outcomes(tmp_path / "b.json", matches))
    result = contrast(treatment, control)

    assert (result["won"], result["lost"]) == (0, 0)
    assert result["mcnemar_p"] == 1.0
    assert not result["exceeds_noise_floor"]


def test_a_difference_inside_the_noise_floor_is_flagged(tmp_path: Path) -> None:
    """A half-point difference must not be presented as a finding."""
    treatment = load_outcomes(write_outcomes(tmp_path / "a.json", [True] * 51 + [False] * 949))
    control = load_outcomes(write_outcomes(tmp_path / "b.json", [True] * 46 + [False] * 954))
    result = contrast(treatment, control)

    assert 0.0 < result["difference_points"] < NOISE_FLOOR_POINTS
    assert not result["exceeds_noise_floor"]


def test_error_regression_is_relative_to_the_baseline(tmp_path: Path) -> None:
    treatment = load_outcomes(write_outcomes(tmp_path / "a.json", [True], position=0.033))
    control = load_outcomes(write_outcomes(tmp_path / "b.json", [True], position=0.030))
    change = error_regression(treatment, control)

    assert change["position_target_mse"] == pytest.approx(0.10)
    assert change["position_target_mse"] > ERROR_TOLERANCE


def test_collect_indexes_arms_by_vocabulary_and_guidance(tmp_path: Path) -> None:
    write_outcomes(tmp_path / "M_k32_seed0" / "eval" / "w1.0" / "OUTCOMES.json", [True, False])
    write_outcomes(tmp_path / "M_k32_seed0" / "eval" / "w2.0" / "OUTCOMES.json", [True, True])
    write_outcomes(tmp_path / "X_k32_seed0" / "eval" / "w1.0" / "OUTCOMES.json", [False, False])
    write_outcomes(tmp_path / "not_a_run" / "eval" / "w1.0" / "OUTCOMES.json", [True, True])

    found = collect(tmp_path)

    assert set(found) == {"M@K=32@w=1", "M@K=32@w=2", "X@K=32@w=1"}
    assert found["M@K=32@w=2"]["guidance_scale"] == 2.0
    assert found["X@K=32@w=1"]["prototypes"] == 32


def test_the_information_axis_is_read_from_the_precheck(tmp_path: Path) -> None:
    path = tmp_path / "precheck.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {"prototypes": 8, "census_information_beyond_composition": 0.34},
                    {"prototypes": 32, "census_information_beyond_composition": 0.52},
                ]
            }
        )
    )
    assert information_axis(path) == {8: 0.34, 32: 0.52}


def test_a_missing_precheck_leaves_the_dose_response_unscored(tmp_path: Path) -> None:
    assert information_axis(tmp_path / "absent.json") == {}
