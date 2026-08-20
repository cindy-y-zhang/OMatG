"""Regression tests for the preregistered descriptor-gate verdict."""

from copy import deepcopy

import pytest

from joint_geometry.scripts.probe_descriptor import DEPTHS, REPRESENTATIONS, TASKS, verdict


def _passing_results() -> dict:
    full_information = {"coordination": 1.0, "geometry": 0.8}
    fractions = {"cn-rdf4": 0.95, "cn-rdf8": 0.98, "radial17": 1.0}
    tasks = {}
    for task in TASKS:
        floor = {
            depth: {
                "cross_entropy_bits": 2.0,
                "shape_given_coordination": 0.40,
            }
            for depth in DEPTHS
        }
        tasks[task] = {
            "floor": floor,
            "representations": {},
            "terminal": {},
            "shuffled": {},
        }
        for representation in REPRESENTATIONS:
            information = full_information[task] * fractions[representation]
            tasks[task]["representations"][representation] = {
                "linear": {
                    "information_bits": information / 2.0,
                    "shape_given_coordination": 0.45,
                },
                "two_layer": {
                    "information_bits": information,
                    "shape_given_coordination": 0.50,
                },
            }
            tasks[task]["terminal"][representation] = {
                depth: {"information_bits": 0.005} for depth in DEPTHS
            }
            tasks[task]["shuffled"][representation] = {
                depth: {"information_bits": 0.006} for depth in DEPTHS
            }

    invariant = {
        "translation_max_abs": 1.0e-7,
        "rotation_max_abs": 1.0e-7,
        "permutation_max_abs": 1.0e-7,
        "unit_cell_max_abs": 1.0e-7,
        "supercell_max_abs": 1.0e-7,
        "continuity_max_change_for_1e-5_angstrom": 1.0e-4,
        "finite": True,
        "unit_variance_max_error": 0.01,
    }
    prior = {
        "max_abs_mean": 0.005,
        "max_std_error": 0.006,
        "max_abs_clean_correlation": 0.007,
    }
    return {
        "tasks": tasks,
        "retrieval": {
            "cn-rdf4": {"top1_retention": 0.95, "mrr_retention": 0.96},
            "cn-rdf8": {"top1_retention": 0.98, "mrr_retention": 0.99},
            "radial17": {},
        },
        "invariant_audit": {
            representation: deepcopy(invariant) for representation in REPRESENTATIONS
        },
        "prior_audit": {
            representation: deepcopy(prior) for representation in REPRESENTATIONS
        },
    }


def test_descriptor_verdict_promotes_smallest_passing_representation() -> None:
    result = verdict(_passing_results())
    assert result["passed"]
    assert result["promoted"] == "cn-rdf4"
    assert result["retention"]["cn-rdf4"]["coordination"] == pytest.approx(0.95)
    assert result["shape_given_coordination"]["cn-rdf4"]["two_layer"][
        "gain_points"
    ] == pytest.approx(10.0)


def test_descriptor_verdict_uses_nested_fallback() -> None:
    results = _passing_results()
    for task in TASKS:
        results["tasks"][task]["representations"]["cn-rdf4"]["two_layer"][
            "information_bits"
        ] *= 0.5
    result = verdict(results)
    assert result["passed"]
    assert result["promoted"] == "cn-rdf8"
    assert not result["eligibility"]["cn-rdf4"]["passed"]


def test_descriptor_verdict_refuses_uninformative_full_representation() -> None:
    results = _passing_results()
    for task in TASKS:
        for representation in REPRESENTATIONS:
            results["tasks"][task]["representations"][representation]["linear"][
                "information_bits"
            ] = 0.0
    result = verdict(results)
    assert not result["passed"]
    assert result["promoted"] is None


@pytest.mark.parametrize("control", ("terminal", "shuffled"))
def test_descriptor_verdict_rejects_control_information(control: str) -> None:
    results = _passing_results()
    results["tasks"]["geometry"][control]["cn-rdf4"]["two_layer"][
        "information_bits"
    ] = 0.03
    result = verdict(results)
    assert not result["passed"]
    assert not result[f"{control}_passed"]
