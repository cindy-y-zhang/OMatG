"""Checks for the geometry-trajectory sweep used to bound the oracle's value."""

import pytest
import torch

from joint_geometry.interpolants import OracleGeometryInterpolants
from joint_geometry.scripts.probe_geometry_paths import (
    DEFAULT_CONFIGS,
    check_interpolants,
    geometry_policy,
)


ANNEALING = 10.182659004291072
EARLY_TIME = 0.3


@pytest.fixture
def endpoints() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a reproducible prior draw and clean endpoint."""
    generator = torch.Generator().manual_seed(0)
    prior = torch.randn(8, 17, generator=generator)
    clean = torch.randn(8, 17, generator=generator)
    return prior, clean


def test_sweep_avoids_the_overriding_oracle_interpolant() -> None:
    # OracleGeometryInterpolants reassigns geometry_b after the model returns,
    # which would discard every policy and integrate the chord in all arms.
    assert not any(config.endswith("/O.yaml") for config in DEFAULT_CONFIGS)
    assert any(config.endswith("/J.yaml") for config in DEFAULT_CONFIGS)


def test_check_interpolants_rejects_the_oracle_and_accepts_the_deployable_path() -> None:
    oracle = OracleGeometryInterpolants.__new__(OracleGeometryInterpolants)
    with pytest.raises(ValueError, match="overwrites geometry_b"):
        check_interpolants(oracle)
    check_interpolants(object())


def test_static_policies_hold_their_declared_state(endpoints) -> None:
    prior, clean = endpoints
    for name, expected in (
        ("zero", torch.zeros_like(prior)),
        ("prior_noise", prior),
        ("clean", clean),
    ):
        initial, velocity = geometry_policy(name, prior, clean, ANNEALING, EARLY_TIME)
        torch.testing.assert_close(initial, expected)
        for time in (0.001, 0.5, 0.999):
            zero_velocity = velocity(initial, torch.tensor(time))
            torch.testing.assert_close(zero_velocity, torch.zeros_like(prior))


def test_chord_policy_reaches_the_clean_endpoint(endpoints) -> None:
    prior, clean = endpoints
    initial, velocity = geometry_policy("chord", prior, clean, ANNEALING, EARLY_TIME)
    torch.testing.assert_close(initial, prior)
    torch.testing.assert_close(initial + velocity(initial, torch.tensor(0.5)), clean)


def test_feedback_policy_self_corrects_towards_the_clean_endpoint(endpoints) -> None:
    prior, clean = endpoints
    _, velocity = geometry_policy("feedback", prior, clean, ANNEALING, EARLY_TIME)
    # From any state, one unit of remaining time lands exactly on the endpoint.
    for time in (0.0, 0.25, 0.9):
        state = prior + time * (clean - prior)
        remaining = 1.0 - time
        stepped = state + remaining * velocity(state, torch.tensor(time))
        torch.testing.assert_close(stepped, clean)


def test_early_policy_completes_by_its_deadline_then_holds(endpoints) -> None:
    prior, clean = endpoints
    _, velocity = geometry_policy("early", prior, clean, ANNEALING, EARLY_TIME)
    torch.testing.assert_close(
        prior + EARLY_TIME * velocity(prior, torch.tensor(0.1)), clean
    )
    torch.testing.assert_close(
        velocity(clean, torch.tensor(EARLY_TIME + 0.1)), torch.zeros_like(prior)
    )


def test_annealed_policy_matches_the_position_reparameterisation(endpoints) -> None:
    prior, clean = endpoints
    _, velocity = geometry_policy("annealed", prior, clean, ANNEALING, EARLY_TIME)
    # Integrating (1 + k t) / (1 + k / 2) over [0, 1] covers the chord exactly once.
    grid = torch.linspace(0.0, 1.0, 20001)
    speeds = torch.stack([velocity(prior, time)[0, 0] for time in grid])
    torch.testing.assert_close(
        torch.trapezoid(speeds, grid), (clean - prior)[0, 0], rtol=1e-4, atol=1e-6
    )
    # The point of the policy is that it accelerates rather than moving linearly.
    assert speeds[0] < speeds[-1]


def test_unknown_policy_is_rejected(endpoints) -> None:
    prior, clean = endpoints
    with pytest.raises(ValueError, match="Unknown geometry policy"):
        geometry_policy("nonexistent", prior, clean, ANNEALING, EARLY_TIME)
