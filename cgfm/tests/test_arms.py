"""Tests for the arm factory."""

from cgfm.arms import (ArmStochasticInterpolants, CELL_VELOCITY_ANNEALING_FACTOR,
                       POSITION_VELOCITY_ANNEALING_FACTOR)


def _annealing_factors(arm: ArmStochasticInterpolants) -> tuple[float, float]:
    """
    Read the inference velocity annealing factors of the position and cell interpolants of an arm.

    :param arm:
        The arm to inspect.
    :type arm: ArmStochasticInterpolants

    :return:
        The position and the cell velocity annealing factor.
    :rtype: tuple[float, float]
    """
    return (arm.get_stochastic_interpolant("pos").get_velocity_annealing_factor(),
            arm.get_stochastic_interpolant("cell").get_velocity_annealing_factor())


def test_annealing_factors_default_to_the_tuned_omatg_values():
    """The default sampler must stay exactly the one the arms were trained and compared under."""
    position, cell = _annealing_factors(ArmStochasticInterpolants(grouping_kind="none"))
    assert position == POSITION_VELOCITY_ANNEALING_FACTOR
    assert cell == CELL_VELOCITY_ANNEALING_FACTOR


def test_position_annealing_factor_is_configurable_without_touching_the_cell():
    """Retuning the overshoot for a bumped path must not change the cell sampler as a side effect."""
    position, cell = _annealing_factors(
        ArmStochasticInterpolants(grouping_kind="none", velocity_annealing_factor=6.79))
    assert position == 6.79
    assert cell == CELL_VELOCITY_ANNEALING_FACTOR


def test_annealing_factor_is_independent_of_the_path():
    """The factor is a knob of the sampler, so it must not depend on eta or on the grouping."""
    plain = ArmStochasticInterpolants(grouping_kind="none", velocity_annealing_factor=4.0)
    bumped = ArmStochasticInterpolants(grouping_kind="precomputed", eta=0.5, velocity_annealing_factor=4.0)
    assert _annealing_factors(plain) == _annealing_factors(bumped)
