"""Tests for the coarse-to-fine stochastic interpolant."""

import pytest
import torch
from omg.si.interpolants import PeriodicLinearInterpolant
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from cgfm.blur import apply_group_mean, one_hot_assignment, structure_mean
from cgfm.grouping import GROUP_FIELD, PrecomputedGrouping
from cgfm.interpolant import CoarseToFineStochasticInterpolant
from .conftest import make_batch


def _baseline() -> SingleStochasticInterpolant:
    """
    Build the stock OMatG interpolant that the atomwise arm must reproduce.

    :return:
        The stock periodic linear stochastic interpolant.
    :rtype: SingleStochasticInterpolant
    """
    return SingleStochasticInterpolant(interpolant=PeriodicLinearInterpolant(), gamma=None, epsilon=None,
                                       differential_equation_type="ODE", correct_center_of_mass_motion=True)


def test_zero_eta_reproduces_the_baseline_exactly():
    """The atomwise arm must be bit-for-bit identical to the stock periodic linear path."""
    batch = make_batch([6, 9], seed=1)
    x_0 = torch.rand_like(batch.pos)
    t = torch.rand_like(batch.pos)

    ours, _ = CoarseToFineStochasticInterpolant(eta=0.0).interpolate(t, x_0, batch.pos, batch.batch, aux=batch)
    theirs, _ = _baseline().interpolate(t, x_0, batch.pos, batch.batch)
    assert torch.equal(ours, theirs)


def test_zero_eta_reproduces_the_baseline_velocity_exactly():
    """The conditional velocity must also be bit-for-bit identical at eta = 0."""
    batch = make_batch([6, 9], seed=2)
    x_0 = torch.rand_like(batch.pos)
    t = torch.rand_like(batch.pos)

    ours = CoarseToFineStochasticInterpolant(eta=0.0).expected_velocity(t, x_0, batch.pos, batch.batch, aux=batch)
    theirs = PeriodicLinearInterpolant().interpolate_derivative(t, x_0, batch.pos)
    assert torch.equal(ours, theirs)


def test_endpoints_are_exact():
    """The bump vanishes at both endpoints, so the marginals are those of the baseline."""
    batch = make_batch([7, 5], [2, 2], seed=3)
    interpolant = CoarseToFineStochasticInterpolant(eta=0.9, grouping=PrecomputedGrouping())
    x_0 = torch.rand_like(batch.pos)

    at_zero, _ = interpolant.interpolate(torch.zeros_like(batch.pos), x_0, batch.pos, batch.batch, aux=batch)
    at_one, _ = interpolant.interpolate(torch.ones_like(batch.pos), x_0, batch.pos, batch.batch, aux=batch)
    corrector = interpolant.get_corrector()
    assert torch.allclose(at_zero, corrector.correct(x_0))
    assert torch.allclose(at_one, corrector.correct(batch.pos))


def test_displacement_uses_the_minimum_image():
    """The path must cross the periodic boundary rather than travel the long way round."""
    interpolant = CoarseToFineStochasticInterpolant(eta=0.0)
    x_0 = torch.full((1, 3), 0.95)
    x_1 = torch.full((1, 3), 0.05)
    batch_indices = torch.zeros(1, dtype=torch.long)

    velocity = interpolant.expected_velocity(torch.full((1, 3), 0.5), x_0, x_1, batch_indices)
    assert torch.allclose(velocity, torch.full((1, 3), 0.1))


def test_path_is_permutation_equivariant():
    """Relabelling the atoms of a structure must permute the path, not change it."""
    batch = make_batch([8], [3], seed=11)
    interpolant = CoarseToFineStochasticInterpolant(eta=0.5, grouping=PrecomputedGrouping())
    x_0 = torch.rand_like(batch.pos)
    t = torch.full_like(batch.pos, 0.4)
    permutation = torch.randperm(8, generator=torch.Generator().manual_seed(12))

    straight, _ = interpolant.interpolate(t, x_0, batch.pos, batch.batch, aux=batch)

    permuted = make_batch([8], [3], seed=11)
    permuted.pos = batch.pos[permutation]
    setattr(permuted, GROUP_FIELD, getattr(batch, GROUP_FIELD)[permutation])
    shuffled, _ = interpolant.interpolate(t, x_0[permutation], permuted.pos, permuted.batch, aux=permuted)

    assert torch.allclose(shuffled, straight[permutation])


def test_single_atom_structure_reduces_to_the_baseline():
    """With one atom there is nothing to coarse-grain, so the coarse-to-fine path must equal the baseline path."""
    batch = make_batch([1], [1], seed=13)
    x_0 = torch.rand_like(batch.pos)
    t = torch.full_like(batch.pos, 0.3)

    ours, _ = CoarseToFineStochasticInterpolant(
        eta=0.5, grouping=PrecomputedGrouping()).interpolate(t, x_0, batch.pos, batch.batch, aux=batch)
    theirs, _ = _baseline().interpolate(t, x_0, batch.pos, batch.batch)
    assert torch.allclose(ours, theirs)


def test_fine_component_follows_the_delayed_schedule():
    """For a hard partition the within-group displacement must advance on b(t) = t - eta s(t)."""
    batch = make_batch([8], [3], seed=4)
    eta, time = 0.5, 0.3
    interpolant = CoarseToFineStochasticInterpolant(eta=eta, grouping=PrecomputedGrouping())
    x_0 = torch.rand_like(batch.pos)
    t = torch.full_like(batch.pos, time)

    corrector = interpolant.get_corrector()
    x_0prime = corrector.correct(x_0)
    displacement = corrector.unwrap(x_0prime, batch.pos) - x_0prime
    assignment = one_hot_assignment(batch.cg_group, int(batch.cg_n_groups.max()))

    # Recomputed without wrapping so that the schedule is visible; the wrap is an isometry of the torus.
    x_t, _ = interpolant.interpolate(t, x_0, batch.pos, batch.batch, aux=batch)
    unwrapped = corrector.unwrap(x_0prime, x_t) - x_0prime

    fine = lambda values: values - apply_group_mean(values, assignment, batch.batch, 1)
    expected = (time - eta * time * (1.0 - time)) * fine(displacement)
    assert torch.allclose(fine(unwrapped), expected)


def test_coarse_component_follows_the_advanced_schedule():
    """For a hard partition the group centroids must advance on a(t) = t + eta s(t)."""
    batch = make_batch([8], [3], seed=5)
    eta, time = 0.5, 0.3
    interpolant = CoarseToFineStochasticInterpolant(eta=eta, grouping=PrecomputedGrouping())
    x_0 = torch.rand_like(batch.pos)
    t = torch.full_like(batch.pos, time)

    corrector = interpolant.get_corrector()
    x_0prime = corrector.correct(x_0)
    displacement = corrector.unwrap(x_0prime, batch.pos) - x_0prime
    assignment = one_hot_assignment(batch.cg_group, int(batch.cg_n_groups.max()))

    x_t, _ = interpolant.interpolate(t, x_0, batch.pos, batch.batch, aux=batch)
    unwrapped = corrector.unwrap(x_0prime, x_t) - x_0prime

    # Both sides are taken relative to the structure mean, which is the quotient the encoder works in and the only
    # place where the coarse schedule is meant to hold.
    coarse = lambda values: (apply_group_mean(values, assignment, batch.batch, 1)
                             - structure_mean(values, batch.batch, 1)[batch.batch])
    expected = (time + eta * time * (1.0 - time)) * coarse(displacement)
    assert torch.allclose(coarse(unwrapped), expected)


def test_path_does_not_translate_the_structure():
    """Switching the path on must not move the centre of mass relative to the baseline."""
    batch = make_batch([9, 4], [3, 2], seed=6)
    x_0 = torch.rand_like(batch.pos)
    t = torch.full_like(batch.pos, 0.4)

    with_groups = CoarseToFineStochasticInterpolant(eta=0.7, grouping=PrecomputedGrouping())
    corrector = with_groups.get_corrector()
    x_0prime = corrector.correct(x_0)

    ours, _ = with_groups.interpolate(t, x_0, batch.pos, batch.batch, aux=batch)
    theirs, _ = CoarseToFineStochasticInterpolant(eta=0.0).interpolate(t, x_0, batch.pos, batch.batch)
    displacement = lambda x: structure_mean(corrector.unwrap(x_0prime, x) - x_0prime, batch.batch, 2)
    assert torch.allclose(displacement(ours), displacement(theirs))


def test_loss_is_the_mean_squared_error():
    """The loss must be the explicit mean squared error rather than OMatG's expansion without the constant term."""
    batch = make_batch([6, 6], [2, 3], seed=7)
    interpolant = CoarseToFineStochasticInterpolant(eta=0.5, grouping=PrecomputedGrouping())
    x_0 = torch.rand_like(batch.pos)
    t = torch.full_like(batch.pos, 0.6)

    x_t, z = interpolant.interpolate(t, x_0, batch.pos, batch.batch, aux=batch)
    prediction = torch.randn_like(x_t)
    losses = interpolant.loss(lambda _: (prediction, None), t, x_0, batch.pos, x_t, z, batch.batch, aux=batch)

    velocity = interpolant.expected_velocity(t, x_0, batch.pos, batch.batch, aux=batch)
    assert float(losses["loss_b"]) == pytest.approx(float(torch.mean((prediction - velocity) ** 2)))


def test_loss_is_zero_for_a_perfect_prediction():
    """The explicit mean squared error must bottom out at zero, which is what makes it safe to learn the grouping."""
    batch = make_batch([6], [2], seed=8)
    interpolant = CoarseToFineStochasticInterpolant(eta=0.5, grouping=PrecomputedGrouping(),
                                                    correct_center_of_mass_motion=True)
    x_0 = torch.rand_like(batch.pos)
    t = torch.full_like(batch.pos, 0.25)

    x_t, z = interpolant.interpolate(t, x_0, batch.pos, batch.batch, aux=batch)
    velocity = interpolant.expected_velocity(t, x_0, batch.pos, batch.batch, aux=batch)
    velocity = velocity - structure_mean(velocity, batch.batch, 1)[batch.batch]
    losses = interpolant.loss(lambda _: (velocity, None), t, x_0, batch.pos, x_t, z, batch.batch, aux=batch)
    assert float(losses["loss_b"]) == pytest.approx(0.0, abs=1e-20)


def test_cached_delta_is_not_reused_across_batches():
    """A stale cache entry must never be consumed by the loss of a different batch."""
    first = make_batch([6], [2], seed=9)
    second = make_batch([6], [3], seed=10)
    interpolant = CoarseToFineStochasticInterpolant(eta=0.5, grouping=PrecomputedGrouping())
    x_0 = torch.rand_like(first.pos)
    t = torch.full_like(first.pos, 0.5)

    _, z = interpolant.interpolate(t, x_0, first.pos, first.batch, aux=first)
    x_t, _ = interpolant.interpolate(t, x_0, second.pos, second.batch, aux=second)
    losses = interpolant.loss(lambda _: (torch.zeros_like(x_t), None), t, x_0, second.pos, x_t, z, second.batch,
                              aux=second)
    velocity = interpolant.expected_velocity(t, x_0, second.pos, second.batch, aux=second)
    assert float(losses["loss_b"]) == pytest.approx(float(torch.mean(velocity ** 2)))


def test_non_monotonic_schedules_are_rejected():
    """An eta that would make the coarse or fine schedule run backwards is a configuration error."""
    with pytest.raises(ValueError, match="non-monotonic"):
        CoarseToFineStochasticInterpolant(eta=1.5, grouping=PrecomputedGrouping())


def test_grouping_and_eta_must_agree():
    """A grouping without a bump, or a bump without a grouping, is a configuration error."""
    with pytest.raises(ValueError):
        CoarseToFineStochasticInterpolant(eta=0.0, grouping=PrecomputedGrouping())
    with pytest.raises(ValueError):
        CoarseToFineStochasticInterpolant(eta=0.5)


def test_missing_group_labels_are_reported_clearly():
    """Running a coarse-grained arm on a batch without partitions must say what is wrong."""
    batch = make_batch([5], seed=11)
    interpolant = CoarseToFineStochasticInterpolant(eta=0.5, grouping=PrecomputedGrouping())
    with pytest.raises(ValueError, match="cg_group"):
        interpolant.interpolate(torch.full_like(batch.pos, 0.5), torch.rand_like(batch.pos), batch.pos, batch.batch,
                                aux=batch)
