"""Tests for the group-mean operator and the coarse-minus-fine displacement."""

import pytest
import torch
from torch_scatter import scatter_mean
from cgfm.blur import apply_group_mean, coarse_fine_delta, fine_energy_fraction, one_hot_assignment, structure_mean


def test_group_mean_matches_scatter_mean():
    """A one-hot assignment must reproduce a plain grouped average."""
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 2])
    values = torch.randn((7, 3))
    assignment = one_hot_assignment(labels, 3)

    result = apply_group_mean(values, assignment, batch, 2)
    expected = scatter_mean(values, batch * 3 + labels, dim=0)[batch * 3 + labels]
    assert torch.allclose(result, expected)


def test_group_mean_is_idempotent_and_mean_preserving():
    """For a hard partition the operator is a projector that preserves the per-structure mean."""
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    labels = torch.tensor([0, 1, 1, 0, 0, 1, 2, 2])
    values = torch.randn((8, 3))
    assignment = one_hot_assignment(labels, 3)

    once = apply_group_mean(values, assignment, batch, 2)
    twice = apply_group_mean(once, assignment, batch, 2)
    assert torch.allclose(once, twice)
    assert torch.allclose(structure_mean(once, batch, 2), structure_mean(values, batch, 2))


def test_group_mean_preserves_mean_for_soft_assignments():
    """Row-stochasticity alone is enough for the operator to preserve the per-structure mean."""
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    assignment = torch.softmax(torch.randn((6, 3)), dim=1)
    values = torch.randn((6, 3))

    blurred = apply_group_mean(values, assignment, batch, 2)
    assert torch.allclose(structure_mean(blurred, batch, 2), structure_mean(values, batch, 2))


def test_delta_is_centred():
    """The coarse-minus-fine displacement must have zero mean in every structure."""
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    labels = torch.tensor([0, 0, 1, 0, 1, 1, 2])
    displacement = torch.randn((7, 3))
    assignment = one_hot_assignment(labels, 3)

    delta = coarse_fine_delta(displacement, assignment, batch, 2)
    assert torch.allclose(structure_mean(delta, batch, 2), torch.zeros((2, 3)), atol=1e-12)


def test_unused_group_columns_do_not_leak():
    """Padding a batch with more group columns than a structure uses must not change its result."""
    batch = torch.tensor([0, 0, 0, 0])
    labels = torch.tensor([0, 0, 1, 1])
    displacement = torch.randn((4, 3))

    narrow = coarse_fine_delta(displacement, one_hot_assignment(labels, 2), batch, 1)
    wide = coarse_fine_delta(displacement, one_hot_assignment(labels, 7), batch, 1)
    assert torch.allclose(narrow, wide)


def test_structures_do_not_interact():
    """Group labels are local to a structure, so batching must not mix structures that share a label."""
    first = torch.randn((5, 3))
    second = torch.randn((4, 3))
    first_labels = torch.tensor([0, 0, 1, 1, 2])
    second_labels = torch.tensor([0, 1, 1, 1])

    separate = torch.cat([
        coarse_fine_delta(first, one_hot_assignment(first_labels, 3), torch.zeros(5, dtype=torch.long), 1),
        coarse_fine_delta(second, one_hot_assignment(second_labels, 3), torch.zeros(4, dtype=torch.long), 1)])
    together = coarse_fine_delta(
        torch.cat([first, second]), one_hot_assignment(torch.cat([first_labels, second_labels]), 3),
        torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(4, dtype=torch.long)]), 2)
    assert torch.allclose(separate, together)


def test_singleton_groups_give_zero_delta():
    """When every atom is its own group, the coarse and fine parts coincide and Delta is the centred displacement."""
    batch = torch.zeros(4, dtype=torch.long)
    displacement = torch.randn((4, 3))
    assignment = one_hot_assignment(torch.arange(4), 4)

    delta = coarse_fine_delta(displacement, assignment, batch, 1)
    assert torch.allclose(delta, displacement - displacement.mean(dim=0))
    assert fine_energy_fraction(displacement, assignment, batch, 1) == pytest.approx(0.0)


def test_single_group_gives_full_fine_energy():
    """When the whole structure is one group, all displacement energy is within the group."""
    batch = torch.zeros(5, dtype=torch.long)
    displacement = torch.randn((5, 3))
    assignment = one_hot_assignment(torch.zeros(5, dtype=torch.long), 1)

    fraction = fine_energy_fraction(displacement, assignment, batch, 1)
    centred = displacement - displacement.mean(dim=0)
    assert float(fraction) == pytest.approx(float((centred ** 2).sum() / (displacement ** 2).sum()))


def test_single_atom_structure_has_no_coarse_fine_split():
    """A one-atom structure cannot be split, so Delta must vanish and leave the path at the baseline."""
    batch = torch.zeros(1, dtype=torch.long)
    displacement = torch.randn((1, 3))
    assignment = one_hot_assignment(torch.zeros(1, dtype=torch.long), 1)

    assert torch.allclose(coarse_fine_delta(displacement, assignment, batch, 1), torch.zeros((1, 3)))
    assert fine_energy_fraction(displacement, assignment, batch, 1) == pytest.approx(0.0)


def test_gradients_reach_the_assignment():
    """The path must be differentiable with respect to a soft assignment."""
    batch = torch.tensor([0, 0, 0, 0])
    logits = torch.randn((4, 2), requires_grad=True)
    displacement = torch.randn((4, 3))

    delta = coarse_fine_delta(displacement, torch.softmax(logits, dim=1), batch, 1)
    (delta ** 2).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0
