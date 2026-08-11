"""Tests for the learned grouping network and its integration with Lightning."""

import pytest
import torch
from torch import nn
from cgfm.blur import coarse_fine_delta
from cgfm.grouper import AnchorMembershipGrouper, build_grouper
from cgfm.grouping import PrecomputedGrouping
from .conftest import make_batch


def test_assignment_is_row_stochastic():
    """Every atom must distribute exactly one unit of membership over the groups."""
    batch = make_batch([12, 8], [3, 2], seed=1)
    assignment = AnchorMembershipGrouper(hidden_dim=16, num_layers=2, num_basis=8).assignment(batch)

    assert assignment.shape == (20, 3)
    assert torch.allclose(assignment.sum(dim=1), torch.ones(20))
    assert (assignment >= 0.0).all()


def test_unused_group_columns_stay_empty():
    """A structure must never place mass in a group column beyond its own group count."""
    batch = make_batch([12, 8], [4, 2], seed=2)
    assignment = AnchorMembershipGrouper(hidden_dim=16, num_layers=2, num_basis=8).assignment(batch)

    second_structure = assignment[batch.batch == 1]
    assert torch.allclose(second_structure[:, 2:], torch.zeros_like(second_structure[:, 2:]))


def test_every_group_is_occupied():
    """Anchors are assigned to their own groups, so no group can come out empty."""
    batch = make_batch([14, 9], [4, 3], seed=3)
    assignment = AnchorMembershipGrouper(hidden_dim=16, num_layers=2, num_basis=8).assignment(batch)

    for structure, num_groups in enumerate(batch.cg_n_groups.tolist()):
        weight = assignment[batch.batch == structure].sum(dim=0)
        assert (weight[:num_groups] > 0.0).all()


def test_assignment_is_permutation_equivariant():
    """Relabelling the atoms of a structure must permute the assignment and nothing else."""
    batch = make_batch([10], [3], seed=4)
    grouper = AnchorMembershipGrouper(hidden_dim=16, num_layers=2, num_basis=8)
    grouper.eval()

    original = grouper.assignment(batch)
    permutation = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    shuffled = batch.clone()
    shuffled.pos = batch.pos[permutation]
    shuffled.species = batch.species[permutation]
    shuffled.cg_group = batch.cg_group[permutation]

    # Group column order is arbitrary, so equivariance is checked on the partition rather than on the matrix.
    original_pairs = original.argmax(dim=1)[permutation]
    shuffled_pairs = grouper.assignment(shuffled).argmax(dim=1)
    assert torch.equal(original_pairs.unsqueeze(0) == original_pairs.unsqueeze(1),
                       shuffled_pairs.unsqueeze(0) == shuffled_pairs.unsqueeze(1))


def test_gradients_reach_every_parameter():
    """The flow-matching loss must be able to train the whole grouping network, anchor scorer included."""
    batch = make_batch([12, 10], [3, 3], seed=5)
    grouper = AnchorMembershipGrouper(hidden_dim=16, num_layers=2, num_basis=8)

    delta = coarse_fine_delta(torch.randn_like(batch.pos), grouper.assignment(batch), batch.batch, 2)
    (delta ** 2).sum().backward()

    without_gradient = [name for name, parameter in grouper.named_parameters()
                        if parameter.grad is None or parameter.grad.abs().sum() == 0.0]
    assert without_gradient == []


def test_temperature_sharpens_the_assignment():
    """Annealing the temperature must move the soft assignment towards a partition."""
    batch = make_batch([12], [3], seed=6)
    grouper = AnchorMembershipGrouper(hidden_dim=16, num_layers=2, num_basis=8)

    with torch.no_grad():
        grouper.temperature = 1.0
        warm = float(grouper.assignment(batch).max(dim=1).values.mean())
        grouper.temperature = 0.01
        cold = float(grouper.assignment(batch).max(dim=1).values.mean())
    assert cold > warm


def test_build_grouper_selects_the_arm():
    """Configuration files select an arm by name, so the names must map to the right objects."""
    assert build_grouper("none") is None
    assert isinstance(build_grouper("precomputed"), PrecomputedGrouping)
    assert isinstance(build_grouper("learned", hidden_dim=8, num_layers=1, num_basis=4), nn.Module)
    with pytest.raises(ValueError, match="Unknown grouping kind"):
        build_grouper("motifs")
