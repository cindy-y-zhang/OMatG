"""Tests that the census reaches the model everywhere the model runs.

The design attaches the census to ``x_0`` and touches nothing in ``omg``. That works only
because both the training interpolation and the inference integration build their states
by cloning ``x_0``, and because cloning a named subset of fields keeps the rest. These
tests pin that behaviour down, since it is an assumption about someone else's code.
"""

from __future__ import annotations

import pytest
import torch

from motif_conditioning.data import CENSUS_FIELD, derangement
from motif_conditioning.sampler import MotifCensusSampler

from .conftest import census_batch


DIMENSION = 8


class StructuralStub:
    """Stands in for IndependentSampler: returns a state built without any census."""

    def __init__(self) -> None:
        self.calls = 0

    def sample_p_0(self, x_1):
        self.calls += 1
        sampled = census_batch(
            sizes=tuple(int(count) for count in x_1.n_atoms),
            dimension=DIMENSION,
            seed=99,
            with_census=False,
        )
        return sampled


def test_the_sampler_copies_the_census_onto_the_sampled_state() -> None:
    sampler = MotifCensusSampler(StructuralStub(), census_dimension=DIMENSION)
    x_1 = census_batch(dimension=DIMENSION)
    x_0 = sampler.sample_p_0(x_1)

    assert torch.equal(getattr(x_0, CENSUS_FIELD), getattr(x_1, CENSUS_FIELD))


def test_the_copied_census_does_not_alias_the_clean_batch() -> None:
    sampler = MotifCensusSampler(StructuralStub(), census_dimension=DIMENSION)
    x_1 = census_batch(dimension=DIMENSION)
    x_0 = sampler.sample_p_0(x_1)

    getattr(x_0, CENSUS_FIELD).zero_()
    assert not torch.equal(getattr(x_0, CENSUS_FIELD), getattr(x_1, CENSUS_FIELD))


def test_the_sampler_refuses_a_batch_without_a_census() -> None:
    sampler = MotifCensusSampler(StructuralStub(), census_dimension=DIMENSION)
    with pytest.raises(ValueError, match="no precomputed motif census"):
        sampler.sample_p_0(census_batch(dimension=DIMENSION, with_census=False))


def test_the_sampler_refuses_a_mis_sized_census() -> None:
    sampler = MotifCensusSampler(StructuralStub(), census_dimension=DIMENSION)
    with pytest.raises(ValueError, match="Expected a motif census of shape"):
        sampler.sample_p_0(census_batch(dimension=DIMENSION + 3))


def test_the_sampler_refuses_a_degenerate_width() -> None:
    with pytest.raises(ValueError, match="census_dimension must be positive"):
        MotifCensusSampler(StructuralStub(), census_dimension=0)


def test_a_census_survives_the_selective_clone_the_integrator_uses() -> None:
    """``StochasticInterpolants.integrate`` clones only its own data fields by name."""
    x_0 = census_batch(dimension=DIMENSION)
    cloned = x_0.clone("species", "pos", "cell")

    assert hasattr(cloned, CENSUS_FIELD)
    assert torch.equal(getattr(cloned, CENSUS_FIELD), getattr(x_0, CENSUS_FIELD))


def test_a_census_survives_a_full_clone_the_loss_path_uses() -> None:
    x_0 = census_batch(dimension=DIMENSION)
    cloned = x_0.clone()

    assert hasattr(cloned, CENSUS_FIELD)
    assert torch.equal(getattr(cloned, CENSUS_FIELD), getattr(x_0, CENSUS_FIELD))


def test_batching_stacks_one_census_row_per_structure() -> None:
    batch = census_batch(sizes=(4, 6, 5), dimension=DIMENSION)
    assert getattr(batch, CENSUS_FIELD).shape == (3, DIMENSION)


def test_batching_keeps_each_structures_census_with_its_own_structure() -> None:
    """A census must not be silently mixed across the batch by concatenation."""
    from torch_geometric.data import Batch, Data

    graphs = []
    for index, count in enumerate((4, 6, 5)):
        graphs.append(
            Data(
                n_atoms=torch.tensor(count),
                species=torch.ones(count, dtype=torch.long),
                pos=torch.zeros((count, 3)),
                cell=torch.eye(3).unsqueeze(0),
                pos_is_fractional=torch.tensor(True),
                **{CENSUS_FIELD: torch.full((1, DIMENSION), float(index))},
            )
        )
    batch = Batch.from_data_list(graphs)
    expected = torch.tensor([[0.0], [1.0], [2.0]]).repeat(1, DIMENSION)
    assert torch.equal(getattr(batch, CENSUS_FIELD), expected)


@pytest.mark.parametrize("count", [2, 3, 17, 1024])
def test_the_mismatched_control_never_pairs_a_structure_with_itself(count: int) -> None:
    permutation = derangement(count, seed=0)

    assert sorted(permutation.tolist()) == list(range(count))
    assert not (permutation == torch.arange(count).numpy()).any()


def test_the_mismatched_control_is_reproducible_and_seed_dependent() -> None:
    assert (derangement(64, seed=0) == derangement(64, seed=0)).all()
    assert not (derangement(64, seed=0) == derangement(64, seed=1)).all()


def test_the_mismatched_control_needs_something_to_mismatch() -> None:
    with pytest.raises(ValueError, match="at least two structures"):
        derangement(1, seed=0)
