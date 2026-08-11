"""Shared fixtures and batch builders for the coarse-to-fine tests."""

from typing import Optional, Sequence
import pytest
import torch
from torch_geometric.data import Batch
from omg.datamodule import OMGData
from cgfm.grouping import GROUP_FIELD, NUM_GROUPS_FIELD, REFERENCE_METHODS, reference_field


@pytest.fixture(autouse=True)
def deterministic_environment():
    """
    Run every test in double precision from a fixed global random state.

    The interpolant is exercised with tight tolerances, and the tests are cheap enough that there is no reason to fight
    single-precision noise while checking algebraic identities.

    Seeding matters because module constructors draw their initial weights from the global generator. Without it a test
    would depend on how much randomness the tests before it happened to consume, so adding an unrelated test elsewhere
    could turn this one red.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


def make_structure(num_atoms: int, num_groups: Optional[int] = None, generator: Optional[torch.Generator] = None,
                   cell_scale: float = 6.0) -> OMGData:
    """
    Build one random structure with fractional coordinates and, optionally, a random balanced partition.

    :param num_atoms:
        Number of atoms.
    :type num_atoms: int
    :param num_groups:
        Number of groups, or None to leave the partition off the structure.
        Defaults to None.
    :type num_groups: Optional[int]
    :param generator:
        Random generator, for reproducibility.
        Defaults to None.
    :type generator: Optional[torch.Generator]
    :param cell_scale:
        Length scale of the cubic cell, in Angstrom.
        Defaults to 6.0.
    :type cell_scale: float

    :return:
        The structure.
    :rtype: OMGData
    """
    data = OMGData()
    data.n_atoms = torch.tensor(num_atoms)
    data.species = torch.randint(1, 90, (num_atoms,), generator=generator)
    data.cell = (cell_scale * torch.eye(3)).unsqueeze(0)
    data.pos = torch.rand((num_atoms, 3), generator=generator)
    data.pos_is_fractional = torch.tensor(True)
    data.property = {}
    if num_groups is not None:
        # Every group is used at least once, and the remaining atoms are spread at random.
        labels = torch.cat([torch.arange(num_groups),
                            torch.randint(0, num_groups, (num_atoms - num_groups,), generator=generator)])
        labels = labels[torch.randperm(num_atoms, generator=generator)]
        setattr(data, GROUP_FIELD, labels)
        setattr(data, NUM_GROUPS_FIELD, torch.tensor(num_groups))
        # CGOMGDataset attaches both fixed partitions to every structure. The tests do not need them to differ, so
        # they mirror the partition that defines the path.
        for method in REFERENCE_METHODS:
            setattr(data, reference_field(method), labels.clone())
    return data


def make_batch(sizes: Sequence[int], group_counts: Optional[Sequence[int]] = None, seed: int = 0) -> Batch:
    """
    Build a batch of random structures.

    :param sizes:
        Number of atoms of every structure.
    :type sizes: Sequence[int]
    :param group_counts:
        Number of groups of every structure, or None to leave partitions off.
        Defaults to None.
    :type group_counts: Optional[Sequence[int]]
    :param seed:
        Seed of the random generator.
        Defaults to 0.
    :type seed: int

    :return:
        The batched structures.
    :rtype: torch_geometric.data.Batch
    """
    generator = torch.Generator().manual_seed(seed)
    if group_counts is None:
        group_counts = [None] * len(sizes)
    return Batch.from_data_list(
        [make_structure(size, groups, generator) for size, groups in zip(sizes, group_counts)])
