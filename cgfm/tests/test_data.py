"""Tests for attaching precomputed partitions to OMatG batches."""

from pathlib import Path
import pickle
from typing import Optional
import lmdb
import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader
from omg.datamodule import StructureDataset
from cgfm.data import CGOMGDataset
from cgfm.groupfile import GroupTable
from cgfm.grouping import REFERENCE_METHODS, reference_field


def _write_dataset(directory: Path, sizes: list[int]) -> str:
    """
    Write a tiny LMDB dataset of random structures.

    :param directory:
        Directory in which the file is written.
    :type directory: Path
    :param sizes:
        Number of atoms of every structure.
    :type sizes: list[int]

    :return:
        Path of the written file.
    :rtype: str
    """
    path = directory / "split.lmdb"
    generator = torch.Generator().manual_seed(0)
    with (lmdb.Environment(str(path), subdir=False, map_size=int(1e8), lock=False) as environment,
          environment.begin(write=True) as transaction):
        for index, size in enumerate(sizes):
            transaction.put(str(index).encode(), pickle.dumps({
                "cell": 6.0 * torch.eye(3, dtype=torch.float64),
                "pos": 6.0 * torch.rand((size, 3), generator=generator, dtype=torch.float64),
                "atomic_numbers": torch.randint(1, 90, (size,), generator=generator, dtype=torch.int64),
                "identifier": f"test-{index}"}))
    return str(path)


def _write_groups(directory: Path, labels: list[np.ndarray], identifiers: Optional[list[str]] = None) -> str:
    """
    Write a group file for the given partitions.

    :param directory:
        Directory in which the file is written.
    :type directory: Path
    :param labels:
        Group labels of every structure.
    :type labels: list[numpy.ndarray]
    :param identifiers:
        Material identifiers, defaulting to those _write_dataset assigns.
    :type identifiers: Optional[list[str]]

    :return:
        Path of the file for the shells method, whose sibling for k-medoids is written alongside it.
    :rtype: str
    """
    if identifiers is None:
        identifiers = [f"test-{index}" for index in range(len(labels))]
    # Both fixed partitions of a split are always written together, and CGOMGDataset loads both so that the
    # diagnostics can compare a learned partition against each of them.
    for method in REFERENCE_METHODS:
        GroupTable.from_labels(labels, identifiers, method=method).save(directory / f"split.{method}.npz")
    return str(directory / "split.shells.npz")


def test_batching_keeps_group_labels_structure_local(tmp_path):
    """PyTorch Geometric must concatenate the labels without offsetting them, and stack the group counts."""
    data_path = _write_dataset(tmp_path, [4, 3])
    group_path = _write_groups(tmp_path, [np.array([0, 0, 1, 1]), np.array([0, 1, 2])])
    dataset = CGOMGDataset(StructureDataset(data_path, lazy_storage=False, floating_point_precision="64-true"),
                           group_path)

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert torch.equal(batch.cg_group, torch.tensor([0, 0, 1, 1, 0, 1, 2]))
    assert torch.equal(batch.cg_n_groups, torch.tensor([2, 3]))
    assert torch.equal(batch.n_atoms, torch.tensor([4, 3]))
    # Both fixed partitions travel with the batch so the learned arm can be compared against each of them.
    for method in REFERENCE_METHODS:
        assert torch.equal(getattr(batch, reference_field(method)), torch.tensor([0, 0, 1, 1, 0, 1, 2]))


def test_group_file_length_is_checked(tmp_path):
    """Group files are positional, so a file for a different split must be rejected immediately."""
    data_path = _write_dataset(tmp_path, [4, 3])
    group_path = _write_groups(tmp_path, [np.array([0, 0, 1, 1])])
    with pytest.raises(ValueError, match="covers 1 structures"):
        CGOMGDataset(StructureDataset(data_path, lazy_storage=False, floating_point_precision="64-true"), group_path)


def test_atom_count_mismatch_is_checked(tmp_path):
    """A file of the right length but the wrong contents must be caught on first use."""
    data_path = _write_dataset(tmp_path, [4, 3])
    group_path = _write_groups(tmp_path, [np.array([0, 0, 1, 1]), np.array([0, 1])])
    dataset = CGOMGDataset(StructureDataset(data_path, lazy_storage=False, floating_point_precision="64-true"),
                           group_path)
    with pytest.raises(ValueError, match="which has 3 atoms"):
        dataset.get(1)


def test_group_file_for_a_reordered_split_is_rejected(tmp_path):
    """A group file of the right length built from a reordered split must be caught by the identifiers."""
    data_path = _write_dataset(tmp_path, [4, 3])
    group_path = _write_groups(tmp_path, [np.array([0, 0, 1, 1]), np.array([0, 1, 2])],
                               identifiers=["test-1", "test-0"])
    with pytest.raises(ValueError, match="disagrees"):
        CGOMGDataset(StructureDataset(data_path, lazy_storage=False, floating_point_precision="64-true"), group_path)
