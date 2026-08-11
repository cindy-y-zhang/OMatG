"""Tests for the offline partitions, their on-disk format and the diagnostics that compare them."""

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from sklearn.metrics import adjusted_rand_score
from cgfm.diagnostics import adjusted_rand_index, group_statistics
from cgfm.blur import one_hot_assignment
from cgfm.groupfile import GroupTable
from cgfm.kmedoids import min_image_distance_matrix, periodic_kmedoids
from cgfm.shells import coordination_shell_partition
from .conftest import make_batch


def _rock_salt() -> Structure:
    """
    Build a rock-salt sodium chloride cell, whose coordination shells are unambiguous octahedra.

    :return:
        The structure.
    :rtype: pymatgen.core.Structure
    """
    return Structure(lattice=Lattice.cubic(5.64), species=["Na", "Cl"], coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def test_min_image_distance_crosses_the_boundary():
    """Two atoms either side of a cell face must be close, not a cell length apart."""
    frac = np.array([[0.95, 0.5, 0.5], [0.05, 0.5, 0.5]])
    distances = min_image_distance_matrix(frac, 10.0 * np.eye(3))
    assert distances[0, 1] == pytest.approx(1.0)


def test_kmedoids_produces_the_requested_number_of_clusters():
    """Every requested cluster must be used, which the group file format relies on."""
    generator = np.random.default_rng(0)
    distances = min_image_distance_matrix(generator.random((20, 3)), 8.0 * np.eye(3))
    labels = periodic_kmedoids(distances, 4, seed=0)

    assert labels.shape == (20,)
    assert sorted(np.unique(labels)) == [0, 1, 2, 3]


def test_kmedoids_is_deterministic():
    """Partitions are cached to disk, so the same input and seed must always give the same partition."""
    generator = np.random.default_rng(1)
    distances = min_image_distance_matrix(generator.random((16, 3)), 7.0 * np.eye(3))
    assert np.array_equal(periodic_kmedoids(distances, 3, seed=7), periodic_kmedoids(distances, 3, seed=7))


def test_kmedoids_recovers_well_separated_clusters():
    """Two tight, distant clumps must come out as the two clusters."""
    frac = np.concatenate([np.full((5, 3), 0.1) + 0.005 * np.arange(5)[:, None],
                           np.full((5, 3), 0.6) + 0.005 * np.arange(5)[:, None]])
    labels = periodic_kmedoids(min_image_distance_matrix(frac, 20.0 * np.eye(3)), 2, seed=0)
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_kmedoids_rejects_impossible_cluster_counts():
    """Asking for more clusters than atoms is a programming error, not something to paper over."""
    distances = min_image_distance_matrix(np.random.default_rng(2).random((3, 3)), 5.0 * np.eye(3))
    with pytest.raises(ValueError):
        periodic_kmedoids(distances, 4, seed=0)


def test_coordination_shells_form_a_partition():
    """Every atom must land in exactly one shell, with consecutive labels."""
    structure = _rock_salt()
    distances = min_image_distance_matrix(structure.frac_coords, structure.lattice.matrix)
    labels = coordination_shell_partition(structure, distances)

    assert labels is not None
    assert labels.shape == (2,)
    assert labels.min() == 0
    assert sorted(np.unique(labels)) == list(range(len(np.unique(labels))))


def test_coordination_shells_group_a_polyhedron_together():
    """A cation and the anions it coordinates must land in the same group."""
    # Perovskite-like cell: the titanium at the centre octahedrally coordinates the three oxygens.
    structure = Structure(lattice=Lattice.cubic(3.9), species=["Ca", "Ti", "O", "O", "O"],
                          coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    distances = min_image_distance_matrix(structure.frac_coords, structure.lattice.matrix)
    labels = coordination_shell_partition(structure, distances)

    assert labels is not None
    assert len(set(labels[1:])) == 1
    assert len(np.unique(labels)) < len(structure)


def test_group_table_round_trips(tmp_path):
    """A partition written to disk must come back unchanged."""
    labels = [np.array([0, 0, 1]), np.array([0, 1, 2, 2])]
    table = GroupTable.from_labels(labels, ["mp-1", "mp-2"], method="shells")
    table.validate()
    table.save(tmp_path / "split.shells.npz")

    reloaded = GroupTable.load(tmp_path / "split.shells.npz")
    assert reloaded.method == "shells"
    assert len(reloaded) == 2
    assert np.array_equal(reloaded[1][0], labels[1])
    assert reloaded[1][1] == 3
    assert list(reloaded.identifiers) == ["mp-1", "mp-2"]


def test_group_table_rejects_gaps_in_the_labels():
    """A declared group that no atom uses would make the group count meaningless."""
    table = GroupTable(labels=np.array([0, 2]), offsets=np.array([0, 2]), num_groups=np.array([3]),
                       identifiers=np.array(["mp-1"]), method="broken")
    with pytest.raises(ValueError):
        table.validate()


def test_group_table_detects_a_reordered_split():
    """A partition paired with a reordered split has the right length, so only the identifiers can catch it."""
    table = GroupTable.from_labels([np.array([0, 0]), np.array([0, 1])], ["mp-1", "mp-2"], method="shells")
    table.check_against(["mp-1", "mp-2"], "the split")
    with pytest.raises(ValueError, match="disagrees"):
        table.check_against(["mp-2", "mp-1"], "the split")


def test_missing_group_file_is_reported_clearly(tmp_path):
    """The most common setup mistake is forgetting to precompute, so it must say so."""
    with pytest.raises(FileNotFoundError, match="precompute_groups"):
        GroupTable.load(tmp_path / "absent.npz")


def test_adjusted_rand_index_is_invariant_to_relabelling():
    """The collapse diagnostic must read one whenever two partitions agree, whatever the group order."""
    batch = torch.zeros(12, dtype=torch.long)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    relabelled = torch.tensor([2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1])

    assert float(adjusted_rand_index(labels, labels, batch, 1)) == pytest.approx(1.0)
    assert float(adjusted_rand_index(labels, relabelled, batch, 1)) == pytest.approx(1.0)


def test_adjusted_rand_index_matches_the_reference_implementation():
    """The batched implementation must agree with scikit-learn on every structure it averages over."""
    generator = np.random.default_rng(0)
    sizes = [9, 12, 7]
    first = [generator.integers(0, 3, size) for size in sizes]
    second = [generator.integers(0, 4, size) for size in sizes]
    batch = torch.tensor(sum(([structure] * size for structure, size in enumerate(sizes)), []))

    ours = adjusted_rand_index(torch.tensor(np.concatenate(first)), torch.tensor(np.concatenate(second)),
                               batch, len(sizes))
    expected = np.mean([adjusted_rand_score(a, b) for a, b in zip(first, second)])
    assert float(ours) == pytest.approx(float(expected))


def test_group_statistics_describe_the_partition():
    """The reported statistics must match the partition attached to the batch."""
    batch = make_batch([10, 6], [2, 3], seed=0)
    assignment = one_hot_assignment(batch.cg_group, int(batch.cg_n_groups.max()))
    statistics = group_statistics(assignment, batch.batch, 2, batch)

    assert statistics["cg_groups_per_structure"] == pytest.approx(2.5)
    assert statistics["cg_group_size_mean"] == pytest.approx(16.0 / 5.0)
    # Both comparisons are reported: collapse onto k-medoids and resemblance to the coordination shells.
    assert statistics["cg_ari_vs_kmedoids"] == pytest.approx(1.0)
    assert statistics["cg_ari_vs_shells"] == pytest.approx(1.0)
    assert statistics["cg_assignment_entropy"] == pytest.approx(0.0)
    assert statistics["cg_group_extent_max"] > 0.0
