"""
Tests for the best-of-n scorer.

A run that matches nothing cannot tell a working scorer from a broken one, and an untrained smoke model matches
nothing, so the matching itself is checked here against structures whose answer is known by construction.
"""

import numpy as np
import pytest
from ase import Atoms
from omg.analysis import ValidAtoms
from cgfm.scripts.score import ATOM_COUNT_BINS, _best_rmsd, _initialise_worker, bin_by_atom_count, summarise


def _rocksalt(lattice_constant: float = 5.0, c_over_a: float = 1.0) -> Atoms:
    """
    Build a two-atom tetragonal structure.

    Note that pymatgen's StructureMatcher normalises volume, so a uniformly scaled cell still matches the original.
    Only the axis ratio makes a structure genuinely different, which is what c_over_a is for.

    :param lattice_constant:
        Length of the two equal cell edges in Angstrom.
        Defaults to 5.0.
    :type lattice_constant: float
    :param c_over_a:
        Ratio of the third cell edge to the other two. One is cubic.
        Defaults to 1.0.
    :type c_over_a: float

    :return:
        The structure.
    :rtype: ase.Atoms
    """
    cell = np.diag([lattice_constant, lattice_constant, lattice_constant * c_over_a])
    return Atoms(symbols=("Na", "Cl"), scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)], cell=cell, pbc=True)


def _validated(structures: list[Atoms]) -> list[ValidAtoms]:
    """
    Wrap structures as ValidAtoms without running the validity checks.

    :param structures:
        The structures.
    :type structures: list[ase.Atoms]

    :return:
        The wrapped structures.
    :rtype: list[ValidAtoms]
    """
    return ValidAtoms.get_valid_atoms(structures, skip_validation=True, number_cpus=1, enable_progress_bar=False)


def test_identical_structures_match_themselves():
    """A draw that reproduces the reference exactly must match it with essentially zero error."""
    reference = _validated([_rocksalt(), _rocksalt(c_over_a=1.5)])
    _initialise_worker([reference], reference)

    for index in range(2):
        rmsd = _best_rmsd(index, ltol=0.3, stol=0.5, angle_tol=10.0, check_reduced=True)
        assert rmsd is not None
        assert rmsd == pytest.approx(0.0, abs=1.0e-6)


def test_best_of_n_recovers_a_match_that_one_shot_misses():
    """The point of best-of-n is that a composition counts as solved if any draw finds it."""
    reference = _validated([_rocksalt(c_over_a=1.0), _rocksalt(c_over_a=1.5)])
    # The first draw only gets the second structure right, the second draw only the first.
    first_draw = _validated([_rocksalt(c_over_a=2.4), _rocksalt(c_over_a=1.5)])
    second_draw = _validated([_rocksalt(c_over_a=1.0), _rocksalt(c_over_a=2.4)])

    _initialise_worker([first_draw], reference)
    one_shot = [_best_rmsd(index, ltol=0.3, stol=0.5, angle_tol=10.0, check_reduced=True) for index in range(2)]
    assert summarise(one_shot, stol=0.5)["match_rate"] == pytest.approx(0.5)

    _initialise_worker([first_draw, second_draw], reference)
    best = [_best_rmsd(index, ltol=0.3, stol=0.5, angle_tol=10.0, check_reduced=True) for index in range(2)]
    assert summarise(best, stol=0.5)["match_rate"] == pytest.approx(1.0)


def test_summarise_penalises_unmatched_structures():
    """Unmatched structures are excluded from the RMSE but contribute stol to the corrected RMSE."""
    summary = summarise([0.1, None, 0.3], stol=0.5)
    assert summary["count"] == 3
    assert summary["match_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_rmse"] == pytest.approx(0.2)
    assert summary["mean_crmse"] == pytest.approx((0.1 + 0.5 + 0.3) / 3.0)


def test_summarise_handles_no_matches_at_all():
    """An untrained model matches nothing, which must give a reportable number rather than a division by zero."""
    summary = summarise([None, None], stol=0.5)
    assert summary["match_rate"] == 0.0
    assert summary["mean_crmse"] == pytest.approx(0.5)
    assert np.isnan(summary["mean_rmse"])


def test_atom_count_bins_partition_the_structures():
    """Every structure must land in exactly one bin, or the binned table would not add up."""
    atom_counts = [1, 10, 11, 20, 21, 36, 37, 52]
    binned = bin_by_atom_count([0.1] * len(atom_counts), atom_counts, stol=0.5)
    assert sum(summary["count"] for summary in binned.values()) == len(atom_counts)
    assert all(binned[f"{low}-{high}"]["count"] == 2 for low, high in ATOM_COUNT_BINS)
