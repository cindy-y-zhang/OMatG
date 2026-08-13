"""
Coordination-shell partition of a single crystal structure.

This is the arm that encodes the hypothesis directly: groups are the coordination polyhedra that CrystalNN identifies,
which is the same neighbour-finding machinery whose environment labels were the strongest oracle feature in the earlier
symmetry-prediction experiments.

CrystalNN gives overlapping shells, since an anion is typically shared between several cation polyhedra, but the
interpolant needs a partition. That is resolved in two steps.

Polyhedron centres are chosen greedily from the highest-coordinated atom downwards, skipping any atom already covered by
an accepted centre. This is a greedy dominating set of the bonding graph, so the centres are spread over the whole cell
and their count is set by the coordination numbers rather than by a chosen compression ratio.

Every remaining atom then joins the nearest centre whose coordination shell actually contains it, so group membership is
a real bond wherever one exists. The few atoms no centre bonds to fall back to their nearest centre. Peeling shells
whole instead, and leaving unclaimed atoms as singletons, was tried first and produced one large polyhedron surrounded
by more than half singleton groups, which is barely a coarse-graining at all.

Groups are sets of atoms in the cell, so two neighbours that are periodic images of the same site count once. That is
why the shells here hold about three atoms rather than the five the project specification assumed, and why the centres
form an independent set of roughly a third of the atoms rather than a sixth. The number of groups this produces is the
number every other coarse-grained arm is given, so the arms differ only in which atoms are grouped together and not in
how much the structure is compressed.
"""

from typing import Optional
import warnings
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Structure as PymatgenStructure


def decorate_with_oxidation_states(structure: PymatgenStructure) -> PymatgenStructure:
    """
    Return a copy of the structure annotated with per-site oxidation states, or the structure itself if that fails.

    CrystalNN weights cation-anion contacts more heavily when it knows the charges. On MPTS-52 this barely changes the
    neighbour counts, but the bond schema whose environment labels motivated this project decorates the same way, and
    the shells arm is only a faithful test of that result if it identifies neighbours by the same rule.

    :param structure:
        The crystal structure.
    :type structure: pymatgen.core.Structure

    :return:
        The decorated structure, or the original if no oxidation states could be assigned.
    :rtype: pymatgen.core.Structure
    """
    for assign in (lambda: BVAnalyzer().get_valences(structure),
                   lambda: [Composition(structure.composition.reduced_formula).oxi_state_guesses()[0]
                            [str(site.specie.symbol)] for site in structure]):
        try:
            decorated = structure.copy()
            decorated.add_oxidation_state_by_site([float(state) for state in assign()])
            return decorated
        except Exception:
            continue
    return structure


def _neighbour_indices(structure: PymatgenStructure) -> Optional[list[set[int]]]:
    """
    Find the CrystalNN neighbours of every site.

    :param structure:
        The crystal structure.
    :type structure: pymatgen.core.Structure

    :return:
        Site indices of the neighbours of every site, or None if the neighbour analysis failed.
    :rtype: Optional[list[set[int]]]
    """
    finder = CrystalNN()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decorated = decorate_with_oxidation_states(structure)
            return [{neighbour["site_index"] for neighbour in finder.get_nn_info(decorated, index)} - {index}
                    for index in range(len(decorated))]
    except (ValueError, RuntimeError, IndexError):
        # CrystalNN fails on pathological cells, for example ones whose Voronoi construction is degenerate.
        return None


def _choose_centres(neighbours: list[set[int]]) -> list[int]:
    """
    Choose polyhedron centres as a greedy dominating set of the bonding graph.

    :param neighbours:
        Site indices of the neighbours of every site.
    :type neighbours: list[set[int]]

    :return:
        Site indices of the chosen centres, in the order they were chosen.
    :rtype: list[int]
    """
    num_atoms = len(neighbours)
    covered = np.zeros(num_atoms, dtype=bool)
    centres = []
    # Highest coordination first, ties broken by site index so that the partition is a deterministic function of the
    # structure as it is stored.
    for candidate in sorted(range(num_atoms), key=lambda index: (-len(neighbours[index]), index)):
        if covered[candidate]:
            continue
        centres.append(candidate)
        covered[candidate] = True
        covered[list(neighbours[candidate])] = True
    return centres


def coordination_shell_partition(structure: PymatgenStructure, distances: np.ndarray) -> Optional[np.ndarray]:
    """
    Partition the atoms of one structure into non-overlapping coordination shells.

    :param structure:
        The crystal structure.
    :type structure: pymatgen.core.Structure
    :param distances:
        Matrix of minimum-image distances between the atoms, of shape (N, N).
    :type distances: numpy.ndarray

    :return:
        Group label of every atom, taking every value in [0, K) at least once, of shape (N,); or None if the
        neighbour analysis failed.
    :rtype: Optional[numpy.ndarray]
    """
    neighbours = _neighbour_indices(structure)
    if neighbours is None:
        return None

    centres = _choose_centres(neighbours)
    labels = np.full(len(neighbours), -1, dtype=np.int64)
    for label, centre in enumerate(centres):
        labels[centre] = label

    for atom in np.flatnonzero(labels < 0):
        bonded = [label for label, centre in enumerate(centres) if atom in neighbours[centre]]
        candidates = bonded if bonded else range(len(centres))
        labels[atom] = min(candidates, key=lambda label: distances[atom, centres[label]])
    return labels
