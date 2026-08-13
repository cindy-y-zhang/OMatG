"""Tests for recovering atom positions from block poses without the sharing graph."""

from collections import Counter
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure as PymatgenStructure
from cgfm.blocks import Block, Decomposition, Template, decompose, fit_templates
from cgfm.periodic import Cell
from cgfm.readout import (Placement, _cluster, cast_votes, displacement, oracle_placement, orphan_free, read_out,
                          read_out_with_graph, read_out_with_oracle_clusters)


CUBE = 6.0 * np.eye(3)
"""Cell of the hand-built decompositions."""

CHAIN = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
"""
Two sodium and two chlorine atoms on a line, with the second chlorine in no polyhedron.

The spacings are deliberately uneven. At equal spacings the orphan sits exactly half a cell from both candidate hosts,
which is both a tie between them and the one separation at which the minimum image is ambiguous, so a reconstruction
that placed the atom on the wrong side would still look right.
"""

CHAIN_NUMBERS = [11, 11, 17, 17]
"""Atomic numbers of CHAIN, which charge balance splits into sodium centres and chlorine vertices."""


def hand_built(coords: np.ndarray, numbers: list[int], blocks: tuple[Block, ...]) -> Decomposition:
    """
    Build a decomposition directly, so that a case the neighbour finder rarely produces can still be tested.

    :param coords:
        Cartesian coordinates of shape (N, 3).
    :type coords: numpy.ndarray
    :param numbers:
        Atomic numbers of the atoms.
    :type numbers: list[int]
    :param blocks:
        The blocks covering the structure.
    :type blocks: tuple[Block, ...]

    :return:
        The decomposition.
    :rtype: Decomposition
    """
    return Decomposition(identifier="hand built", lattice=CUBE, coords=coords,
                         numbers=np.array(numbers, dtype=np.int64), blocks=blocks,
                         num_singletons=sum(1 for block in blocks if len(block.ligands) == 0),
                         centre_rule="charge balance")


def sodium_chloride() -> PymatgenStructure:
    """
    Build a rocksalt cell, whose octahedra are all identical so a type template describes them exactly.

    :return:
        The structure, holding four sodium and four chlorine atoms.
    :rtype: pymatgen.core.Structure
    """
    return PymatgenStructure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
                                             [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="module")
def rocksalt() -> tuple[Decomposition, dict]:
    """
    Decompose a rocksalt cell and fit templates to it, once for every test that needs a real decomposition.

    :return:
        The repaired decomposition and its templates.
    :rtype: tuple[Decomposition, dict]
    """
    decomposition = orphan_free(decompose(sodium_chloride(), identifier="NaCl"))
    templates = fit_templates({block.type_key: [(block.offsets, block.species)]
                               for block in decomposition.blocks if len(block.ligands) > 0})
    return decomposition, templates


def test_oracle_clusters_recover_a_rocksalt_cell_exactly(rocksalt):
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates)
    result = read_out_with_oracle_clusters(placement, decomposition, templates)

    assert result.shortfall == 0
    assert Counter(int(n) for n in result.numbers) == Counter(int(n) for n in decomposition.numbers)
    assert displacement(result, decomposition) < 1e-8


def test_oracle_clusters_beat_the_clustering_they_are_the_ceiling_of(rocksalt):
    # The whole reason the oracle column exists: it can never be worse than the readout that has to work the grouping
    # out, so a run where it is would mean the two are not measuring the same votes.
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates, translation_sigma=0.15, rotation_sigma=8.0,
                                    rng=np.random.default_rng(0))
    oracle = read_out_with_oracle_clusters(placement, decomposition, templates)
    free = read_out(placement, templates, decomposition.numbers)
    assert displacement(oracle, decomposition) <= displacement(free, decomposition) + 1e-12


def test_oracle_clusters_count_an_atom_no_vote_reached(rocksalt):
    # Explicit empty templates for every type leave every non-centre atom uncovered. lookup_template would otherwise
    # supply a regular polyhedron, which is the train-only last resort and is not what this column is measuring.
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates)
    empty = {key: Template(offsets=np.empty((0, 3)), species=(), count=0, spread=0.0) for key in placement.type_keys}
    result = read_out_with_oracle_clusters(placement, decomposition, empty)

    unknown = sum(1 for number in decomposition.numbers if int(number) == 17)
    assert result.shortfall == unknown
    assert Counter(int(n) for n in result.numbers) == Counter(int(n) for n in decomposition.numbers)


def test_oracle_clusters_do_not_place_an_uncovered_atom_at_its_true_position(rocksalt):
    # Placing it at the truth would let a readout that reached none of the atoms still match perfectly.
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates)
    empty = {key: Template(offsets=np.empty((0, 3)), species=(), count=0, spread=0.0) for key in placement.type_keys}
    result = read_out_with_oracle_clusters(placement, decomposition, empty)
    assert displacement(result, decomposition) > 1.0


def test_periodic_mean_averages_across_a_cell_boundary():
    points = np.array([[0.1, 0.0, 0.0], [5.9, 0.0, 0.0]])
    mean = Cell.of(CUBE).mean(points)
    assert np.isclose(np.abs(mean[0]) % 6.0, 0.0, atol=1e-9) or np.isclose(np.abs(mean[0]) % 6.0, 6.0, atol=1e-9)


def test_cluster_returns_exactly_the_requested_number_of_clusters():
    points = np.random.default_rng(0).uniform(0.0, 6.0, size=(20, 3))
    for count in (1, 3, 7, 20):
        labels = _cluster(points, count, Cell.of(CUBE))
        assert len(np.unique(labels)) == count
        assert set(np.unique(labels)) == set(range(count))


def test_cluster_separates_tight_well_separated_groups():
    centres = np.array([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [1.0, 4.0, 1.0]])
    rng = np.random.default_rng(0)
    points = np.concatenate([centre + rng.normal(scale=0.05, size=(5, 3)) for centre in centres])
    labels = _cluster(points, 3, Cell.of(CUBE))
    for group in range(3):
        assert len(np.unique(labels[5 * group:5 * (group + 1)])) == 1
    assert len(np.unique(labels)) == 3


def test_cluster_rejects_more_clusters_than_points():
    with pytest.raises(ValueError):
        _cluster(np.zeros((3, 3)), 4, Cell.of(CUBE))


def chain_blocks() -> tuple[Block, ...]:
    """
    Cover CHAIN with a single polyhedron, leaving one chlorine and one sodium for the repair to deal with.

    :return:
        The blocks.
    :rtype: tuple[Block, ...]
    """
    return (Block(centre=0, ligands=np.array([2]), offsets=CHAIN[[2]] - CHAIN[0], species=("Cl",),
                  type_key=("Na", 1, ("Cl",))),)


def test_orphan_free_gives_one_block_per_centre_atom():
    blocks = chain_blocks() + (Block(centre=3, ligands=np.empty(0, dtype=np.int64), offsets=np.empty((0, 3)),
                                     species=(), type_key=("Cl", 0)),)
    repaired = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, blocks))

    assert len(repaired.blocks) == 2
    assert sorted(block.centre for block in repaired.blocks) == [0, 1]


def test_orphan_free_claims_every_non_centre_atom():
    repaired = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, chain_blocks()))

    assert {int(index) for block in repaired.blocks for index in block.ligands} == {2, 3}
    hosts = {block.centre: {int(index) for index in block.ligands} for block in repaired.blocks}
    assert hosts[1] == {3}


def test_orphan_free_places_an_attached_atom_at_its_true_position():
    repaired = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, chain_blocks()))

    for block in repaired.blocks:
        for index, offset in zip(block.ligands, block.offsets):
            separation = (CHAIN[index] - (CHAIN[block.centre] + offset)) @ np.linalg.inv(CUBE)
            assert np.allclose(separation - np.round(separation), 0.0, atol=1e-9)


def test_orphan_free_is_idempotent():
    once = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, chain_blocks()))
    twice = orphan_free(once)

    assert len(once.blocks) == len(twice.blocks)
    for first, second in zip(once.blocks, twice.blocks):
        assert first.centre == second.centre
        assert first.type_key == second.type_key
        assert np.array_equal(first.ligands, second.ligands)


def test_orphan_free_keeps_the_block_count_at_the_centre_count(rocksalt):
    decomposition, _ = rocksalt
    assert len(decomposition.blocks) == 4
    assert all(decomposition.numbers[block.centre] == 11 for block in decomposition.blocks)


def test_cast_votes_counts_every_template_vertex(rocksalt):
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates)
    positions, numbers, blocks, _probs = cast_votes(placement, templates)

    assert len(positions) == sum(len(templates[key].offsets) for key in placement.type_keys)
    assert set(numbers.tolist()) == {17}
    assert np.array_equal(np.bincount(blocks), np.array([len(templates[key].offsets)
                                                         for key in placement.type_keys]))


def test_read_out_separates_atoms_every_block_predicts_in_pairs():
    # Four blocks on a line, each predicting the two atoms flanking it, so every atom is predicted twice by two
    # different blocks and a clustering that folded a block's own pair together would show up as one vote per atom.
    lattice = 8.0 * np.eye(3)
    centres = np.array([[x, 0.0, 0.0] for x in (0.0, 2.0, 4.0, 6.0)])
    template = Template(offsets=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), species=("Cl", "Cl"), spread=0.0,
                        count=4)
    placement = Placement(lattice=lattice, centre_numbers=np.full(4, 11), centre_coords=centres,
                          rotations=np.eye(3)[None, :, :].repeat(4, axis=0), type_keys=(("Na", 2),) * 4)

    result = read_out(placement, {("Na", 2): template}, np.array([11, 11, 11, 11, 17, 17, 17, 17]))
    assert result.votes_per_atom == pytest.approx(2.0)
    assert result.cluster_spread < 1e-9


def test_oracle_placement_returns_one_pose_per_block(rocksalt):
    decomposition, templates = rocksalt
    placement, correspondences = oracle_placement(decomposition, templates)

    assert len(placement.centre_coords) == len(decomposition.blocks)
    assert placement.rotations.shape == (len(decomposition.blocks), 3, 3)
    assert len(correspondences) == len(decomposition.blocks)
    assert np.allclose(placement.rotations @ placement.rotations.transpose(0, 2, 1), np.eye(3), atol=1e-9)


def test_read_out_reproduces_the_target_composition(rocksalt):
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates)
    result = read_out(placement, templates, decomposition.numbers)

    assert Counter(result.numbers.tolist()) == Counter(decomposition.numbers.tolist())
    assert result.shortfall == 0


def test_read_out_recovers_a_rigid_structure_without_the_sharing_graph(rocksalt):
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates)
    result = read_out(placement, templates, decomposition.numbers)

    assert displacement(result, decomposition) < 1e-6
    # Each octahedron holds six vertices, but the cell edge is twice the bond length, so the two vertices on each axis
    # are one chlorine seen through opposite faces. Three distinct atoms per block is the merging working.
    assert result.votes_per_atom == pytest.approx(3.0)


def test_read_out_with_graph_recovers_a_rigid_structure(rocksalt):
    decomposition, templates = rocksalt
    placement, correspondences = oracle_placement(decomposition, templates)
    result = read_out_with_graph(placement, decomposition, templates, correspondences)

    assert displacement(result, decomposition) < 1e-6


def test_both_readouts_agree_when_the_poses_are_exact(rocksalt):
    decomposition, templates = rocksalt
    placement, correspondences = oracle_placement(decomposition, templates)

    free = read_out(placement, templates, decomposition.numbers)
    graph = read_out_with_graph(placement, decomposition, templates, correspondences)
    assert displacement(free, decomposition) == pytest.approx(displacement(graph, decomposition), abs=1e-6)


def test_read_out_degrades_with_pose_noise_but_keeps_the_composition(rocksalt):
    decomposition, templates = rocksalt
    errors = []
    for sigma in (0.0, 0.05, 0.2):
        placement, _ = oracle_placement(decomposition, templates, translation_sigma=sigma, rotation_sigma=5.0 * sigma,
                                        rng=np.random.default_rng(0))
        result = read_out(placement, templates, decomposition.numbers)
        assert Counter(result.numbers.tolist()) == Counter(decomposition.numbers.tolist())
        errors.append(displacement(result, decomposition))

    assert errors[0] < errors[1] < errors[2]


def test_read_out_reports_a_shortfall_when_no_block_predicts_an_atom():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    placement = Placement(lattice=CUBE, centre_numbers=np.array([11]), centre_coords=coords[[0]],
                          rotations=np.eye(3)[None, :, :], type_keys=(("Na", 0),))
    result = read_out(placement, {}, np.array([11, 17]))

    assert result.shortfall == 1
    assert Counter(result.numbers.tolist()) == Counter([11, 17])


def test_oracle_placement_rejects_noise_without_a_generator(rocksalt):
    decomposition, templates = rocksalt
    with pytest.raises(ValueError):
        oracle_placement(decomposition, templates, translation_sigma=0.1)


def test_displacement_rejects_a_different_composition(rocksalt):
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates)
    result = read_out(placement, templates, decomposition.numbers)
    other = Decomposition(identifier="other", lattice=decomposition.lattice, coords=decomposition.coords,
                          numbers=np.full(len(decomposition.numbers), 11, dtype=np.int64),
                          blocks=decomposition.blocks, num_singletons=0, centre_rule="elemental")

    with pytest.raises(ValueError):
        displacement(result, other)


def test_read_out_rejects_centres_the_composition_does_not_hold(rocksalt):
    decomposition, templates = rocksalt
    placement, _ = oracle_placement(decomposition, templates)

    with pytest.raises(ValueError):
        read_out(placement, templates, np.array([17, 17, 17, 17]))


def test_graph_readout_handles_an_uncovered_atom_without_nan():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    numbers = np.array([11, 17], dtype=np.int64)
    blocks = (Block(centre=0, ligands=np.empty(0, dtype=np.int64), offsets=np.empty((0, 3)), species=(),
                    type_key=("Na", 0)),)
    decomposition = hand_built(coords, numbers.tolist(), blocks)
    placement = Placement(lattice=CUBE, centre_numbers=np.array([11]), centre_coords=coords[[0]],
                          rotations=np.eye(3)[None, :, :], type_keys=(("Na", 0),))
    result = read_out_with_graph(placement, decomposition, {}, (np.empty(0, dtype=np.int64),))

    assert not np.isnan(result.coords).any()
    assert result.shortfall == 1
    assert Counter(int(n) for n in result.numbers) == Counter(int(n) for n in numbers)
    assert displacement(result, decomposition) > 1.0


def test_read_out_assigns_elements_from_slot_probabilities():
    lattice = 10.0 * np.eye(3)
    centres = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    oxygen = np.zeros((1, 101))
    oxygen[0, 8] = 1.0
    fluorine = np.zeros((1, 101))
    fluorine[0, 9] = 1.0
    templates = {
        ("Na", 1, ("O",)): Template(offsets=np.array([[1.5, 0.0, 0.0]]), species=("F",), count=1, spread=0.0,
                                    species_probs=oxygen, species_aware=False),
        ("Na", 1, ("F",)): Template(offsets=np.array([[1.5, 0.0, 0.0]]), species=("O",), count=1, spread=0.0,
                                    species_probs=fluorine, species_aware=False),
    }
    placement = Placement(lattice=lattice, centre_numbers=np.array([11, 11]), centre_coords=centres,
                          rotations=np.eye(3)[None, :, :].repeat(2, axis=0),
                          type_keys=(("Na", 1, ("O",)), ("Na", 1, ("F",))))
    result = read_out(placement, templates, np.array([11, 11, 8, 9]))
    ligands = result.numbers[2:]
    coords = result.coords[2:]
    oxygen_coord = coords[ligands == 8]
    fluorine_coord = coords[ligands == 9]
    assert len(oxygen_coord) == 1 and len(fluorine_coord) == 1
    assert oxygen_coord[0, 0] < fluorine_coord[0, 0]
