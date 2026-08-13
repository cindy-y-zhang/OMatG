"""Tests for the rigid-block decomposition and its reconstruction."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from scipy.spatial.transform import Rotation
from cgfm.blocks import (Block, Decomposition, Template, align, centre_elements, decompose, fit_templates, kabsch,
                         lookup_template, reconstruct, regular_template)


def _octahedron(bond: float = 2.0) -> np.ndarray:
    """
    Build the vertex offsets of a regular octahedron.

    :param bond:
        Distance from the centre to every vertex, in Angstrom.
        Defaults to 2.0.
    :type bond: float

    :return:
        Vertex offsets of shape (6, 3).
    :rtype: numpy.ndarray
    """
    return bond * np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                            [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])


def test_kabsch_recovers_a_known_rotation():
    """The alignment primitive has to be exact on rigid data, or every deviation it reports is its own."""
    rotation = Rotation.random(random_state=7).as_matrix()
    source = np.random.default_rng(0).normal(size=(8, 3))
    assert np.allclose(kabsch(source, source @ rotation.T), rotation)


def test_kabsch_never_reflects():
    """A rigid body moves on SO(3), so a mirrored target must not be fitted by a reflection."""
    source = np.random.default_rng(1).normal(size=(8, 3))
    rotation = kabsch(source, source * np.array([1.0, 1.0, -1.0]))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_align_places_a_rotated_and_permuted_octahedron_exactly():
    """Vertices arrive in an arbitrary order, so the correspondence has to be solved along with the rotation."""
    template = _octahedron()
    species = ("O",) * 6
    rotation = Rotation.random(random_state=3).as_matrix()
    permutation = np.random.default_rng(2).permutation(6)
    instance = (template @ rotation.T)[permutation]

    found, correspondence, deviation = align(template, instance, species, species)
    assert deviation == pytest.approx(0.0, abs=1.0e-8)
    # The octahedron has twenty-four rotational symmetries, so the pose is only defined up to its point group and the
    # recovered rotation need not be the one applied. What must hold is that the placed template lands on the instance.
    assert np.allclose(template @ found.T, instance[correspondence])


def test_align_recovers_the_pose_of_an_asymmetric_block():
    """Without a point-group degeneracy the rotation itself is identifiable and must be found exactly."""
    template = np.random.default_rng(11).normal(size=(5, 3)) * 2.0
    species = ("O",) * 5
    rotation = Rotation.random(random_state=12).as_matrix()
    permutation = np.random.default_rng(13).permutation(5)

    found, _, deviation = align(template, (template @ rotation.T)[permutation], species, species)
    assert deviation == pytest.approx(0.0, abs=1.0e-8)
    assert np.allclose(found, rotation)


def test_align_matches_only_vertices_of_the_same_element():
    """A template slot may not be filled by a vertex of a different element, whatever the geometry says."""
    template = _octahedron()
    template_species = ("O", "O", "O", "O", "F", "F")
    instance_species = ("F", "F", "O", "O", "O", "O")
    instance = template[[4, 5, 0, 1, 2, 3]]

    _, correspondence, deviation = align(template, instance, template_species, instance_species)
    assert deviation == pytest.approx(0.0, abs=1.0e-8)
    assert all(template_species[slot] == instance_species[vertex] for slot, vertex in enumerate(correspondence))


def test_align_falls_back_to_assignment_for_large_vertex_counts():
    """Beyond the enumeration limit the alternating solver takes over and still has to recover a rigid rotation."""
    generator = np.random.default_rng(4)
    template = generator.normal(size=(8, 3)) * 2.0
    species = ("O",) * 8
    rotation = Rotation.random(random_state=5).as_matrix()

    _, _, deviation = align(template, template @ rotation.T, species, species, permutation_limit=1)
    assert deviation == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.parametrize("seed", range(12))
def test_align_by_assignment_finds_what_enumeration_finds(seed: int):
    """
    The alternating solver handles one block in seven and only reaches a local optimum, so its starting rotations decide
    whether it is trustworthy. Distorted polyhedra are used rather than rigid ones because a rigid one is easy for any
    start; with random starts this comparison failed on 23 per cent of real blocks and by up to 2.1 Angstrom.
    """
    generator = np.random.default_rng(seed)
    template = generator.normal(size=(7, 3)) * 2.0
    species = ("O",) * 7
    instance = template @ Rotation.random(random_state=seed).as_matrix().T + 0.15 * generator.normal(size=(7, 3))

    _, _, enumerated = align(template, instance, species, species, permutation_limit=10 ** 9)
    _, _, assigned = align(template, instance, species, species, permutation_limit=0)
    assert assigned == pytest.approx(enumerated, abs=1.0e-9)


def test_align_rejects_mismatched_vertex_counts():
    """Silently aligning a template to a differently sized block would corrupt every downstream number."""
    with pytest.raises(ValueError):
        align(_octahedron(), _octahedron()[:5], ("O",) * 6, ("O",) * 5)


def test_fit_templates_averages_arbitrarily_oriented_copies_of_one_shape():
    """A type whose instances are rigid copies must fit a template of zero spread."""
    shape = _octahedron()
    species = ("O",) * 6
    instances = {("Ti", 6, species): [(shape @ Rotation.random(random_state=seed).as_matrix().T, species)
                                      for seed in range(12)]}

    template = fit_templates(instances)[("Ti", 6, species)]
    assert template.spread == pytest.approx(0.0, abs=1.0e-6)
    assert template.count == 12
    # The fitted shape is the octahedron in some frame, so its sorted vertex distances must be the original ones.
    assert np.allclose(np.sort(np.linalg.norm(template.offsets, axis=-1)), np.sort(np.linalg.norm(shape, axis=-1)))


def test_reconstruction_is_exact_when_the_template_is_the_true_geometry():
    """Oracle poses plus the block's own geometry must return the structure untouched, noise aside."""
    lattice = np.diag([8.0, 8.0, 8.0])
    coords = np.array([[4.0, 4.0, 4.0], [6.0, 4.0, 4.0], [2.0, 4.0, 4.0],
                       [4.0, 6.0, 4.0], [4.0, 2.0, 4.0], [4.0, 4.0, 6.0], [4.0, 4.0, 2.0]])
    ligands = np.arange(1, 7)
    block = Block(centre=0, ligands=ligands, offsets=coords[ligands] - coords[0], species=("O",) * 6,
                  type_key=("Ti", 6, ("O",) * 6))
    decomposition = Decomposition(identifier="test", lattice=lattice, coords=coords,
                                  numbers=np.array([22, 8, 8, 8, 8, 8, 8]), blocks=(block,), num_singletons=0,
                                  centre_rule="charge balance")
    templates = {block.type_key: Template(offsets=block.offsets, species=block.species, count=1, spread=0.0)}

    assert np.allclose(reconstruct(decomposition, templates), coords)


def test_reconstruction_places_a_vertex_shared_by_two_blocks_at_the_consensus():
    """Corner sharing is the case a partition cannot express, so the average of the two predictions must be taken."""
    lattice = np.diag([10.0, 10.0, 10.0])
    coords = np.array([[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    shared = np.array([2])
    left = Block(centre=0, ligands=shared, offsets=np.array([[3.0, 0.0, 0.0]]), species=("O",),
                 type_key=("A", 1, ("O",)))
    right = Block(centre=1, ligands=shared, offsets=np.array([[-3.0, 0.0, 0.0]]), species=("O",),
                  type_key=("B", 1, ("O",)))
    decomposition = Decomposition(identifier="test", lattice=lattice, coords=coords,
                                  numbers=np.array([22, 22, 8]), blocks=(left, right), num_singletons=0,
                                  centre_rule="charge balance")
    # Each template pulls the shared vertex half an Angstrom in the opposite direction, so the consensus is the truth.
    templates = {left.type_key: Template(offsets=np.array([[3.5, 0.0, 0.0]]), species=("O",), count=1, spread=0.0),
                 right.type_key: Template(offsets=np.array([[-3.5, 0.0, 0.0]]), species=("O",), count=1, spread=0.0)}

    rebuilt = reconstruct(decomposition, templates)
    assert np.allclose(rebuilt[2], coords[2])
    assert np.allclose(rebuilt[:2], coords[:2])


def test_reconstruction_resolves_vertices_that_are_distant_periodic_images():
    """A vertex bonded across the cell boundary must not be averaged as though it sat inside the cell."""
    lattice = np.diag([6.0, 6.0, 6.0])
    coords = np.array([[0.5, 3.0, 3.0], [5.5, 3.0, 3.0]])
    # The centre bonds to the image of site 1 at x = -0.5, one lattice vector away from its representative.
    block = Block(centre=0, ligands=np.array([1]), offsets=np.array([[-1.0, 0.0, 0.0]]), species=("O",),
                  type_key=("A", 1, ("O",)))
    decomposition = Decomposition(identifier="test", lattice=lattice, coords=coords, numbers=np.array([22, 8]),
                                  blocks=(block,), num_singletons=0, centre_rule="charge balance")
    templates = {block.type_key: Template(offsets=block.offsets, species=("O",), count=1, spread=0.0)}

    assert np.allclose(reconstruct(decomposition, templates), coords)


def test_pose_noise_needs_a_generator():
    """Reproducibility of the sensitivity curve depends on the generator being supplied, not created silently."""
    decomposition = Decomposition(identifier="test", lattice=np.eye(3) * 5.0, coords=np.zeros((1, 3)),
                                  numbers=np.array([8]),
                                  blocks=(Block(centre=0, ligands=np.empty(0, dtype=np.int64),
                                                offsets=np.empty((0, 3)), species=(), type_key=("O", 0)),),
                                  num_singletons=1, centre_rule="elemental")
    with pytest.raises(ValueError):
        reconstruct(decomposition, {}, translation_sigma=0.1)


def test_centre_elements_picks_the_cation_of_an_ionic_composition():
    """The centre count is what a model has to emit, so it must follow from the composition and nothing else."""
    centres, rule = centre_elements(("Ti", "O", "O"))
    assert centres == frozenset({"Ti"})
    assert rule == "charge balance"


def test_centre_elements_falls_back_to_electronegativity_when_charge_balance_fails():
    """LaNi5 assigns no neutral oxidation states, but nickel is 0.81 more electronegative, so lanthanum is the centre."""
    centres, rule = centre_elements(("La",) + ("Ni",) * 5)
    assert centres == frozenset({"La"})
    assert rule == "electronegativity"


def test_centre_elements_refuses_to_split_a_symmetric_alloy():
    """Iron is only 0.22 more electronegative than aluminium, which is too little to put the centres anywhere."""
    centres, rule = centre_elements(("Fe", "Fe", "Fe", "Al"))
    assert centres == frozenset({"Fe", "Al"})
    assert rule == "no polar split"


def test_centre_elements_makes_every_atom_a_centre_of_an_elemental_structure():
    """A single-element crystal has no cation the composition could point to, so compression is given up honestly."""
    centres, rule = centre_elements(("C",) * 8)
    assert centres == frozenset({"C"})
    assert rule == "elemental"


def test_centre_elements_splits_every_ionic_composition():
    """For a polar compound there must be an anion left over to be a vertex, or the decomposition is vacuous."""
    for composition in (("Na", "Cl"), ("Si", "O", "O"), ("Li", "Fe", "P", "O", "O", "O", "O")):
        centres, rule = centre_elements(composition)
        assert centres != frozenset(composition)
        assert rule in ("charge balance", "electronegativity")


def test_centre_elements_reads_only_the_composition():
    """Two orderings of the same atoms are the same composition and must give the same centres."""
    assert centre_elements(("Ti", "O", "O")) == centre_elements(("O", "Ti", "O"))


def test_decompose_builds_overlapping_blocks_covering_every_atom():
    """The decomposition has to cover the structure and, unlike the partition, keep the sharing."""
    structure = Structure(Lattice.cubic(4.0), ["Sr", "Ti", "O", "O", "O"],
                          [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    decomposition = decompose(structure, identifier="perovskite")
    assert decomposition is not None

    covered = {index for block in decomposition.blocks
               for index in [block.centre, *block.ligands.tolist()]}
    assert covered == set(range(len(structure)))
    memberships = sum(len(block.ligands) for block in decomposition.blocks)
    # Both cations coordinate the same oxygens, so the vertex slots outnumber the atoms they are drawn from.
    assert memberships > len(structure)


def test_decompose_turns_an_unbonded_atom_into_a_singleton_block():
    """An atom the neighbour finder bonds to nothing must still be covered, and must not become an empty polyhedron."""
    structure = Structure(Lattice.cubic(14.0), ["Ti", "O", "O", "Ar"],
                          [[0.0, 0.0, 0.0], [0.14, 0.0, 0.0], [0.0, 0.14, 0.0], [0.5, 0.5, 0.5]])
    decomposition = decompose(structure)
    assert decomposition is not None

    assert decomposition.num_singletons >= 1
    assert all(len(block.ligands) > 0 or block.type_key[1] == 0 for block in decomposition.blocks)
    covered = {index for block in decomposition.blocks for index in [block.centre, *block.ligands.tolist()]}
    assert covered == set(range(len(structure)))


def test_decompose_rejects_an_unknown_type_mode():
    """A typo in the type key would silently change what a template is shared across."""
    structure = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError):
        decompose(structure, type_key_mode="centre-only")


def test_align_species_blind_matches_across_elements():
    template = _octahedron()
    instance = template[[4, 5, 0, 1, 2, 3]]
    found, correspondence, deviation = align(template, instance, ("O",) * 6, ("F",) * 6, species_aware=False)
    assert deviation == pytest.approx(0.0, abs=1.0e-8)
    assert np.allclose(template @ found.T, instance[correspondence])


def test_fit_coarse_templates_learn_slot_probabilities():
    octahedron = _octahedron()
    oxygen = ("O",) * 6
    mixed = ("O", "O", "O", "O", "F", "F")
    instances = {("Ti", 6): [(octahedron, oxygen), (octahedron, mixed)]}
    template = fit_templates(instances, species_aware=False)[("Ti", 6)]
    assert template.species_aware is False
    assert template.species_probs is not None
    assert template.species_probs.shape == (6, 101)
    assert template.species_probs[:, 8].sum() > template.species_probs[:, 9].sum()


def test_lookup_template_never_returns_none_and_records_the_fallback():
    exact = {("Ti", 6): regular_template(6)}
    template, provenance = lookup_template(("Fe", 6), exact)
    assert provenance == "same-cn"
    assert len(template.offsets) == 6
    template, provenance = lookup_template(("Xe", 4), {})
    assert provenance == "regular"
    assert len(template.offsets) == 4
