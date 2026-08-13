"""
Rigid-block decomposition of a crystal into overlapping coordination polyhedra.

This module exists to measure the ceiling of a rigid-body parameterization before anything is built on top of one. The
coarse-to-fine interpolant expressed the motif hypothesis as a schedule warp on a hard partition, which cannot represent
a motif as a unit: there is no orientation, no template, and the group's shape is interpolated out of noise the whole
way. The parameterization that can represent it replaces the 3N atomic coordinates by M block poses in SE(3), one per
coordination polyhedron, exactly as rigid-body flow matching does for metal-organic and molecular crystals.

That substitution is only worth building if it is nearly lossless, and for inorganic crystals it is not obviously so, for
two reasons this module quantifies:

- coordination polyhedra are not rigid. Two octahedra of the same type differ by real distortions, and a type-shared
  template throws that difference away;
- coordination polyhedra share atoms. Corner, edge and face sharing means one anion is a vertex of several polyhedra, so
  the blocks disagree about where it goes and the disagreement has to be reconciled.

Both losses are measured by reconstruction under oracle poses. Every polyhedron is replaced by the canonical template of
its type, placed at the true centre with the rotation that best fits the true geometry, and shared atoms are averaged. No
generative model can do better than that, so the resulting match rate is an upper bound.

Blocks here are deliberately overlapping, unlike the partition in shells.py. Sharing is the structural information the
partition destroyed, and a rigid-body model does not need a partition: an atom may be a vertex of several blocks, and the
consensus between them is what places it.

One asymmetry in the decomposition is deliberate and worth stating, because getting it wrong makes the whole approach
circular. CrystalNN is used to find which atoms are the vertices of which polyhedron, and that is legitimate: it only
ever runs on a known training structure to build a target. The choice of which atoms are polyhedron *centres* is not
legitimate on the same terms, because it fixes how many poses the model has to emit, and the model has nothing but the
composition to go on. Centres are therefore chosen from the composition alone, by charge balance where a neutral
assignment of oxidation states exists and by electronegativity otherwise.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
from math import factorial, isnan
from typing import Optional
import warnings
import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Element, Structure as PymatgenStructure
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from omg.globals import MAX_ATOM_NUM
from .periodic import Cell
from .shells import decorate_with_oxidation_states


COORDINATION_CAP = 12
"""
Largest coordination number a block may carry, which is also the number of classes the discrete stage has to predict.

The cap exists so that the generative problem is a fixed, small classification rather than an open-ended count, and 12
is chosen because it is the coordination of a close-packed shell. It is almost free: once vertices that are themselves
block centres are excluded, no block in a hundred MPTS-52 validation structures exceeds it. Nearly all of the very high
coordination numbers seen before, up to 28, were cation-cation contacts in intermetallics, which the exclusion removes.
"""

CN_CLASSES = COORDINATION_CAP + 1
"""
Number of discrete coordination-number classes the generative model emits.

Class 0 is the mask token. Classes 1 through 13 store coordination numbers 0 through 12 as ``CN + 1``.
"""

EXCLUDE_CENTRE_VERTICES = True
"""
Whether a neighbour that is itself a block centre is kept as a vertex of another block. It should not be.

An atom's position is recovered in one of two ways: exactly, as the translation of the block it centres, or by consensus
over the vertex votes cast for it. Letting an atom be both puts it in both paths at once, and the readout then has to
cluster votes for atoms whose positions are already known, which mislabels the clusters of atoms whose positions are
not. Excluding centre vertices makes the two roles disjoint and, incidentally, is what makes COORDINATION_CAP free and
halves the fraction of blocks whose vertex correspondence needs the inexact solver.
"""

PERMUTATION_LIMIT = 5040
"""Largest number of species-consistent vertex correspondences that is enumerated exactly rather than by assignment."""

HUNGARIAN_ITERATIONS = 12
"""Largest number of alternating assignment and Kabsch rounds per start."""

MINIMUM_ELECTRONEGATIVITY_GAP = 0.4
"""
Smallest Pauling electronegativity gap at which a composition is still split into centres and vertices.

The gap only decides compositions for which no charge-balanced assignment of oxidation states exists, which in practice
means intermetallics. Below this gap the two elements are close enough that naming one of them the anion says nothing, so
the structure is treated like an elemental one and every atom becomes a centre. Fe3Al sits at 0.22 and is not split;
Mg2Si sits at 0.59 and is.
"""


@dataclass(frozen=True)
class Block:
    """
    One coordination polyhedron of one structure, as a centre atom and its image-resolved ligand vertices.

    A site index may appear twice in ``ligands`` when two periodic images of the same site are both bonded to the centre.
    Both are genuine vertices of the polyhedron, so both are kept.

    :param centre:
        Site index of the central atom.
    :type centre: int
    :param ligands:
        Site indices of the vertices, of shape (n,).
    :type ligands: numpy.ndarray
    :param offsets:
        Cartesian offsets of the vertices from the centre, of shape (n, 3).
    :type offsets: numpy.ndarray
    :param species:
        Element symbols of the vertices, aligned with ``ligands``.
    :type species: tuple[str, ...]
    :param type_key:
        Identifier of the block type, which is what a template is shared across.
    :type type_key: tuple
    """

    centre: int
    ligands: np.ndarray
    offsets: np.ndarray
    species: tuple[str, ...]
    type_key: tuple


@dataclass(frozen=True)
class Decomposition:
    """
    A whole structure written as overlapping rigid blocks.

    :param identifier:
        Material identifier of the structure.
    :type identifier: str
    :param lattice:
        Cell vectors of shape (3, 3), with row i the i-th lattice vector.
    :type lattice: numpy.ndarray
    :param coords:
        True Cartesian coordinates of shape (N, 3).
    :type coords: numpy.ndarray
    :param numbers:
        Atomic numbers of shape (N,).
    :type numbers: numpy.ndarray
    :param blocks:
        The blocks covering the structure. Every atom is a vertex or a centre of at least one block.
    :type blocks: tuple[Block, ...]
    :param num_singletons:
        Number of blocks that hold no vertices, which are the atoms no polyhedron reaches.
    :type num_singletons: int
    :param centre_rule:
        Name of the composition-only rule that chose the polyhedron centres.
    :type centre_rule: str
    """

    identifier: str
    lattice: np.ndarray
    coords: np.ndarray
    numbers: np.ndarray
    blocks: tuple[Block, ...]
    num_singletons: int
    centre_rule: str


@dataclass(frozen=True)
class Template:
    """
    The canonical rigid geometry of one block type.

    :param offsets:
        Vertex offsets from the centre in the template's own frame, of shape (n, 3).
    :type offsets: numpy.ndarray
    :param species:
        Majority element symbol of every vertex slot, aligned with ``offsets``. For a species-blind coarse template
        these labels are only a summary of ``species_probs``; alignment and the stabiliser ignore them.
    :type species: tuple[str, ...]
    :param count:
        Number of instances the template was averaged over.
    :type count: int
    :param spread:
        Root-mean-square deviation of those instances from the template after optimal alignment, in Angstrom.
    :type spread: float
    :param species_probs:
        Per-slot distribution over atomic numbers, of shape (n, MAX_ATOM_NUM + 1), learned from the aligned training
        instances. ``None`` is treated as a one-hot encoding of ``species``.
        Defaults to None.
    :type species_probs: Optional[numpy.ndarray]
    :param species_aware:
        Whether alignment, the stabiliser and the readout should treat vertex elements as hard labels. Coarse
        ``(centre, CN)`` templates are species-blind; fine templates whose composition uniquely determines the ligand
        element are species-aware.
        Defaults to True.
    :type species_aware: bool
    """

    offsets: np.ndarray
    species: tuple[str, ...]
    count: int
    spread: float
    species_probs: Optional[np.ndarray] = None
    species_aware: bool = True


def encode_coordination(coordination: int) -> int:
    """
    Map a coordination number in ``[0, COORDINATION_CAP]`` to a discrete class in ``[1, CN_CLASSES]``.

    :param coordination:
        Coordination number.
    :type coordination: int

    :return:
        The class index, with 0 reserved for the mask token.
    :rtype: int

    :raises ValueError:
        If the coordination number is outside the representable range.
    """
    if coordination < 0 or coordination > COORDINATION_CAP:
        raise ValueError(f"Coordination number {coordination} is outside [0, {COORDINATION_CAP}].")
    return coordination + 1


def decode_coordination(token: int) -> int:
    """
    Map a discrete class in ``[1, CN_CLASSES]`` back to a coordination number.

    :param token:
        Class index produced by ``encode_coordination``.
    :type token: int

    :return:
        The coordination number.
    :rtype: int

    :raises ValueError:
        If the token is the mask or lies outside the class range.
    """
    if token < 1 or token > CN_CLASSES:
        raise ValueError(f"Coordination token {token} is outside [1, {CN_CLASSES}].")
    return token - 1


def template_species_probs(template: Template) -> np.ndarray:
    """
    Return the per-slot element distribution of a template, one-hot encoding ``species`` when none was stored.

    :param template:
        The template.
    :type template: Template

    :return:
        Probabilities of shape (n, MAX_ATOM_NUM + 1).
    :rtype: numpy.ndarray
    """
    if template.species_probs is not None:
        return template.species_probs
    probabilities = np.zeros((len(template.offsets), MAX_ATOM_NUM + 1), dtype=np.float64)
    for slot, symbol in enumerate(template.species):
        probabilities[slot, Element(symbol).Z] = 1.0
    return probabilities


def regular_offsets(coordination: int, bond: float = 2.0) -> np.ndarray:
    """
    Build a regular polyhedron of the requested coordination number, used only as a last-resort fallback.

    :param coordination:
        Number of vertices.
    :type coordination: int
    :param bond:
        Distance from the centre to every vertex, in Angstrom.
        Defaults to 2.0.
    :type bond: float

    :return:
        Vertex offsets of shape (coordination, 3).
    :rtype: numpy.ndarray

    :raises ValueError:
        If the coordination number is outside ``[0, COORDINATION_CAP]``.
    """
    if coordination < 0 or coordination > COORDINATION_CAP:
        raise ValueError(f"Coordination number {coordination} is outside [0, {COORDINATION_CAP}].")
    if coordination == 0:
        return np.empty((0, 3), dtype=np.float64)
    if coordination == 1:
        return bond * np.array([[0.0, 0.0, 1.0]])
    if coordination == 2:
        return bond * np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    if coordination == 3:
        angles = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
        return bond * np.stack([np.cos(angles), np.sin(angles), np.zeros(3)], axis=-1)
    if coordination == 4:
        return bond * np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
                               ) / np.sqrt(3.0)
    if coordination == 5:
        equatorial = regular_offsets(3, bond)
        return np.concatenate([equatorial, bond * np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])])
    if coordination == 6:
        return bond * np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    if coordination == 8:
        signs = np.array([-1.0, 1.0])
        return bond * np.array(list(product(signs, signs, signs)), dtype=np.float64) / np.sqrt(3.0)
    if coordination == 12:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        cyclic = np.array([[0.0, 1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, phi], [0.0, -1.0, -phi],
                           [1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, phi, 0.0], [-1.0, -phi, 0.0],
                           [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0]])
        return bond * cyclic / np.linalg.norm(cyclic[0])
    # Even coordinations that are not platonic solids sit on a latitude ring plus poles, which is enough for a
    # placeable fallback whose vertex count matches the requested coordination number.
    poles = 2 if coordination >= 7 else 0
    ring = coordination - poles
    angles = np.linspace(0.0, 2.0 * np.pi, ring, endpoint=False)
    latitude = 0.5 if poles else 0.0
    radius = np.sqrt(max(1.0 - latitude * latitude, 1.0e-8))
    ring_points = bond * np.stack([radius * np.cos(angles), radius * np.sin(angles),
                                   np.full(ring, latitude)], axis=-1)
    if poles:
        return np.concatenate([ring_points, bond * np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])])
    return ring_points


def regular_template(coordination: int, bond: float = 2.0) -> Template:
    """
    Build a species-blind regular template of the requested coordination number.

    :param coordination:
        Number of vertices.
    :type coordination: int
    :param bond:
        Distance from the centre to every vertex, in Angstrom.
        Defaults to 2.0.
    :type bond: float

    :return:
        A template with no species evidence.
    :rtype: Template
    """
    offsets = regular_offsets(coordination, bond=bond)
    probabilities = np.zeros((len(offsets), MAX_ATOM_NUM + 1), dtype=np.float64)
    return Template(offsets=offsets, species=("X",) * len(offsets), count=0, spread=0.0,
                    species_probs=probabilities, species_aware=False)


def lookup_template(key: tuple, templates: dict[tuple, Template],
                    fallback: Optional[dict[tuple, Template]] = None) -> tuple[Template, str]:
    """
    Resolve a block type to a training-only template, never to an evaluation block's own geometry.

    The search order is the exact key, the coarse ``(centre, CN)`` key, the most common training template of the same
    coordination number, and finally a regular polyhedron of that coordination number. Every generated type therefore
    has a placeable template, and an unseen ``(centre, CN)`` pair is recorded rather than silently reconstructed from
    the structure being scored.

    :param key:
        Block type key, either ``(centre, CN)`` or ``(centre, CN, ligands)``.
    :type key: tuple
    :param templates:
        Primary template table, normally the train-only vocabulary of the same granularity as ``key``.
    :type templates: dict[tuple, Template]
    :param fallback:
        Secondary table consulted after the primary one, typically the coarse train-only vocabulary.
        Defaults to None.
    :type fallback: Optional[dict[tuple, Template]]

    :return:
        The template and a label of how it was resolved.
    :rtype: tuple[Template, str]
    """
    tables = (("exact", templates),) + ((("fallback-table", fallback),) if fallback is not None else ())
    for label, table in tables:
        if key in table:
            return table[key], label
    if len(key) > 2:
        coarse_key = (key[0], key[1])
        for label, table in tables:
            if coarse_key in table:
                return table[coarse_key], "coarse" if label == "exact" else "coarse-fallback"
    coordination = int(key[1]) if len(key) > 1 else 0
    pool = dict(templates)
    if fallback is not None:
        pool.update({key_: template for key_, template in fallback.items() if key_ not in pool})
    same_coordination = [key_ for key_ in pool if len(key_) >= 2 and int(key_[1]) == coordination]
    if same_coordination:
        best = max(same_coordination, key=lambda key_: pool[key_].count)
        return pool[best], "same-cn"
    return regular_template(coordination), "regular"


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Find the proper rotation that best maps one point set onto another about their common origin.

    Reflections are excluded because a rigid-body model moves on SO(3): the mirror image of a chiral polyhedron is a
    different environment and is not reachable by transporting the same block.

    :param source:
        Points to rotate, of shape (n, 3), taken relative to the rotation centre.
    :type source: numpy.ndarray
    :param target:
        Points to rotate onto, of shape (n, 3), taken relative to the same centre.
    :type target: numpy.ndarray

    :return:
        Rotation matrix R of shape (3, 3) minimising the sum of squared distances between R @ source[k] and target[k].
    :rtype: numpy.ndarray
    """
    if len(source) == 0:
        return np.eye(3)
    left, _, right = np.linalg.svd(source.T @ target)
    flip = np.sign(np.linalg.det(right.T @ left.T))
    return right.T @ np.diag([1.0, 1.0, flip]) @ left.T


def _rotate(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Apply a rotation to every point of a set stored as rows.

    :param points:
        Points of shape (n, 3).
    :type points: numpy.ndarray
    :param rotation:
        Rotation matrix of shape (3, 3).
    :type rotation: numpy.ndarray

    :return:
        Rotated points of shape (n, 3), with row k equal to rotation @ points[k].
    :rtype: numpy.ndarray
    """
    return points @ rotation.T


def _correspondence_groups(template_species: tuple[str, ...],
                           instance_species: tuple[str, ...]) -> list[tuple[list[int], list[int]]]:
    """
    Group template slots and instance vertices that may be matched to each other.

    Correspondences are restricted to vertices of the same element, which is both physically right and a large reduction
    of the search. When the two vertex compositions differ, which only happens for a block type that does not record
    ligand species, the restriction is dropped.

    :param template_species:
        Element symbols of the template vertices.
    :type template_species: tuple[str, ...]
    :param instance_species:
        Element symbols of the instance vertices.
    :type instance_species: tuple[str, ...]

    :return:
        Pairs of equal-length template slot indices and instance vertex indices that may be matched within.
    :rtype: list[tuple[list[int], list[int]]]
    """
    if sorted(template_species) != sorted(instance_species):
        indices = list(range(len(template_species)))
        return [(indices, list(range(len(instance_species))))]
    return [([slot for slot, symbol in enumerate(template_species) if symbol == species],
             [vertex for vertex, symbol in enumerate(instance_species) if symbol == species])
            for species in sorted(set(template_species))]


def _assign(rotated: np.ndarray, instance: np.ndarray,
            groups: list[tuple[list[int], list[int]]]) -> np.ndarray:
    """
    Match every template slot to an instance vertex of the same element at least cost.

    :param rotated:
        Template vertices already rotated into the instance frame, of shape (n, 3).
    :type rotated: numpy.ndarray
    :param instance:
        Instance vertices, of shape (n, 3).
    :type instance: numpy.ndarray
    :param groups:
        Template slots and instance vertices that may be matched within, from _correspondence_groups.
    :type groups: list[tuple[list[int], list[int]]]

    :return:
        Correspondence of shape (n,), where entry k is the instance vertex matched to template slot k.
    :rtype: numpy.ndarray
    """
    correspondence = np.empty(len(rotated), dtype=np.int64)
    for slots, vertices in groups:
        cost = np.linalg.norm(rotated[slots][:, None, :] - instance[vertices][None, :, :], axis=-1)
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            correspondence[slots[row]] = vertices[column]
    return correspondence


@lru_cache(maxsize=512)
def _enumerated_correspondences(groups: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
                               num_vertices: int) -> np.ndarray:
    """
    Enumerate every species-consistent correspondence between template slots and instance vertices.

    The result depends only on which slots and vertices share an element, not on any geometry, so it is cached across the
    many blocks of a dataset that have the same vertex composition.

    :param groups:
        Template slots and instance vertices that may be matched within, as hashable tuples.
    :type groups: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    :param num_vertices:
        Number of vertices n of the block.
    :type num_vertices: int

    :return:
        Every correspondence, of shape (P, n), where entry (p, k) is the instance vertex matched to template slot k.
    :rtype: numpy.ndarray
    """
    rows = []
    for choice in product(*(permutations(vertices) for _, vertices in groups)):
        correspondence = np.empty(num_vertices, dtype=np.int64)
        for (slots, _), assignment in zip(groups, choice):
            correspondence[list(slots)] = assignment
        rows.append(correspondence)
    return np.array(rows, dtype=np.int64)


def _best_enumerated(template: np.ndarray, instance: np.ndarray,
                     correspondences: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the rotation for every candidate correspondence at once and keep the best.

    Kabsch is a three-by-three singular value decomposition, so thousands of candidates are one batched decomposition
    rather than thousands of small ones. That is what makes exact enumeration affordable for the coordination numbers
    that allow it.

    :param template:
        Template vertices of shape (n, 3).
    :type template: numpy.ndarray
    :param instance:
        Instance vertices of shape (n, 3).
    :type instance: numpy.ndarray
    :param correspondences:
        Candidate correspondences of shape (P, n).
    :type correspondences: numpy.ndarray

    :return:
        Rotation of shape (3, 3), the winning correspondence of shape (n,), and the root-mean-square deviation in
        Angstrom.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, float]
    """
    targets = instance[correspondences]
    left, _, right = np.linalg.svd(np.einsum("ki,pkj->pij", template, targets))
    left_transposed, right_transposed = left.transpose(0, 2, 1), right.transpose(0, 2, 1)
    flip = np.sign(np.linalg.det(right_transposed @ left_transposed))
    scaling = np.zeros((len(correspondences), 3, 3))
    scaling[:, 0, 0] = scaling[:, 1, 1] = 1.0
    scaling[:, 2, 2] = flip
    rotations = right_transposed @ scaling @ left_transposed
    residual = np.einsum("ki,pji->pkj", template, rotations) - targets
    deviations = np.sqrt(np.mean(np.sum(residual * residual, axis=-1), axis=-1))
    best = int(np.argmin(deviations))
    return rotations[best], correspondences[best], float(deviations[best])


def _radius_correspondence(template: np.ndarray, instance: np.ndarray,
                           groups: list[tuple[list[int], list[int]]]) -> np.ndarray:
    """
    Match template slots to instance vertices by their distance from the centre.

    Distance from the centre is invariant under rotation, so for a rigid pair whose vertex radii are distinct this is
    already the correct correspondence and the rotation follows in one Kabsch step. It is the initialisation that makes
    the alternating solver reliable; random restarts alone settle into local optima.

    :param template:
        Template vertices of shape (n, 3).
    :type template: numpy.ndarray
    :param instance:
        Instance vertices of shape (n, 3).
    :type instance: numpy.ndarray
    :param groups:
        Template slots and instance vertices that may be matched within, from _correspondence_groups.
    :type groups: list[tuple[list[int], list[int]]]

    :return:
        Correspondence of shape (n,), where entry k is the instance vertex matched to template slot k.
    :rtype: numpy.ndarray
    """
    template_radii = np.linalg.norm(template, axis=-1)
    instance_radii = np.linalg.norm(instance, axis=-1)
    correspondence = np.empty(len(template), dtype=np.int64)
    for slots, vertices in groups:
        ordered_slots = np.array(slots)[np.argsort(template_radii[slots])]
        ordered_vertices = np.array(vertices)[np.argsort(instance_radii[vertices])]
        correspondence[ordered_slots] = ordered_vertices
    return correspondence


def _rmsd(template: np.ndarray, instance: np.ndarray, rotation: np.ndarray, correspondence: np.ndarray) -> float:
    """
    Compute the root-mean-square deviation of a placed template from an instance.

    :param template:
        Template vertices of shape (n, 3).
    :type template: numpy.ndarray
    :param instance:
        Instance vertices of shape (n, 3).
    :type instance: numpy.ndarray
    :param rotation:
        Rotation applied to the template, of shape (3, 3).
    :type rotation: numpy.ndarray
    :param correspondence:
        Instance vertex matched to every template slot, of shape (n,).
    :type correspondence: numpy.ndarray

    :return:
        Root-mean-square deviation in Angstrom, or zero for an empty block.
    :rtype: float
    """
    if len(template) == 0:
        return 0.0
    residual = _rotate(template, rotation) - instance[correspondence]
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1))))


def _frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Build an orthonormal frame from two vectors that are not parallel.

    :param first:
        The vector the first axis is taken along, of shape (3,).
    :type first: numpy.ndarray
    :param second:
        The vector the second axis is taken in the plane of, of shape (3,).
    :type second: numpy.ndarray

    :return:
        The frame, of shape (3, 3), with the axes as columns.
    :rtype: numpy.ndarray
    """
    axis = first / np.linalg.norm(first)
    planar = second - np.dot(second, axis) * axis
    planar = planar / np.linalg.norm(planar)
    return np.stack([axis, planar, np.cross(axis, planar)], axis=-1)


def _frame_starts(template: np.ndarray, instance: np.ndarray,
                  groups: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """
    Enumerate the rotations that could carry a template onto an instance, by where they could send two of its vertices.

    A rotation is fixed by the images of two vertices that are not collinear with the centre, so every rotation worth
    trying is generated by choosing those two images. There are fewer than the square of the vertex count of them, and
    for a rigid polyhedron the true rotation is one of them up to the distortion of the instance. That is what random
    initial rotations cannot promise: on MPTS-52 blocks where exact enumeration is also affordable, alternating from
    ten random starts missed the exact optimum on 23 per cent of blocks and by as much as 2.1 Angstrom, which is far
    more than the tolerance any of this is scored at.

    :param template:
        Template vertices of shape (n, 3), relative to the centre.
    :type template: numpy.ndarray
    :param instance:
        Instance vertices of shape (n, 3), relative to the centre.
    :type instance: numpy.ndarray
    :param groups:
        Template slots and the instance vertices they may map to, grouped by species.
    :type groups: list[tuple[numpy.ndarray, numpy.ndarray]]

    :return:
        Candidate rotations, each of shape (3, 3).
    :rtype: list[numpy.ndarray]
    """
    radii = np.linalg.norm(template, axis=-1)
    primary = int(np.argmax(radii))
    secondary = next((index for index in range(len(template))
                      if index != primary
                      and np.linalg.norm(np.cross(template[primary], template[index])) > 1e-6 * radii[primary]), None)
    if secondary is None:
        return []

    allowed = {int(slot): vertices for slots, vertices in groups for slot in slots}
    source = _frame(template[primary], template[secondary])
    starts = []
    for first in allowed.get(primary, ()):
        for second in allowed.get(secondary, ()):
            if second == first or np.linalg.norm(np.cross(instance[first], instance[second])) < 1e-9:
                continue
            starts.append(_frame(instance[first], instance[second]) @ source.T)
    return starts


def align(template: np.ndarray, instance: np.ndarray, template_species: tuple[str, ...],
          instance_species: tuple[str, ...],
          permutation_limit: int = PERMUTATION_LIMIT,
          species_aware: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find the rotation and the vertex correspondence that best place a template onto one instance.

    Both the rotation and the correspondence are unknown, and neither can be solved without the other. Small vertex
    counts are handled by enumerating every species-consistent correspondence exactly, which is what most coordination
    numbers allow. Larger ones alternate optimal assignment and Kabsch, which only reaches a local optimum, so the
    starting rotations are the ones that could carry two template vertices onto two instance vertices rather than
    random ones; see ``_frame_starts``.

    :param template:
        Template vertices of shape (n, 3), relative to the centre.
    :type template: numpy.ndarray
    :param instance:
        Instance vertices of shape (n, 3), relative to the centre.
    :type instance: numpy.ndarray
    :param template_species:
        Element symbols of the template vertices.
    :type template_species: tuple[str, ...]
    :param instance_species:
        Element symbols of the instance vertices.
    :type instance_species: tuple[str, ...]
    :param permutation_limit:
        Largest number of correspondences to enumerate exactly.
        Defaults to PERMUTATION_LIMIT.
    :type permutation_limit: int
    :param species_aware:
        Whether correspondences are restricted to vertices of the same element. Coarse ``(centre, CN)`` templates are
        species-blind, so this is False for them.
        Defaults to True.
    :type species_aware: bool

    :return:
        Rotation of shape (3, 3), correspondence of shape (n,) giving the instance vertex of every template slot, and the
        root-mean-square deviation in Angstrom.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, float]

    :raises ValueError:
        If the template and the instance hold different numbers of vertices.
    """
    if len(template) != len(instance):
        raise ValueError(f"Cannot align {len(template)} template vertices to {len(instance)} instance vertices.")
    if len(template) == 0:
        return np.eye(3), np.empty(0, dtype=np.int64), 0.0

    if species_aware:
        groups = _correspondence_groups(template_species, instance_species)
    else:
        indices = list(range(len(template)))
        groups = [(indices, list(range(len(instance))))]
    exact_count = int(np.prod([factorial(len(slots)) for slots, _ in groups]))

    if exact_count <= permutation_limit:
        hashable = tuple((tuple(slots), tuple(vertices)) for slots, vertices in groups)
        return _best_enumerated(template, instance, _enumerated_correspondences(hashable, len(template)))

    starts = [kabsch(template, instance[_radius_correspondence(template, instance, groups)]), np.eye(3)]
    starts += _frame_starts(template, instance, groups)

    best: Optional[tuple[np.ndarray, np.ndarray, float]] = None
    for start in starts:
        rotation, correspondence = start, None
        for _ in range(HUNGARIAN_ITERATIONS):
            candidate = _assign(_rotate(template, rotation), instance, groups)
            if correspondence is not None and np.array_equal(candidate, correspondence):
                break
            correspondence = candidate
            rotation = kabsch(template, instance[correspondence])
        assert correspondence is not None
        deviation = _rmsd(template, instance, rotation, correspondence)
        if best is None or deviation < best[2]:
            best = (rotation, correspondence, deviation)

    assert best is not None
    return best


def _neighbour_info(structure: PymatgenStructure) -> Optional[list[list[dict]]]:
    """
    Find the CrystalNN neighbours of every site, keeping the periodic image of each neighbour.

    The image matters here and does not in shells.py: a polyhedron's geometry is only a polyhedron if its vertices are
    the image actually bonded to the centre rather than the representative inside the cell.

    :param structure:
        The crystal structure, decorated with oxidation states where possible.
    :type structure: pymatgen.core.Structure

    :return:
        CrystalNN neighbour records of every site, or None if the neighbour analysis failed.
    :rtype: Optional[list[list[dict]]]
    """
    finder = CrystalNN()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return [finder.get_nn_info(structure, index) for index in range(len(structure))]
    except (ValueError, RuntimeError, IndexError):
        # CrystalNN fails on pathological cells, for example ones whose Voronoi construction is degenerate.
        return None


def _charge_balanced_cations(symbols: tuple[str, ...], distinct: frozenset[str]) -> Optional[frozenset[str]]:
    """
    Find the cations of a composition from a charge-balanced assignment of oxidation states.

    :param symbols:
        Element symbol of every atom.
    :type symbols: tuple[str, ...]
    :param distinct:
        The distinct elements present.
    :type distinct: frozenset[str]

    :return:
        The cation elements, or None if no assignment exists or it makes every element a cation.
    :rtype: Optional[frozenset[str]]
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            guesses = Composition(dict(Counter(symbols))).oxi_state_guesses()
    except (ValueError, KeyError, IndexError):
        return None
    if not guesses:
        return None
    cations = frozenset(element for element, state in guesses[0].items() if state > 0.0)
    return cations if 0 < len(cations) < len(distinct) else None


def _electronegative_split(distinct: frozenset[str]) -> Optional[frozenset[str]]:
    """
    Find the cations of a composition by treating its most electronegative element as the anion.

    This is the fallback for compositions that do not charge-balance. It refuses to answer when the gap to the next
    element is too small to be meaningful, because calling the marginally less electronegative element of an alloy its
    cation would put the centres somewhere arbitrary and, worse, make the block count depend on a coin flip.

    :param distinct:
        The distinct elements present.
    :type distinct: frozenset[str]

    :return:
        The cation elements, or None if any element has no Pauling electronegativity, all of them share the highest, or
        the gap is below MINIMUM_ELECTRONEGATIVITY_GAP.
    :rtype: Optional[frozenset[str]]
    """
    values = {}
    for symbol in distinct:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            electronegativity = Element(symbol).X
        if electronegativity is None or isnan(electronegativity):
            return None
        values[symbol] = electronegativity

    ordered = sorted(values.values())
    highest = ordered[-1]
    cations = frozenset(symbol for symbol, value in values.items() if value < highest)
    if not cations or highest - max(value for value in ordered if value < highest) < MINIMUM_ELECTRONEGATIVITY_GAP:
        return None
    return cations


@lru_cache(maxsize=8192)
def centre_elements(symbols: tuple[str, ...]) -> tuple[frozenset[str], str]:
    """
    Decide which elements are polyhedron centres, using nothing but the composition.

    This is the constraint that keeps a rigid-block model buildable: the number of blocks is the number of centres, so it
    has to follow from what is known at inference. Composition-conditioned crystal structure prediction gives the exact
    multiset of atoms, so any rule that reads only element symbols is admissible and any rule that reads positions is not.

    A structure with no cation, either because it holds one element or because no element is decisively more
    electronegative than the rest, makes every atom a centre. That gives up the compression but keeps the decomposition
    defined, and it is the honest answer: there is no coordination polyhedron in elemental carbon or in a symmetric alloy
    that the composition could point to.

    :param symbols:
        Element symbol of every atom.
    :type symbols: tuple[str, ...]

    :return:
        The centre elements, and the name of the rule that decided them.
    :rtype: tuple[frozenset[str], str]
    """
    distinct = frozenset(symbols)
    if len(distinct) == 1:
        return distinct, "elemental"
    cations = _charge_balanced_cations(symbols, distinct)
    if cations is not None:
        return cations, "charge balance"
    cations = _electronegative_split(distinct)
    if cations is not None:
        return cations, "electronegativity"
    return distinct, "no polar split"


def decompose(structure: PymatgenStructure, identifier: str = "", type_key_mode: str = "centre-cn-ligands",
              exclude_centre_vertices: bool = True,
              coordination_cap: int = COORDINATION_CAP) -> Optional[Decomposition]:
    """
    Write one structure as overlapping coordination polyhedra.

    :param structure:
        The crystal structure.
    :type structure: pymatgen.core.Structure
    :param identifier:
        Material identifier of the structure.
        Defaults to the empty string.
    :type identifier: str
    :param type_key_mode:
        How block types are defined. "centre-cn" shares a template across every polyhedron with the same central element
        and coordination number, and "centre-cn-ligands" additionally requires the same vertex composition.
        Defaults to "centre-cn-ligands".
    :type type_key_mode: str
    :param exclude_centre_vertices:
        Whether to drop neighbours that are themselves block centres, so that the two roles an atom can play are
        disjoint. See EXCLUDE_CENTRE_VERTICES for why this is on by default.
        Defaults to True.
    :type exclude_centre_vertices: bool
    :param coordination_cap:
        Largest number of vertices a block may keep; the nearest are kept and the rest are left to be picked up as
        orphans. Defaults to COORDINATION_CAP.
    :type coordination_cap: int

    :return:
        The decomposition, or None if the neighbour analysis failed.
    :rtype: Optional[Decomposition]

    :raises ValueError:
        If the block type mode is unknown.
    """
    if type_key_mode not in ("centre-cn", "centre-cn-ligands"):
        raise ValueError(f"Unknown block type mode {type_key_mode!r}.")

    decorated = decorate_with_oxidation_states(structure)
    info = _neighbour_info(decorated)
    if info is None:
        return None

    symbols = tuple(site.specie.symbol for site in structure)
    centre_species, centre_rule = centre_elements(symbols)
    centres = [index for index, symbol in enumerate(symbols) if symbol in centre_species]
    coords = np.array([site.coords for site in structure], dtype=np.float64)
    numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)

    blocks, covered = [], set()
    for centre in centres:
        records = info[centre]
        if exclude_centre_vertices:
            records = [record for record in records if symbols[record["site_index"]] not in centre_species]
        if not records:
            # An atom with no vertices is not a polyhedron; the singleton pass below gives it a translation of its own.
            continue
        ligands = np.array([record["site_index"] for record in records], dtype=np.int64)
        offsets = np.array([record["site"].coords for record in records], dtype=np.float64) - coords[centre]
        if len(records) > coordination_cap:
            # Truncation keeps the nearest vertices, since those are the ones a coordination shell is defined by and the
            # ones a template can place accurately. Whatever is dropped loses its cover here and is repaired downstream
            # by cgfm.readout.orphan_free, which is why the cap and the repair have to agree on the same cap.
            nearest = np.argsort(np.linalg.norm(offsets, axis=-1))[:coordination_cap]
            ligands, offsets = ligands[nearest], offsets[nearest]
        species = tuple(symbols[index] for index in ligands)
        key = ((symbols[centre], len(ligands)) if type_key_mode == "centre-cn"
               else (symbols[centre], len(ligands), tuple(sorted(species))))
        blocks.append(Block(centre=centre, ligands=ligands, offsets=offsets, species=species, type_key=key))
        covered.add(centre)
        covered.update(int(index) for index in ligands)

    # Atoms no polyhedron reaches carry their own position, which is an optimistic but unavoidable choice: a rigid-body
    # model would have to spend a translation on each of them, and the reported count says how many that is.
    singletons = [index for index in range(len(structure)) if index not in covered]
    for index in singletons:
        blocks.append(Block(centre=index, ligands=np.empty(0, dtype=np.int64), offsets=np.empty((0, 3)),
                            species=(), type_key=(structure[index].specie.symbol, 0)))

    return Decomposition(identifier=identifier, lattice=np.array(structure.lattice.matrix, dtype=np.float64),
                         coords=coords, numbers=numbers, blocks=tuple(blocks), num_singletons=len(singletons),
                         centre_rule=centre_rule)


def fit_templates(instances: dict[tuple, list[tuple[np.ndarray, tuple[str, ...]]]], iterations: int = 3,
                  max_instances: int = 200, species_aware: bool = True) -> dict[tuple, Template]:
    """
    Fit one canonical rigid geometry per block type by generalised Procrustes averaging.

    :param instances:
        Vertex offsets and vertex species of every observed block, grouped by block type.
    :type instances: dict[tuple, list[tuple[numpy.ndarray, tuple[str, ...]]]]
    :param iterations:
        Number of times the template is refitted to the mean of the aligned instances.
        Defaults to 3.
    :type iterations: int
    :param max_instances:
        Largest number of instances of a type used for the fit. Templates converge long before this, and the alignment
        of one instance is the expensive part.
        Defaults to 200.
    :type max_instances: int
    :param species_aware:
        Whether alignment is restricted to vertices of the same element. Coarse ``(centre, CN)`` templates pass False
        so that the fit is species-blind and the per-slot element distribution is learned from the aligned instances.
        Defaults to True.
    :type species_aware: bool

    :return:
        Canonical template of every block type.
    :rtype: dict[tuple, Template]

    :raises ValueError:
        If fewer than one iteration is requested, which would leave the reported spread unmeasured.
    """
    if iterations < 1:
        raise ValueError("Fitting a template takes at least one iteration.")
    templates = {}
    for key, observed in instances.items():
        sample = observed[:max_instances]
        offsets, species = sample[0]
        template = offsets.copy()
        deviations: list[float] = []
        probabilities = np.zeros((len(template), MAX_ATOM_NUM + 1), dtype=np.float64)
        last_correspondences: list[np.ndarray] = []
        for _ in range(iterations):
            accumulated = np.zeros_like(template)
            deviations = []
            last_correspondences = []
            for instance_offsets, instance_species in sample:
                rotation, correspondence, deviation = align(
                    template, instance_offsets, species, instance_species, species_aware=species_aware)
                accumulated += _rotate(instance_offsets[correspondence], rotation.T)
                deviations.append(deviation)
                last_correspondences.append(correspondence)
            template = accumulated / len(sample)
        for (_, instance_species), correspondence in zip(sample, last_correspondences):
            for slot, vertex in enumerate(correspondence):
                probabilities[slot, Element(instance_species[vertex]).Z] += 1.0
        if len(sample) > 0 and len(template) > 0:
            probabilities /= len(sample)
            species = tuple(Element.from_Z(int(np.argmax(slot[1:]) + 1)).symbol for slot in probabilities)
        templates[key] = Template(offsets=template, species=species, count=len(observed),
                                  spread=float(np.mean(deviations)) if deviations else 0.0,
                                  species_probs=probabilities, species_aware=species_aware)
    return templates


def reconstruct(decomposition: Decomposition, templates: dict[tuple, Template], translation_sigma: float = 0.0,
                rotation_sigma: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Rebuild a structure from type-shared rigid templates placed at oracle poses.

    Every block contributes a predicted position for its centre and for each of its vertices. Predictions are combined as
    displacements from the true site rather than as absolute positions, which resolves the periodic-image ambiguity of a
    shared atom automatically: a vertex that is a distant image of a site still yields a small displacement once the
    displacement is reduced to its minimum image, so predictions from different blocks average correctly.

    :param decomposition:
        The block decomposition of the structure.
    :type decomposition: Decomposition
    :param templates:
        Canonical template of every block type. A block whose type is absent is resolved against the training vocabulary
        by ``lookup_template`` rather than by keeping the block's own geometry.
    :type templates: dict[tuple, Template]
    :param translation_sigma:
        Standard deviation of isotropic Gaussian noise added to every block translation, in Angstrom. Zero gives the
        oracle pose.
        Defaults to 0.0.
    :type translation_sigma: float
    :param rotation_sigma:
        Standard deviation of the rotation angle of the noise added to every block orientation, in degrees. Zero gives
        the oracle pose.
        Defaults to 0.0.
    :type rotation_sigma: float
    :param rng:
        Random generator used for the pose noise. Required when either noise level is positive.
        Defaults to None.
    :type rng: Optional[numpy.random.Generator]

    :return:
        Reconstructed Cartesian coordinates of shape (N, 3).
    :rtype: numpy.ndarray

    :raises ValueError:
        If pose noise is requested without a random generator.
    """
    noisy = translation_sigma > 0.0 or rotation_sigma > 0.0
    if noisy and rng is None:
        raise ValueError("Pose noise needs a random generator.")

    cell = Cell.of(decomposition.lattice)
    num_atoms = len(decomposition.coords)
    displacement = np.zeros((num_atoms, 3))
    counts = np.zeros(num_atoms)

    for block in decomposition.blocks:
        template, _ = lookup_template(block.type_key, templates)
        if len(block.offsets) == 0:
            offsets, correspondence = block.offsets, np.arange(len(block.offsets))
            rotation = np.eye(3)
        elif len(template.offsets) != len(block.offsets):
            offsets, correspondence = block.offsets, np.arange(len(block.offsets))
            rotation = np.eye(3)
        else:
            rotation, correspondence, _ = align(
                template.offsets, block.offsets, template.species, block.species,
                species_aware=template.species_aware)
            offsets = template.offsets

        centre_shift = np.zeros(3)
        if noisy:
            assert rng is not None
            if translation_sigma > 0.0:
                centre_shift = rng.normal(scale=translation_sigma, size=3)
            if rotation_sigma > 0.0:
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = np.deg2rad(rng.normal(scale=rotation_sigma))
                rotation = Rotation.from_rotvec(angle * axis).as_matrix() @ rotation

        centre = block.centre
        displacement[centre] += cell.minimum_image(centre_shift)
        counts[centre] += 1.0

        if len(offsets) > 0:
            predicted = decomposition.coords[centre] + centre_shift + _rotate(offsets, rotation)
            sites = block.ligands[correspondence]
            shifts = cell.minimum_image(predicted - decomposition.coords[sites])
            # np.add.at rather than fancy-index assignment because one site may be several vertices of one block.
            np.add.at(displacement, sites, shifts)
            np.add.at(counts, sites, 1.0)

    uncovered = counts == 0.0
    coords = decomposition.coords + displacement / np.maximum(counts, 1.0)[:, None]
    if np.any(uncovered):
        centres = decomposition.coords[[block.centre for block in decomposition.blocks]]
        for atom in np.flatnonzero(uncovered):
            nearest = int(np.argmin(np.linalg.norm(
                cell.minimum_image(centres - decomposition.coords[atom]), axis=-1)))
            coords[atom] = centres[nearest]
    return coords
