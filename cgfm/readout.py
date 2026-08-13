"""
Recovery of atom positions from overlapping rigid blocks without being told which blocks share which atoms.

``cgfm.blocks.reconstruct`` rebuilds a structure from block poses by indexing ``block.ligands[correspondence]``, which is
the sharing graph of the true structure. That is legitimate for measuring a ceiling and useless for generating one: at
sampling time there is no true structure to read the graph off, and the graph is the one part of the parameterization
with no published precedent for inorganic crystals. Generating it is a hard combinatorial problem over a bipartite
incidence with periodic image labels.

This module exists to avoid having to. A block that has been placed does not merely assert that some atom is one of its
vertices; it asserts *where* that atom is. Two blocks that share an atom therefore place a vertex at nearly the same
point, and the sharing graph is recoverable from the geometry instead of generated: cluster the vertex predictions and
each cluster is an atom. The composition fixes how many clusters there must be of each element, so the count is not
guessed either.

The construction rests on one property of the decomposition that has to be arranged rather than assumed. Every atom of a
centre element is a block centre, so its position is a block translation and is exact; only the non-centre atoms are ever
in question. ``cgfm.blocks.decompose`` leaves a small number of non-centre atoms in no polyhedron at all and gives each
its own singleton block, which breaks the property that makes the block count knowable at inference: the number of blocks
would then depend on how many atoms the neighbour finder happened to miss, which is not a function of the composition.
``orphan_free`` repairs that by attaching every such atom to its nearest centre, after which the block count is exactly
the number of centre atoms and follows from the composition alone.

What the clustering can lose is measured rather than argued, by ``cgfm.scripts.readout_ceiling``, which scores this
readout and the sharing-graph readout of ``cgfm.blocks.reconstruct`` on identical poses.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional
import numpy as np
from pymatgen.core import Element
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation
from omg.globals import MAX_ATOM_NUM
from .blocks import COORDINATION_CAP, Block, Decomposition, Template, align, centre_elements, lookup_template, template_species_probs
from .periodic import Cell


LINKAGE_METHOD = "average"
"""
Agglomerative linkage rule used to group vertex predictions into atoms.

Average linkage rather than single linkage because the failure that matters is chaining: two atoms two Angstrom apart,
each holding a spread of predictions, are joined by single linkage as soon as one prediction of one drifts towards the
other, and the merge then swallows both. Averaging over the cluster makes a merge require the whole cluster to be close.
"""

REFINEMENT_ITERATIONS = 8
"""
Largest number of rounds of constrained reassignment run after the initial clustering.

The rounds are cheap and converge in two or three, since each one only moves predictions already near a boundary. The
cap exists to stop a rare oscillation between two equal-cost assignments, not because further rounds would cost much.
"""

DUPLICATE_TOLERANCE = 0.1
"""
Separation in Angstrom below which two predictions from the same block are taken to be of the same atom.

Almost always a block's vertices are distinct atoms, which is the constraint the refinement rests on. The exception is a
cell small enough that two periodic images of one site both bond to the same centre, and that block then really does
hold two vertices of one atom. Both images wrap to the same place, so merging predictions this close restores the
constraint rather than approximating it.
"""


@dataclass(frozen=True)
class Placement:
    """
    Poses of every block of one structure, in the form a generative model would emit them.

    There is deliberately no atom in this object other than the block centres. Everything a sampler knows is here, and
    nothing that only a known structure could supply.

    :param lattice:
        Cell vectors of shape (3, 3), with row i the i-th lattice vector.
    :type lattice: numpy.ndarray
    :param centre_numbers:
        Atomic number of every block centre, of shape (M,).
    :type centre_numbers: numpy.ndarray
    :param centre_coords:
        Cartesian position of every block centre, of shape (M, 3), which is the block's translation.
    :type centre_coords: numpy.ndarray
    :param rotations:
        Orientation of every block, of shape (M, 3, 3), mapping the block's template into the cell frame.
    :type rotations: numpy.ndarray
    :param type_keys:
        Block type of every block, which selects its template.
    :type type_keys: tuple[tuple, ...]
    """

    lattice: np.ndarray
    centre_numbers: np.ndarray
    centre_coords: np.ndarray
    rotations: np.ndarray
    type_keys: tuple[tuple, ...]


@dataclass(frozen=True)
class Readout:
    """
    A structure recovered from block poses alone, with the diagnostics that say how well the recovery was determined.

    :param coords:
        Cartesian coordinates of shape (N, 3). Atoms are in no particular order, since the readout has no atom
        identities to preserve; the accompanying atomic numbers say what each one is.
    :type coords: numpy.ndarray
    :param numbers:
        Atomic numbers of shape (N,), whose multiset is the target composition by construction.
    :type numbers: numpy.ndarray
    :param votes_per_atom:
        Mean number of vertex predictions that were averaged into each non-centre atom. One means the readout had no
        consensus to draw on for that structure.
    :type votes_per_atom: float
    :param cluster_spread:
        Mean distance from a vertex prediction to the centroid of its cluster, in Angstrom. This is the disagreement
        between blocks that the consensus had to reconcile.
    :type cluster_spread: float
    :param shortfall:
        Number of atoms the composition demanded that could not be supplied with a vertex prediction. For ``read_out``
        this is a total: how far the number of votes fell short of the number of atoms, which says nothing about
        whether the votes that do exist are spread over distinct atoms, so a zero here is not evidence that every atom
        was reached. For ``read_out_with_oracle_clusters`` it is the stronger per-atom count, of atoms that no vote
        landed nearest to.
    :type shortfall: int
    """

    coords: np.ndarray
    numbers: np.ndarray
    votes_per_atom: float
    cluster_spread: float
    shortfall: int


def _cluster(points: np.ndarray, count: int, cell: Cell, method: str = LINKAGE_METHOD) -> np.ndarray:
    """
    Group points into exactly the requested number of clusters.

    The count is not inferred from the geometry because it does not have to be: the composition states it. Requiring it
    exactly is also what keeps the readout honest, since a readout free to choose its own atom count could always
    produce something and would never be wrong in a way that shows up in the composition.

    :param points:
        Cartesian points of shape (n, 3).
    :type points: numpy.ndarray
    :param count:
        Number of clusters, which must not exceed the number of points.
    :type count: int
    :param cell:
        The periodic cell the points live in.
    :type cell: cgfm.periodic.Cell
    :param method:
        Agglomerative linkage rule.
        Defaults to LINKAGE_METHOD.
    :type method: str

    :return:
        Cluster index of every point, of shape (n,), taking every value in [0, count).
    :rtype: numpy.ndarray

    :raises ValueError:
        If more clusters are requested than there are points.
    """
    if count > len(points):
        raise ValueError(f"Cannot split {len(points)} predictions into {count} atoms.")
    if count == len(points):
        return np.arange(count)

    distances = cell.distances(points)
    labels = fcluster(linkage(squareform(distances, checks=False), method=method), t=count,
                      criterion="maxclust") - 1

    # fcluster cuts the tree at the smallest threshold yielding at most count clusters, which is fewer than count when
    # several merges happen at the same height. The widest cluster is then split at its own widest merge until the count
    # is met, which cannot fail because a cluster of one point is never the widest of a set that is short of clusters.
    while len(np.unique(labels)) < count:
        members = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        widest = max(members, key=lambda group: distances[np.ix_(group, group)].max())
        inner = fcluster(linkage(squareform(distances[np.ix_(widest, widest)], checks=False), method=method), t=2,
                         criterion="maxclust")
        labels[widest[inner == 2]] = labels.max() + 1
        labels = np.unique(labels, return_inverse=True)[1]
    return labels


def orphan_free(decomposition: Decomposition, type_key_mode: str = "centre-cn-ligands",
                coordination_cap: int = COORDINATION_CAP) -> Decomposition:
    """
    Rewrite a decomposition so that its blocks stand in one-to-one correspondence with its centre atoms.

    Two things follow from that correspondence, and neither holds for the decomposition as ``cgfm.blocks.decompose``
    returns it. The number of blocks becomes a function of the composition, which is what a sampler has, rather than of
    how many atoms the neighbour finder happened to leave uncovered, which is a property of the answer. And every
    non-centre atom becomes a vertex of at least one block, so every atom that is not placed by a translation is placed
    by at least one vertex prediction and the readout is never asked to invent one from nothing. The second holds
    except where the coordination cap binds, which is discussed below.

    An atom the neighbour finder left uncovered joins the nearest centre. That lengthens one edge of that polyhedron and
    so blurs its type's template a little, which is the price of the guarantee and is measured with everything else.

    Repair and truncation are two halves of one rule and are written against the same cap. ``decompose`` drops a block's
    furthest vertices past the cap, which can leave the dropped atom uncovered, and this function would happily hand it
    back to the block that just shed it. So an orphan joins the nearest centre *with room left*, and orphans are placed
    nearest-first so that the least ambiguous attachments get their first choice.

    Room can run out, and then the guarantee fails rather than the cap. Vertices are shared, so the slots a structure
    needs are not bounded by its atom count and a cell with very few cations and many anions can exceed twelve times
    the centre count: four of 250 MPTS-52 validation structures do. Those atoms are left in no block at all. Exceeding
    the cap instead would produce a coordination number the discrete stage cannot emit, which is a representation that
    quietly does not exist; leaving the atom out is a representation that exists and loses it, and the loss is visible
    in the coverage reported by ``read_out_with_oracle_clusters``. Callers that need the guarantee unconditionally
    should raise the cap rather than relax the check.

    :param decomposition:
        The decomposition to repair.
    :type decomposition: Decomposition
    :param type_key_mode:
        How block types are defined, which must match the mode the decomposition was built with because attaching an
        atom changes the coordination number and so the type.
        Defaults to "centre-cn-ligands".
    :type type_key_mode: str
    :param coordination_cap:
        Largest number of vertices a block may hold, which must match the cap the decomposition was truncated at.
        Defaults to COORDINATION_CAP.
    :type coordination_cap: int

    :return:
        A decomposition with one block per centre atom.
    :rtype: Decomposition

    :raises ValueError:
        If the block type mode is unknown.
    """
    if type_key_mode not in ("centre-cn", "centre-cn-ligands"):
        raise ValueError(f"Unknown block type mode {type_key_mode!r}.")

    symbols = tuple(Element.from_Z(int(number)).symbol for number in decomposition.numbers)
    centre_species, _ = centre_elements(symbols)
    cell = Cell.of(decomposition.lattice)

    vertices: dict[int, list[tuple[int, np.ndarray]]] = {
        index: [] for index, symbol in enumerate(symbols) if symbol in centre_species}
    for block in decomposition.blocks:
        if len(block.ligands) > 0:
            vertices[block.centre] = [(int(index), offset) for index, offset in zip(block.ligands, block.offsets)]

    claimed = {index for members in vertices.values() for index, _ in members}
    orphans = [index for index, symbol in enumerate(symbols) if symbol not in centre_species and index not in claimed]
    if orphans:
        hosts = np.array(sorted(vertices))
        separations = cell.minimum_image(
            decomposition.coords[orphans][:, None, :] - decomposition.coords[hosts][None, :, :])
        distances = np.linalg.norm(separations, axis=-1)
        for orphan in np.argsort(np.min(distances, axis=-1)):
            for host in np.argsort(distances[orphan]):
                if len(vertices[int(hosts[host])]) < coordination_cap:
                    vertices[int(hosts[host])].append((orphans[orphan], separations[orphan, host]))
                    break

    blocks = []
    for centre in sorted(vertices):
        members = vertices[centre]
        ligands = np.array([index for index, _ in members], dtype=np.int64)
        offsets = np.array([offset for _, offset in members], dtype=np.float64).reshape(len(members), 3)
        species = tuple(symbols[index] for index in ligands)
        key = ((symbols[centre], len(members)) if type_key_mode == "centre-cn"
               else (symbols[centre], len(members), tuple(sorted(species))))
        blocks.append(Block(centre=centre, ligands=ligands, offsets=offsets, species=species, type_key=key))

    return Decomposition(identifier=decomposition.identifier, lattice=decomposition.lattice,
                         coords=decomposition.coords, numbers=decomposition.numbers, blocks=tuple(blocks),
                         num_singletons=sum(1 for block in blocks if len(block.ligands) == 0),
                         centre_rule=decomposition.centre_rule)


def oracle_placement(decomposition: Decomposition, templates: dict[tuple, Template],
                     translation_sigma: float = 0.0, rotation_sigma: float = 0.0,
                     rng: Optional[np.random.Generator] = None
                     ) -> tuple[Placement, tuple[np.ndarray, ...]]:
    """
    Compute the poses a perfect generative model would emit for a known structure.

    The vertex correspondences are returned alongside the poses and are *not* part of the placement, because they are
    the sharing graph and a sampler does not have them. They are here so that the graph-based readout and the
    assignment-free readout can be scored on exactly the same poses, which is the only way the difference between the
    two measures the readout rather than the noise draw.

    :param decomposition:
        The block decomposition, which must have one block per centre atom, as ``orphan_free`` produces.
    :type decomposition: Decomposition
    :param templates:
        Canonical template of every block type. A block whose type is absent is resolved by ``lookup_template`` against
        this table rather than by keeping the block's own geometry.
    :type templates: dict[tuple, Template]
    :param translation_sigma:
        Standard deviation of isotropic Gaussian noise added to every block translation, in Angstrom.
        Defaults to 0.0.
    :type translation_sigma: float
    :param rotation_sigma:
        Standard deviation of the rotation angle of the noise added to every block orientation, in degrees.
        Defaults to 0.0.
    :type rotation_sigma: float
    :param rng:
        Random generator used for the pose noise. Required when either noise level is positive.
        Defaults to None.
    :type rng: Optional[numpy.random.Generator]

    :return:
        The poses, and the instance vertex matched to every template slot of every block.
    :rtype: tuple[Placement, tuple[numpy.ndarray, ...]]

    :raises ValueError:
        If pose noise is requested without a random generator.
    """
    if (translation_sigma > 0.0 or rotation_sigma > 0.0) and rng is None:
        raise ValueError("Pose noise needs a random generator.")

    coords, rotations, correspondences = [], [], []
    for block in decomposition.blocks:
        template, _ = lookup_template(block.type_key, templates)
        if len(block.offsets) == 0 or len(template.offsets) != len(block.offsets):
            rotation = np.eye(3)
            correspondence = np.arange(len(block.offsets))
        else:
            rotation, correspondence, _ = align(
                template.offsets, block.offsets, template.species, block.species,
                species_aware=template.species_aware)

        centre = decomposition.coords[block.centre]
        if rng is not None:
            if translation_sigma > 0.0:
                centre = centre + rng.normal(scale=translation_sigma, size=3)
            if rotation_sigma > 0.0:
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                perturbation = Rotation.from_rotvec(np.deg2rad(rng.normal(scale=rotation_sigma)) * axis).as_matrix()
                rotation = perturbation @ rotation
        coords.append(centre)
        rotations.append(rotation)
        correspondences.append(correspondence)

    centres = np.array([block.centre for block in decomposition.blocks], dtype=np.int64)
    placement = Placement(lattice=decomposition.lattice, centre_numbers=decomposition.numbers[centres],
                          centre_coords=np.array(coords, dtype=np.float64),
                          rotations=np.array(rotations, dtype=np.float64),
                          type_keys=tuple(block.type_key for block in decomposition.blocks))
    return placement, tuple(correspondences)


def read_out_with_graph(placement: Placement, decomposition: Decomposition, templates: dict[tuple, Template],
                        correspondences: tuple[np.ndarray, ...]) -> Readout:
    """
    Rebuild a structure from block poses using the true sharing graph, for comparison against doing without it.

    This is ``cgfm.blocks.reconstruct`` expressed over an explicit placement rather than over a decomposition and a
    noise level, so that it can be given the same poses as the assignment-free readout. Predictions are combined as
    displacements from the true site, which resolves the periodic-image ambiguity of a shared atom automatically.

    :param placement:
        The block poses.
    :type placement: Placement
    :param decomposition:
        The true structure, which supplies the sharing graph and the sites the displacements are taken from.
    :type decomposition: Decomposition
    :param templates:
        Canonical template of every block type.
    :type templates: dict[tuple, Template]
    :param correspondences:
        Instance vertex matched to every template slot of every block, from ``oracle_placement``.
    :type correspondences: tuple[numpy.ndarray, ...]

    :return:
        The rebuilt structure, whose atoms are in the order of the true structure.
    :rtype: Readout
    """
    cell = Cell.of(decomposition.lattice)
    displacements = np.zeros_like(decomposition.coords)
    counts = np.zeros(len(decomposition.coords))

    for index, block in enumerate(decomposition.blocks):
        template, _ = lookup_template(block.type_key, templates)
        offsets = template.offsets if len(template.offsets) == len(block.offsets) else block.offsets
        centre_shift = placement.centre_coords[index] - decomposition.coords[block.centre]

        displacements[block.centre] += cell.minimum_image(centre_shift)
        counts[block.centre] += 1.0
        if len(offsets) == 0:
            continue

        predicted = placement.centre_coords[index] + offsets @ placement.rotations[index].T
        sites = block.ligands[correspondences[index]]
        np.add.at(displacements, sites, cell.minimum_image(predicted - decomposition.coords[sites]))
        np.add.at(counts, sites, 1.0)

    uncovered = counts == 0.0
    shortfall = int(uncovered.sum())
    coords = decomposition.coords + displacements / np.maximum(counts, 1.0)[:, None]
    if shortfall:
        centres = placement.centre_coords
        for atom in np.flatnonzero(uncovered):
            nearest = int(np.argmin(cell.distances(decomposition.coords[atom][None, :], centres)[0]))
            coords[atom] = centres[nearest]
    return Readout(coords=coords, numbers=decomposition.numbers,
                   votes_per_atom=float(counts.mean()), cluster_spread=0.0, shortfall=shortfall)


def derive_templates(numbers: np.ndarray, fine: dict[tuple, Template],
                     coarse: dict[tuple, Template]) -> dict[tuple, Template]:
    """
    Sharpen a structure's templates using the one thing about its vertex species the composition already fixes.

    A template keyed by the central element and the coordination number alone averages over every vertex composition
    that key was ever seen with, which blunts it, and it inherits the vertex element labels of whichever instance it was
    fitted from, which are then arbitrary. Keying on the vertex composition as well fixes both and cannot be used
    directly, because the vertex composition is a property of the structure, not of the composition a sampler starts
    from.

    Except when it is. A structure whose atoms are all either centres or one single other element has only one possible
    vertex composition per coordination number, so its fine key follows from its composition and nothing is being
    borrowed from the answer. That covers 81.1 per cent of MPTS-52, and those blocks get the sharper species-aware
    template. The rest keep the coarse template, whose per-slot element probabilities were learned from aligned training
    instances rather than from arbitrary hard vertex labels.

    :param numbers:
        Atomic numbers of the target composition, of shape (N,).
    :type numbers: numpy.ndarray
    :param fine:
        Templates keyed by central element, coordination number and sorted vertex composition.
    :type fine: dict[tuple, Template]
    :param coarse:
        Templates keyed by central element and coordination number only.
    :type coarse: dict[tuple, Template]

    :return:
        Templates keyed by central element and coordination number, which is the key a block carries, holding the fine
        template wherever the composition determined one.
    :rtype: dict[tuple, Template]
    """
    symbols = tuple(Element.from_Z(int(number)).symbol for number in numbers)
    centre_species, _ = centre_elements(symbols)
    others = sorted(set(symbols) - set(centre_species))
    if len(others) != 1:
        return dict(coarse)
    return {key: fine.get((key[0], key[1], (others[0],) * key[1]), template) for key, template in coarse.items()}


def read_out_with_oracle_clusters(placement: Placement, decomposition: Decomposition, templates: dict[tuple, Template],
                                  fallback: Optional[dict[tuple, Template]] = None) -> Readout:
    """
    Rebuild a structure by letting every vote join the true atom it is nearest to, which no sampler could do.

    This exists to answer one question that neither of the other readouts can. When the assignment-free readout loses a
    structure, either the votes were never good enough or the clustering grouped good votes wrongly, and the remedies
    are opposite: sharper templates in the first case, a better grouping in the second. Counting votes cannot tell them
    apart, because having as many votes as atoms says nothing about whether they are spread over the right atoms.

    Giving the grouping the answer separates them exactly. The votes here are the same votes, cast from the same
    placement through the same templates; only the grouping is oracular, so this is the best any clustering of these
    votes could do. The gap from the sharing-graph readout down to this is what the votes cost, and the gap from this
    down to the assignment-free readout is what the clustering costs.

    :param placement:
        The block poses.
    :type placement: Placement
    :param decomposition:
        The true structure, which supplies the atom each vote is nearest to.
    :type decomposition: Decomposition
    :param templates:
        Canonical template of every block type.
    :type templates: dict[tuple, Template]
    :param fallback:
        Templates to fall back on for a block type with no template of its own.
        Defaults to None.
    :type fallback: Optional[dict[tuple, Template]]

    :return:
        The rebuilt structure, whose shortfall is the number of non-centre atoms that no vote landed nearest to and so
        is a true per-atom coverage count rather than a total.
    :rtype: Readout
    """
    cell = Cell.of(placement.lattice)
    positions, _numbers, blocks, _probs = cast_votes(placement, templates, fallback=fallback)
    positions, _, blocks = _merge_duplicate_votes(positions, _numbers, blocks, cell)

    symbols = tuple(Element.from_Z(int(number)).symbol for number in decomposition.numbers)
    centre_species, _ = centre_elements(symbols)
    unknown = np.array([index for index, symbol in enumerate(symbols) if symbol not in centre_species], dtype=np.int64)
    if len(unknown) == 0:
        return Readout(coords=placement.centre_coords.copy(), numbers=placement.centre_numbers.copy(),
                       votes_per_atom=0.0, cluster_spread=0.0, shortfall=0)

    nearest = np.argmin(cell.distances(positions, decomposition.coords[unknown]), axis=-1)
    coords, numbers, spreads, sizes, missing = list(placement.centre_coords), list(placement.centre_numbers), [], [], 0
    for atom in range(len(unknown)):
        members = positions[nearest == atom]
        if len(members) == 0:
            # An atom no vote reached cannot be recovered by any grouping of the votes, and the composition still
            # demands it be placed somewhere. It goes on top of the nearest block centre, which is the worst a real
            # sampler would do and costs the structure its match. Putting it at its true position instead would be
            # borrowing the answer, and would let a readout that reached only half the atoms still report a ceiling
            # near a hundred per cent, which is exactly the reading this column exists to prevent.
            missing += 1
            coords.append(placement.centre_coords[np.argmin(cell.distances(
                decomposition.coords[unknown[atom]][None, :], placement.centre_coords)[0])])
        else:
            centroid = cell.mean(members)
            coords.append(centroid)
            spreads.append(float(np.linalg.norm(cell.minimum_image(members - centroid), axis=-1).mean()))
            sizes.append(len(members))
        numbers.append(decomposition.numbers[unknown[atom]])

    return Readout(coords=np.array(coords, dtype=np.float64), numbers=np.array(numbers, dtype=np.int64),
                   votes_per_atom=float(np.mean(sizes)) if sizes else 0.0,
                   cluster_spread=float(np.mean(spreads)) if spreads else 0.0, shortfall=missing)


def cast_votes(placement: Placement, templates: dict[tuple, Template],
               fallback: Optional[dict[tuple, Template]] = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Place every block's template and collect the position every block predicts for every vertex it holds.

    Which block cast each vote is returned as well, and it is not bookkeeping: a block's vertices are distinct atoms, so
    the provenance of a vote says which other votes it cannot be grouped with. That is the only constraint available to
    the readout beyond the positions themselves.

    :param placement:
        The block poses.
    :type placement: Placement
    :param templates:
        Canonical template of every block type.
    :type templates: dict[tuple, Template]
    :param fallback:
        Templates keyed by central element and coordination number only, consulted for a type the fine table does not
        hold. A block whose type is in neither table is resolved by ``lookup_template``, which never uses the
        evaluation block's own geometry.
        Defaults to None.
    :type fallback: Optional[dict[tuple, Template]]

    :return:
        Cartesian position of every vote, of shape (V, 3), the majority atomic number of the vertex it predicts, of
        shape (V,), the index of the block that cast it, of shape (V,), and the per-vote element distribution, of
        shape (V, MAX_ATOM_NUM + 1).
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    positions, numbers, blocks, probabilities = [], [], [], []
    for index, key in enumerate(placement.type_keys):
        template, _ = lookup_template(key, templates, fallback=fallback)
        if len(template.offsets) == 0:
            continue
        positions.append(placement.centre_coords[index] + template.offsets @ placement.rotations[index].T)
        slot_probs = template_species_probs(template)
        numbers.append(np.argmax(slot_probs[:, 1:], axis=-1).astype(np.int64) + 1)
        probabilities.append(slot_probs)
        blocks.append(np.full(len(template.offsets), index, dtype=np.int64))

    if not positions:
        return (np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty((0, MAX_ATOM_NUM + 1)))
    return (np.concatenate(positions, axis=0), np.concatenate(numbers, axis=0), np.concatenate(blocks),
            np.concatenate(probabilities, axis=0))


def _merge_duplicate_votes(positions: np.ndarray, numbers: np.ndarray, blocks: np.ndarray,
                           cell: Cell, probabilities: Optional[np.ndarray] = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
                               np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse predictions a single block makes twice for the same atom, which happens when the cell is small.

    :param positions:
        Cartesian position of every vote, of shape (V, 3).
    :type positions: numpy.ndarray
    :param numbers:
        Atomic number every vote predicts, of shape (V,).
    :type numbers: numpy.ndarray
    :param blocks:
        Index of the block that cast every vote, of shape (V,).
    :type blocks: numpy.ndarray
    :param cell:
        The periodic cell the votes live in.
    :type cell: cgfm.periodic.Cell
    :param probabilities:
        Optional per-vote element distribution, of shape (V, MAX_ATOM_NUM + 1).
        Defaults to None.
    :type probabilities: Optional[numpy.ndarray]

    :return:
        The votes, with same-block duplicates removed. The distribution is returned when it was supplied.
    """
    keep = np.ones(len(positions), dtype=bool)
    for block in np.unique(blocks):
        members = np.flatnonzero(blocks == block)
        if len(members) < 2:
            continue
        close = cell.distances(positions[members]) < DUPLICATE_TOLERANCE
        for first in range(len(members)):
            if keep[members[first]]:
                keep[members[first + 1:][close[first, first + 1:]]] = False
    if probabilities is None:
        return positions[keep], numbers[keep], blocks[keep]
    return positions[keep], numbers[keep], blocks[keep], probabilities[keep]


def _refine(positions: np.ndarray, blocks: np.ndarray, labels: np.ndarray, count: int,
            cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    """
    Reassign votes to atoms under the constraint that one block cannot predict the same atom twice.

    Agglomerative clustering treats the votes as an unstructured cloud, and its characteristic error is to split an atom
    that many blocks predict while merging two that few do. The constraint rules that error out directly: if a cluster
    already holds a vote from block m, no other vote from block m belongs in it. Imposing it turns the reassignment of
    one block's votes into a small rectangular assignment problem, solved exactly, and alternating that with recomputing
    the atom positions is Lloyd's algorithm with the constraint enforced in the assignment step.

    **This is measured to be worse than leaving it out, and is off by default.** On MPTS-52 it costs between one and
    four points of match rate at every pose noise level under both block type keys. The premise is not the problem: a
    quarter of blocks do hold two periodic images of one site, but those predictions coincide exactly and
    ``_merge_duplicate_votes`` removes all of them, so the vertices really are distinct atoms by the time this runs. The
    problem is that the assignment is total. Every vote of a block is placed somewhere, so a cluster that the initial
    clustering put in the wrong place gets filled rather than starved, and one bad cluster pulls good votes out of their
    atoms instead of quietly emptying. It is kept, and kept measurable, so that the finding is reproducible and nobody
    reaches for the constraint again without seeing what it does.

    :param positions:
        Cartesian position of every vote, of shape (V, 3).
    :type positions: numpy.ndarray
    :param blocks:
        Index of the block that cast every vote, of shape (V,). A vote with a negative index is unconstrained.
    :type blocks: numpy.ndarray
    :param labels:
        Initial cluster index of every vote, of shape (V,).
    :type labels: numpy.ndarray
    :param count:
        Number of atoms, which is the number of clusters.
    :type count: int
    :param cell:
        The periodic cell the votes live in.
    :type cell: cgfm.periodic.Cell

    :return:
        The refined cluster index of every vote, of shape (V,), and the atom positions, of shape (count, 3).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """

    def centroids_of(assignment: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
        """Recompute the atom positions, leaving a cluster that lost all of its votes where it was."""
        centres = np.empty((count, 3)) if previous is None else previous.copy()
        for cluster in range(count):
            members = positions[assignment == cluster]
            if len(members) > 0:
                centres[cluster] = cell.mean(members)
        return centres

    centroids = centroids_of(labels, None)
    for _ in range(REFINEMENT_ITERATIONS):
        proposal = np.empty_like(labels)
        for block in np.unique(blocks):
            members = np.flatnonzero(blocks == block)
            distances = cell.distances(positions[members], centroids)
            if block < 0 or len(members) > count:
                # A block holding more votes than there are atoms cannot place them all distinctly, and the filler votes
                # of a shortfall belong to no block at all. Both fall back to the nearest atom.
                proposal[members] = np.argmin(distances, axis=-1)
            else:
                rows, columns = linear_sum_assignment(distances)
                proposal[members[rows]] = columns

        if np.array_equal(proposal, labels):
            break
        labels = proposal
        centroids = centroids_of(labels, centroids)

    return labels, centroids_of(labels, centroids)


def _assign_elements(evidence: np.ndarray, remainder: dict[int, int]) -> np.ndarray:
    """
    Give every cluster an element so that the recovered composition is exactly the target one.

    :param evidence:
        Number of votes of every element in every cluster, of shape (K, E), where the columns follow the sorted order
        of the remainder.
    :type evidence: numpy.ndarray
    :param remainder:
        Number of atoms of every element that the clusters have to account for.
    :type remainder: dict[int, int]

    :return:
        Atomic number of every cluster, of shape (K,).
    :rtype: numpy.ndarray
    """
    elements = sorted(remainder)
    if len(elements) == 1:
        return np.full(len(evidence), elements[0], dtype=np.int64)

    slots = np.concatenate([np.full(remainder[element], index) for index, element in enumerate(elements)])
    rows, columns = linear_sum_assignment(-evidence[:, slots])
    return np.array([elements[slots[column]] for _, column in sorted(zip(rows, columns))], dtype=np.int64)


def read_out(placement: Placement, templates: dict[tuple, Template], target_numbers: np.ndarray,
             fallback: Optional[dict[tuple, Template]] = None, method: str = LINKAGE_METHOD,
             refine: bool = False) -> Readout:
    """
    Recover a whole structure from block poses, without being told which blocks share which atoms.

    Centres are placed by their translations and every other atom by the consensus of the blocks that predict it, where
    which predictions belong to the same atom is decided by clustering rather than supplied. The composition fixes how
    many atoms of each element the clustering must produce.

    Clustering ignores which element each prediction claims to be, and the elements are assigned to the clusters
    afterwards. That is not a refinement, it is the only correct order. A coarse template's hard vertex labels are
    arbitrary, so filtering predictions by those labels would discard most of them. Clustering on position alone uses
    only what a placed block actually asserts, and the assignment afterwards uses the per-slot element probabilities
    learned from aligned training instances rather than those hard labels. For the large majority of structures, which
    hold a single non-centre element, the assignment afterwards is forced and the question does not arise at all.

    :param placement:
        The block poses.
    :type placement: Placement
    :param templates:
        Canonical template of every block type.
    :type templates: dict[tuple, Template]
    :param target_numbers:
        Atomic number of every atom of the structure, of shape (N,), which is the composition and is known at sampling
        time. Its order carries no information and is not preserved.
    :type target_numbers: numpy.ndarray
    :param fallback:
        Templates keyed by central element and coordination number only, for a type the fine table does not hold.
        Defaults to None.
    :type fallback: Optional[dict[tuple, Template]]
    :param method:
        Agglomerative linkage rule used to group vertex predictions into atoms.
        Defaults to LINKAGE_METHOD.
    :type method: str
    :param refine:
        Whether to follow the clustering with the constrained reassignment of ``_refine``, which is measured to lose
        between one and four points of match rate and is off for that reason. Exposed so that the measurement can keep
        reporting it rather than the finding resting on a comment.
        Defaults to False.
    :type refine: bool

    :return:
        The recovered structure and the diagnostics of how well determined it was.
    :rtype: Readout

    :raises ValueError:
        If the block centres are not a sub-multiset of the target composition, which means the placement and the
        composition describe different structures.
    """
    counted = Counter(int(number) for number in target_numbers)
    counted.subtract(Counter(int(number) for number in placement.centre_numbers))
    if any(count < 0 for count in counted.values()):
        raise ValueError("The block centres hold atoms the target composition does not.")
    remainder = {number: count for number, count in counted.items() if count > 0}

    total = sum(remainder.values())
    if total == 0:
        return Readout(coords=placement.centre_coords.copy(), numbers=placement.centre_numbers.copy(),
                       votes_per_atom=0.0, cluster_spread=0.0, shortfall=0)

    cell = Cell.of(placement.lattice)
    positions, numbers, blocks, probabilities = cast_votes(placement, templates, fallback=fallback)
    positions, numbers, blocks, probabilities = _merge_duplicate_votes(
        positions, numbers, blocks, cell, probabilities=probabilities)

    shortfall = max(0, total - len(positions))
    if shortfall > 0:
        # Nothing in the poses says where these atoms are, so the structure is already wrong. They are placed on the
        # centres so that the composition still comes out right and the failure is scored rather than crashed on.
        filler = placement.centre_coords[np.arange(shortfall) % len(placement.centre_coords)]
        positions = np.concatenate([positions, filler], axis=0)
        numbers = np.concatenate([numbers, np.zeros(shortfall, dtype=np.int64)])
        blocks = np.concatenate([blocks, np.full(shortfall, -1, dtype=np.int64)])
        probabilities = np.concatenate(
            [probabilities, np.zeros((shortfall, probabilities.shape[1] if len(probabilities) else MAX_ATOM_NUM + 1))],
            axis=0)

    labels = _cluster(positions, total, cell, method=method)
    if refine:
        labels, centroids = _refine(positions, blocks, labels, total, cell)
    else:
        centroids = np.array([cell.mean(positions[labels == label]) for label in range(total)])
    elements = sorted(remainder)
    evidence = np.zeros((total, len(elements)))
    spreads, sizes = [], []
    for label in range(total):
        members = positions[labels == label]
        sizes.append(len(members))
        if len(members) > 0:
            offsets = cell.minimum_image(members - centroids[label])
            spreads.append(float(np.linalg.norm(offsets, axis=-1).mean()))
        for index, element in enumerate(elements):
            evidence[label, index] = float(probabilities[labels == label, element].sum())

    return Readout(coords=np.concatenate([placement.centre_coords, centroids], axis=0),
                   numbers=np.concatenate([placement.centre_numbers, _assign_elements(evidence, remainder)]),
                   votes_per_atom=float(np.mean(sizes)),
                   cluster_spread=float(np.mean(spreads)) if spreads else 0.0, shortfall=shortfall)


def displacement(readout: Readout, decomposition: Decomposition) -> float:
    """
    Measure how far the recovered atoms sit from the true ones, matching them by element at least total cost.

    The readout does not preserve atom identity, so the comparison has to solve for the correspondence. Matching at
    least cost is generous to the readout, which is the right direction for a ceiling: it credits a recovery that placed
    every atom well even if it could not say which was which.

    :param readout:
        The recovered structure.
    :type readout: Readout
    :param decomposition:
        The true structure.
    :type decomposition: Decomposition

    :return:
        Mean per-atom Cartesian displacement in Angstrom.
    :rtype: float

    :raises ValueError:
        If the recovered composition differs from the true one.
    """
    if Counter(int(n) for n in readout.numbers) != Counter(int(n) for n in decomposition.numbers):
        raise ValueError("The recovered structure and the true structure have different compositions.")

    cell = Cell.of(decomposition.lattice)
    total = 0.0
    for number in np.unique(readout.numbers):
        recovered = readout.coords[readout.numbers == number]
        true = decomposition.coords[decomposition.numbers == number]
        distances = cell.distances(recovered, true)
        rows, columns = linear_sum_assignment(distances)
        total += distances[rows, columns].sum()
    return total / len(readout.numbers)
