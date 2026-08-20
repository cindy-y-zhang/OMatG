"""
Coordination polyhedra of a crystal, counted by type and by how they connect.

The lithium thiophosphate electrolytes this package targets are built from anion
tetrahedra -- GeS4, PS4, SiS4 -- threaded through a lithium sublattice, and a family is
named by how many of each it contains and by whether they stand apart, meet at a corner
or share an edge. Li21Ge8P3S34 is interesting because eight-to-three is a ratio nobody
had made, and because eight GeS4 plus three PS4 would need forty-four sulfurs while the
formula supplies thirty-four, so ten of them are shared and the sharing is forced.

Counting a ratio is therefore not enough: the same ratio can be realised as isolated
tetrahedra or as a condensed network, and those are different materials with different
lithium conduction. Everything here reports both.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import acos, degrees, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Structure

IDEAL_TETRAHEDRAL_ANGLE = degrees(acos(-1.0 / 3.0))
"""The 109.47 degrees between two vertices of a regular tetrahedron seen from its centre."""

SHARING_NAMES = {0: "isolated", 1: "corner", 2: "edge", 3: "face"}
"""How many ligands two polyhedra hold in common, named the way crystallographers name it."""


@dataclass(frozen=True)
class PolyhedronSettings:
    """
    Which polyhedra to look for and how generous to be about what counts as a bond.

    :param centres:
        Element symbols allowed at the centre of a polyhedron.
        Defaults to the tetrahedron formers of the thiophosphate electrolytes.
    :type centres: tuple[str, ...]
    :param ligands:
        Element symbols allowed at the vertices.
        Defaults to sulfur alone.
    :type ligands: tuple[str, ...]
    :param bond_scale:
        A centre and a ligand are bonded when they are closer than this multiple of the
        sum of their covalent radii. The default of 1.3 is the usual slack for ionic
        solids; :func:`bond_cutoff` reports the resulting distance for any pair.
    :type bond_scale: float
    :param tetrahedral_tolerance:
        Largest root-mean-square deviation, in degrees, of the ligand-centre-ligand
        angles from 109.47 that still counts as a tetrahedron. Generated structures
        before relaxation are distorted, so this is deliberately loose.
        Defaults to 15.0.
    :type tetrahedral_tolerance: float
    """

    centres: tuple[str, ...] = ("Ge", "P", "Si", "Sn")
    ligands: tuple[str, ...] = ("S",)
    bond_scale: float = 1.3
    tetrahedral_tolerance: float = 15.0

    def __post_init__(self) -> None:
        if self.bond_scale <= 0.0:
            raise ValueError("bond_scale must be positive.")
        if self.tetrahedral_tolerance < 0.0:
            raise ValueError("tetrahedral_tolerance cannot be negative.")
        unknown = [s for s in (*self.centres, *self.ligands) if s not in CovalentRadius.radius]
        if unknown:
            raise ValueError(f"No covalent radius is tabulated for {sorted(unknown)}.")

    def bond_cutoff(self, centre: str, ligand: str) -> float:
        """Largest separation at which this centre and ligand are treated as bonded, in angstrom."""
        return self.bond_scale * (CovalentRadius.radius[centre] + CovalentRadius.radius[ligand])

    @property
    def search_radius(self) -> float:
        """The widest cutoff over all allowed pairs, which bounds the neighbour search."""
        return max(self.bond_cutoff(c, l) for c in self.centres for l in self.ligands)


@dataclass(frozen=True)
class Polyhedron:
    """
    One coordination polyhedron: a centre atom and the ligands bonded to it.

    :param centre_index: Index of the centre in the structure.
    :type centre_index: int
    :param centre: Element symbol of the centre.
    :type centre: str
    :param ligand: Element symbol of the ligands, which are all of one element.
    :type ligand: str
    :param coordination: Number of bonded ligands.
    :type coordination: int
    :param angle_deviation:
        Root-mean-square departure of the ligand-centre-ligand angles from the ideal
        tetrahedral 109.47 degrees. Only meaningful at coordination four; ``nan`` below two.
    :type angle_deviation: float
    :param mean_bond_length: Mean centre-to-ligand distance in angstrom.
    :type mean_bond_length: float
    :param bond_length_spread: Standard deviation of those distances in angstrom.
    :type bond_length_spread: float
    """

    centre_index: int
    centre: str
    ligand: str
    coordination: int
    angle_deviation: float
    mean_bond_length: float
    bond_length_spread: float

    @property
    def formula(self) -> str:
        """The polyhedron named the way a chemist would write it, such as ``GeS4``."""
        return f"{self.centre}{self.ligand}{self.coordination}"

    def is_tetrahedral(self, settings: PolyhedronSettings) -> bool:
        """Whether this is a four-coordinate polyhedron close enough to a regular tetrahedron."""
        return (
            self.coordination == 4
            and np.isfinite(self.angle_deviation)
            and self.angle_deviation <= settings.tetrahedral_tolerance
        )


@dataclass(frozen=True)
class PolyhedraCensus:
    """
    Every polyhedron in one structure, with the counts and the connectivity.

    :param polyhedra: The polyhedra, in order of their centre index.
    :type polyhedra: tuple[Polyhedron, ...]
    :param counts:
        How many of each named polyhedron, such as ``{"GeS4": 8, "PS4": 3}``. Malformed
        output shows up here directly as entries like ``GeS3`` or ``GeS6``.
    :type counts: Mapping[str, int]
    :param sharing:
        How many pairs of polyhedra share one, two or three ligands, keyed ``corner``,
        ``edge`` and ``face``, together with ``isolated``, the number of polyhedra that
        share nothing with anything.
    :type sharing: Mapping[str, int]
    :param shared_ligands:
        Number of ligands bonded to more than one centre, the quantity the sulfur budget
        of a formula forces to be non-zero.
    :type shared_ligands: int
    """

    polyhedra: tuple[Polyhedron, ...]
    counts: Mapping[str, int]
    sharing: Mapping[str, int]
    shared_ligands: int

    def tetrahedra(self, settings: PolyhedronSettings) -> tuple[Polyhedron, ...]:
        """The subset that are genuine tetrahedra under ``settings``."""
        return tuple(p for p in self.polyhedra if p.is_tetrahedral(settings))

    def ratio(self, settings: PolyhedronSettings) -> dict[str, int]:
        """
        Counts of the tetrahedra only, keyed by centre element.

        This is the design variable: ``{"Ge": 8, "P": 3}`` is the Li21Ge8P3S34 motif.
        """
        return dict(Counter(p.centre for p in self.tetrahedra(settings)))


def _bonds(structure: Structure, settings: PolyhedronSettings
           ) -> dict[int, list[tuple[int, tuple[int, int, int], float]]]:
    """
    Map each centre site to its bonded ligands, recording which periodic image each is in.

    The image matters. Two centres count as sharing a ligand only when they touch the same
    periodic copy of it, and a bare site index cannot tell those cases apart.
    """
    centres = set(settings.centres)
    ligands = set(settings.ligands)
    found: dict[int, list[tuple[int, tuple[int, int, int], float]]] = defaultdict(list)
    for index, site in enumerate(structure):
        symbol = site.specie.symbol
        if symbol not in centres:
            continue
        for neighbour in structure.get_neighbors(site, settings.search_radius):
            other = neighbour.specie.symbol
            if other not in ligands:
                continue
            if neighbour.nn_distance < settings.bond_cutoff(symbol, other):
                image = tuple(int(round(v)) for v in neighbour.image)
                found[index].append((neighbour.index, image, float(neighbour.nn_distance)))
    return found


def _angle_deviation(structure: Structure, centre_index: int,
                     ligands: Sequence[tuple[int, tuple[int, int, int], float]]) -> float:
    """Root-mean-square departure of the ligand-centre-ligand angles from 109.47 degrees."""
    if len(ligands) < 2:
        return float("nan")
    origin = structure[centre_index].coords
    lattice = structure.lattice
    vectors = []
    for index, image, _ in ligands:
        position = structure[index].coords + lattice.get_cartesian_coords(image)
        vectors.append(position - origin)
    squares = []
    for first, second in combinations(vectors, 2):
        norms = np.linalg.norm(first) * np.linalg.norm(second)
        if norms == 0.0:
            continue
        cosine = float(np.clip(np.dot(first, second) / norms, -1.0, 1.0))
        squares.append((degrees(acos(cosine)) - IDEAL_TETRAHEDRAL_ANGLE) ** 2)
    return sqrt(sum(squares) / len(squares)) if squares else float("nan")


def _sharing(bonds: Mapping[int, Sequence[tuple[int, tuple[int, int, int], float]]]
             ) -> tuple[Counter, set[int], int]:
    """
    Count how many ligands each pair of polyhedra holds in common, respecting periodicity.

    A centre ``c`` sitting in image ``T`` is bonded to the copy of ligand ``j`` in image
    ``image + T``. Two polyhedra therefore meet at a given ligand exactly when the
    difference of their recorded images matches the translation between them, so the pair
    is labelled by that difference and counted once per ligand they have in common.
    """
    per_ligand: dict[int, list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
    for centre, entries in bonds.items():
        for ligand_index, image, _ in entries:
            per_ligand[ligand_index].append((centre, image))

    shared_between: Counter = Counter()
    shared_ligands = 0
    for ligand_index, touching in per_ligand.items():
        if len(touching) < 2:
            continue
        shared_ligands += 1
        for (first, image_a), (second, image_b) in combinations(touching, 2):
            offset = tuple(a - b for a, b in zip(image_a, image_b))
            # Label the pair from the lower centre index so that a polyhedron and its own
            # periodic copy, and the two orderings of a genuine pair, are counted once.
            if (second, tuple(-v for v in offset)) < (first, offset):
                key = (second, first, tuple(-v for v in offset))
            else:
                key = (first, second, offset)
            shared_between[key] += 1

    involved = {centre for key in shared_between for centre in key[:2]}
    return shared_between, involved, shared_ligands


def census(structure: Structure, settings: PolyhedronSettings | None = None) -> PolyhedraCensus:
    """
    Count the coordination polyhedra of one structure and how they are connected.

    :param structure: The crystal to analyse.
    :type structure: Structure
    :param settings:
        Which polyhedra to look for. Defaults to tetrahedron formers coordinated by sulfur.
    :type settings: PolyhedronSettings | None

    :return: The polyhedra, their counts by formula, and the corner, edge and face sharing.
    :rtype: PolyhedraCensus
    """
    settings = settings or PolyhedronSettings()
    bonds = _bonds(structure, settings)

    polyhedra = []
    for centre_index in sorted(bonds):
        entries = bonds[centre_index]
        distances = [d for _, _, d in entries]
        polyhedra.append(Polyhedron(
            centre_index=centre_index,
            centre=structure[centre_index].specie.symbol,
            ligand=settings.ligands[0] if len(settings.ligands) == 1
            else structure[entries[0][0]].specie.symbol,
            coordination=len(entries),
            angle_deviation=_angle_deviation(structure, centre_index, entries),
            mean_bond_length=float(np.mean(distances)),
            bond_length_spread=float(np.std(distances)),
        ))

    shared_between, involved, shared_ligands = _sharing(bonds)
    sharing: Counter = Counter()
    for count in shared_between.values():
        sharing[SHARING_NAMES.get(min(count, 3), "face")] += 1
    sharing["isolated"] = sum(1 for p in polyhedra if p.centre_index not in involved)

    return PolyhedraCensus(
        polyhedra=tuple(polyhedra),
        counts=dict(Counter(p.formula for p in polyhedra)),
        sharing=dict(sharing),
        shared_ligands=shared_ligands,
    )


def summarise(structures: Iterable[Structure], settings: PolyhedronSettings | None = None
              ) -> dict[str, object]:
    """
    Pool a census over many structures, which is how a batch of generated samples is read.

    :param structures: The crystals to analyse.
    :type structures: Iterable[Structure]
    :param settings: Which polyhedra to look for.
    :type settings: PolyhedronSettings | None

    :return:
        Totals, the mean number of tetrahedra per structure, the share of structures in
        which every centre reached a well-formed tetrahedron, and the pooled sharing.
    :rtype: dict[str, object]
    """
    settings = settings or PolyhedronSettings()
    formulas: Counter = Counter()
    sharing: Counter = Counter()
    ratios: Counter = Counter()
    deviations: list[float] = []
    lengths: list[float] = []
    complete = 0
    total = 0

    for structure in structures:
        total += 1
        result = census(structure, settings)
        formulas.update(result.counts)
        sharing.update(result.sharing)
        tetrahedra = result.tetrahedra(settings)
        deviations.extend(p.angle_deviation for p in tetrahedra)
        lengths.extend(p.mean_bond_length for p in tetrahedra)
        expected = sum(1 for site in structure if site.specie.symbol in settings.centres)
        if expected and len(tetrahedra) == expected:
            complete += 1
            ratios[_ratio_key(result.ratio(settings))] += 1

    return {
        "structures": total,
        "polyhedra_by_formula": dict(formulas),
        "sharing": dict(sharing),
        "fully_tetrahedral_structures": complete,
        "fully_tetrahedral_fraction": complete / total if total else 0.0,
        "mean_angle_deviation": float(np.mean(deviations)) if deviations else float("nan"),
        "mean_bond_length": float(np.mean(lengths)) if lengths else float("nan"),
        "ratios_among_fully_tetrahedral": dict(ratios),
    }


def _ratio_key(ratio: Mapping[str, int]) -> str:
    """Name a tetrahedron ratio the way the literature does, such as ``Ge8:P3``."""
    return ":".join(f"{element}{ratio[element]}" for element in sorted(ratio)) or "none"
