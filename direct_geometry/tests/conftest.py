"""Shared fixtures and geometry builders for the direct-geometry tests."""

from typing import Optional, Sequence
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_environment():
    """
    Run every test in double precision from a fixed global random state.

    Double precision because most of these tests assert an algebraic identity -- a rotation invariance, a moment expansion
    against a brute-force pair sum -- to a tight tolerance, and there is no reason to fight single-precision noise while
    checking algebra. The production runs are ``32-true``, and the one place that matters is the cost audit, which measures
    the real dtype on the real card rather than being a test.

    Seeding matters because module constructors draw their weights from the global generator. Without it a test would
    depend on how much randomness the tests before it consumed, so adding an unrelated test could turn this one red.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


# Spelled out in double rather than left to the default, because these are built when the module is imported and the
# fixture above has not run yet. Inheriting the ambient default would make them single precision and every comparison
# against a double-precision structure a dtype error -- which is how the upstream builder's hardcoded ``torch.float``
# surfaces, so it is worth not reproducing here.
CUBIC = torch.tensor([[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]], dtype=torch.float64)
"""A plain cube, where the answer can be worked out by hand."""

SKEWED = torch.tensor([[6.0, 0.0, 0.0], [2.6, 5.4, 0.0], [1.9, 1.4, 5.1]], dtype=torch.float64)
"""
Deliberately far from orthogonal.

This is where a fold in fractional space and the true minimum image part company, and where a repetition count derived
from cell *lengths* rather than interplanar spacings is wrong.
"""

SHORT = torch.tensor([[2.4, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.6]], dtype=torch.float64)
"""
A cell far shorter than a 6 Angstrom cutoff in every direction.

Needs two repetitions per axis, so 125 images. The upstream ``radius_graph_pbc`` hardcodes one repetition and would miss
most of the neighbourhood, silently.
"""

FLAT = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]], dtype=torch.float64)
"""A cell whose third vector lies in the plane of the first two, so it has no volume and no defined images."""


def single(frac: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Package one structure as the batched arguments the neighbour builder takes.

    :param frac:
        Fractional coordinates of shape ``(atoms, 3)``.
    :type frac: torch.Tensor
    :param cell:
        Lattice of shape ``(3, 3)``.
    :type cell: torch.Tensor

    :return:
        Fractional coordinates, lattices and atom counts.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    return frac, cell.unsqueeze(0), torch.tensor([frac.shape[0]])


def random_structure(num_atoms: int, cell: torch.Tensor = CUBIC,
                     seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one structure with random fractional coordinates.

    :param num_atoms:
        Number of atoms.
    :type num_atoms: int
    :param cell:
        Lattice of shape ``(3, 3)``.
        Defaults to CUBIC.
    :type cell: torch.Tensor
    :param seed:
        Seed of the coordinates.
        Defaults to 0.
    :type seed: int

    :return:
        Fractional coordinates, lattices and atom counts.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    generator = torch.Generator().manual_seed(seed)
    return single(torch.rand((num_atoms, 3), generator=generator), cell)


def random_batch(sizes: Sequence[int], cells: Optional[Sequence[torch.Tensor]] = None,
                 seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a batch of structures with random fractional coordinates.

    :param sizes:
        Number of atoms of every structure.
    :type sizes: Sequence[int]
    :param cells:
        Lattice of every structure, or None for cubes of varying size.
        Defaults to None.
    :type cells: Optional[Sequence[torch.Tensor]]
    :param seed:
        Seed of the coordinates.
        Defaults to 0.
    :type seed: int

    :return:
        Fractional coordinates, lattices and atom counts.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    generator = torch.Generator().manual_seed(seed)
    if cells is None:
        cells = [torch.eye(3) * (4.0 + 0.7 * index) for index in range(len(sizes))]
    frac = torch.rand((sum(sizes), 3), generator=generator)
    return frac, torch.stack(list(cells)), torch.tensor(list(sizes))


def cluster(directions: Sequence[Sequence[float]], radius: float = 2.0,
            box: float = 60.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an isolated coordination polyhedron: one central atom and its neighbours, in a box large enough to be aperiodic.

    The box is ten times the descriptor cutoff, so every periodic image is far outside it and the central atom's
    environment is exactly the polyhedron asked for. That is what makes a hand-computed invariant a usable assertion.

    :param directions:
        Directions of the neighbours, normalised internally.
    :type directions: Sequence[Sequence[float]]
    :param radius:
        Distance of every neighbour from the centre, in Angstrom.
        Defaults to 2.0.
    :type radius: float
    :param box:
        Side of the cubic box, in Angstrom.
        Defaults to 60.0.
    :type box: float

    :return:
        Fractional coordinates, lattices and atom counts. The central atom is index zero.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    cell = torch.eye(3) * box
    if not directions:
        return single(torch.tensor([[0.5, 0.5, 0.5]]), cell)
    unit = torch.tensor(directions, dtype=torch.get_default_dtype())
    unit = unit / unit.norm(dim=-1, keepdim=True)
    cartesian = torch.cat([torch.zeros(1, 3), radius * unit], dim=0) + box / 2.0
    return single(cartesian / box, cell)


TETRAHEDRAL = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
"""Four neighbours at the tetrahedral angle, whose pairwise cosine is -1/3."""

SQUARE_PLANAR = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
"""Four neighbours in a plane. Same count and same radius as the tetrahedron, so the radial block cannot tell them apart."""

OCTAHEDRAL = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
"""Six neighbours on the axes."""

BODY_CENTRED = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]
"""Eight neighbours at the body-centred-cubic angles."""


def supercell(frac: torch.Tensor, cell: torch.Tensor,
              repeats: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tile a structure into a supercell.

    Every atom of the original appears once per tile and describes exactly the same physical environment, so a descriptor
    that is not invariant here is reading the cell rather than the crystal.

    :param frac:
        Fractional coordinates of shape ``(atoms, 3)``.
    :type frac: torch.Tensor
    :param cell:
        Lattice of shape ``(3, 3)``.
    :type cell: torch.Tensor
    :param repeats:
        Number of tiles along each axis.
        Defaults to 2.
    :type repeats: int

    :return:
        Fractional coordinates, lattices and atom counts of the supercell. The first ``atoms`` entries correspond, in
        order, to the atoms of the original.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    span = torch.arange(repeats, dtype=frac.dtype)
    tiles = torch.stack(torch.meshgrid(span, span, span, indexing="ij"), dim=-1).reshape(-1, 3)
    tiled = ((frac.unsqueeze(0) + tiles.unsqueeze(1)) / repeats).reshape(-1, 3)
    return single(tiled, cell * repeats)


def rotation(angles: Sequence[float]) -> torch.Tensor:
    """
    Build a rotation matrix from three successive axis rotations.

    :param angles:
        Rotation angles about the x, y and z axes, in radians.
    :type angles: Sequence[float]

    :return:
        Rotation matrix of shape ``(3, 3)``.
    :rtype: torch.Tensor
    """
    matrix = torch.eye(3, dtype=torch.get_default_dtype())
    for axis, angle in enumerate(angles):
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        step = torch.eye(3, dtype=matrix.dtype)
        first, second = (axis + 1) % 3, (axis + 2) % 3
        step[first, first], step[second, second] = cosine, cosine
        step[first, second], step[second, first] = -sine, sine
        matrix = matrix @ step
    return matrix
