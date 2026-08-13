"""Tests for exact minimum-image geometry."""

import itertools
import numpy as np
import pytest
from pymatgen.core import Lattice

from cgfm.periodic import Cell, IMAGE_RANGE


DEGENERACY_FLOOR = 0.02
"""
Smallest cell volume, as a fraction of the product of the three edge lengths, that counts as a cell rather than a plane.

Cells are drawn at random to get shear, and a random basis is occasionally almost flat. Its reduction is numerically
meaningless and it is not a crystal, so testing against it would measure the reduction's behaviour on garbage. Note that
pymatgen's Lattice.from_parameters does not raise on an angle triple that fails to close; it returns a matrix of
determinant around 1e-14, which is why cells here are built from a basis and screened rather than from parameters.
"""


def sheared_cells(count: int = 40) -> list[np.ndarray]:
    """Random cells spanning mild to extreme shear, which is where the rounding rule fails."""
    rng = np.random.default_rng(20260813)
    cells = [np.diag([4.0, 5.0, 6.0]), np.array([[4.0, 0.0, 0.0], [3.9, 1.0, 0.0], [0.0, 0.0, 6.0]])]
    while len(cells) < count:
        basis = rng.normal(size=(3, 3))
        lengths = rng.uniform(3.0, 12.0, size=3)
        basis = basis / np.linalg.norm(basis, axis=1, keepdims=True) * lengths[:, None]
        if abs(np.linalg.det(basis)) > DEGENERACY_FLOOR * np.prod(lengths):
            cells.append(basis)
    return cells


def random_points(cell: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Cartesian positions drawn uniformly over one cell."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(count, 3)) @ cell


@pytest.mark.parametrize("matrix", sheared_cells())
def test_distances_match_pymatgen_exactly(matrix: np.ndarray) -> None:
    """The whole point of the module: agreement with a reference implementation on sheared cells."""
    points = random_points(matrix, 12, seed=1)
    lattice = Lattice(matrix)
    expected = lattice.get_all_distances(lattice.get_fractional_coords(points), lattice.get_fractional_coords(points))
    assert Cell.of(matrix).distances(points) == pytest.approx(expected, abs=1e-8)


def test_fractional_rounding_is_not_enough() -> None:
    """The rule this module replaces really does overestimate, so the test above is not vacuous."""
    matrix = np.array([[4.0, 0.0, 0.0], [3.9, 1.0, 0.0], [0.0, 0.0, 6.0]])
    points = random_points(matrix, 12, seed=2)
    fractional = (points[:, None, :] - points[None, :, :]) @ np.linalg.inv(matrix)
    rounded = np.linalg.norm((fractional - np.round(fractional)) @ matrix, axis=-1)
    assert np.any(rounded > Cell.of(matrix).distances(points) + 1e-6)


@pytest.mark.parametrize("matrix", sheared_cells(12))
def test_one_step_of_image_search_is_exhaustive(matrix: np.ndarray) -> None:
    """IMAGE_RANGE is a claim about reduced bases, so widening the search must change nothing."""
    cell = Cell.of(matrix)
    points = random_points(matrix, 10, seed=3)
    vectors = points[:, None, :] - points[None, :, :]
    steps = range(-IMAGE_RANGE - 1, IMAGE_RANGE + 2)
    wider = np.array(list(itertools.product(steps, repeat=3)), dtype=np.float64) @ cell.basis
    fractional = vectors @ cell.inverse
    folded = (fractional - np.round(fractional)) @ cell.basis
    candidates = folded[..., None, :] + wider
    assert np.min(np.linalg.norm(candidates, axis=-1), axis=-1) == pytest.approx(cell.distances(points), abs=1e-9)


def test_minimum_image_returns_a_translate_of_its_input() -> None:
    """A shortened displacement has to differ from the original by a lattice vector, or it is a different displacement."""
    matrix = sheared_cells()[7]
    cell = Cell.of(matrix)
    vectors = np.random.default_rng(4).uniform(-3.0, 3.0, size=(50, 3)) @ matrix
    offsets = (cell.minimum_image(vectors) - vectors) @ np.linalg.inv(matrix)
    assert offsets == pytest.approx(np.round(offsets), abs=1e-8)


def test_minimum_image_is_no_longer_than_the_input() -> None:
    """Reduction can only shorten, since the input is always one of the candidates."""
    matrix = np.array(Lattice.from_parameters(6.0, 6.0, 6.0, 40.0, 50.0, 60.0).matrix)
    cell = Cell.of(matrix)
    vectors = np.random.default_rng(5).uniform(-2.0, 2.0, size=(60, 3)) @ matrix
    assert np.all(np.linalg.norm(cell.minimum_image(vectors), axis=-1) <= np.linalg.norm(vectors, axis=-1) + 1e-9)


def test_minimum_image_preserves_shape() -> None:
    """Callers pass matrices of displacements, not just lists of them."""
    cell = Cell.of(np.diag([4.0, 5.0, 6.0]))
    assert cell.minimum_image(np.zeros((7, 3, 3))).shape == (7, 3, 3)


def test_mean_averages_across_a_cell_boundary() -> None:
    """Two sites straddling a boundary average to the point between them, not to the middle of the cell."""
    cell = Cell.of(np.diag([10.0, 10.0, 10.0]))
    mean = cell.mean(np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]]))
    assert cell.distances(mean[None, :], np.array([[0.0, 0.0, 0.0]]))[0, 0] == pytest.approx(0.0, abs=1e-8)


def test_mean_of_one_point_is_that_point() -> None:
    """A cluster of one has no averaging to do."""
    cell = Cell.of(np.diag([4.0, 5.0, 6.0]))
    point = np.array([[1.0, 2.0, 3.0]])
    assert cell.mean(point) == pytest.approx(point[0])


@pytest.mark.parametrize("matrix", sheared_cells(12))
def test_reduced_basis_spans_the_original_lattice(matrix: np.ndarray) -> None:
    """The reduction must change the basis without changing the lattice, or every distance is against the wrong cell."""
    transform = Cell.of(matrix).basis @ np.linalg.inv(matrix)
    assert transform == pytest.approx(np.round(transform), abs=1e-8)
    assert abs(round(np.linalg.det(transform))) == 1
