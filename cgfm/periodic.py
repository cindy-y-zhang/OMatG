"""
Exact minimum-image geometry in a periodic cell.

Every distance in the block pipeline is a distance between periodic images, and the obvious way to compute one is wrong.
Subtracting the rounded fractional coordinate finds the image inside the fractional unit cube, which is the nearest image
only when the cube is also the Wigner-Seitz cell of the lattice. That holds for a cube and fails for anything sheared:
on the first hundred MPTS-52 validation cells the rounding rule overestimated 11.9 per cent of atom-pair distances, on 66
of the 100 cells, by a median of 0.70 Angstrom and by as much as 6.67. Those errors are larger than the site tolerance
the match rate is scored at and larger than the separation the readout clusters votes by, so they corrupt the
decomposition, the reconstruction and the readout alike.

The fix is to search. After reducing the basis so that its vectors are short and close to orthogonal, the nearest image
differs from the rounded one by at most one step along each reduced axis, so trying the twenty-seven neighbouring images
and keeping the shortest is exact. The reduction is the expensive part and depends only on the cell, so a
:class:`Cell` computes it once and every distance taken against that cell reuses it.
"""

from dataclasses import dataclass
import itertools
import numpy as np
from pymatgen.core import Lattice


IMAGE_RANGE = 1
"""
Number of steps along each reduced axis that the image search covers, giving (2 * IMAGE_RANGE + 1) ** 3 candidates.

One step is enough. In three dimensions a reduced basis has the property that the closest lattice point to any target
sits within one step of the rounded coordinate along every axis, so widening the search cannot change an answer. This is
asserted rather than assumed: the tests compare the result against pymatgen's exact distances on strongly sheared cells
and on real MPTS-52 lattices, and separately check that a two-step search returns the same distances.
"""


@dataclass(frozen=True)
class Cell:
    """
    A periodic cell that knows how to measure inside itself.

    :param matrix:
        The cell vectors as given, of shape (3, 3), with row i the i-th lattice vector. Kept so that a caller can
        recover the cell it passed in, and unused by the geometry, which works in the reduced basis.
    :type matrix: numpy.ndarray
    :param basis:
        A reduced basis of the same lattice, of shape (3, 3). Spans exactly the same set of lattice points as the
        matrix, so a displacement reduced against it is a valid displacement of the original cell.
    :type basis: numpy.ndarray
    :param inverse:
        Inverse of the reduced basis, of shape (3, 3).
    :type inverse: numpy.ndarray
    :param images:
        Cartesian offsets of the candidate images, of shape (K, 3), with the zero offset first.
    :type images: numpy.ndarray
    """

    matrix: np.ndarray
    basis: np.ndarray
    inverse: np.ndarray
    images: np.ndarray

    @staticmethod
    def of(matrix: np.ndarray) -> "Cell":
        """
        Build a cell from its lattice vectors, paying the basis reduction once.

        :param matrix:
            Cell vectors of shape (3, 3), with row i the i-th lattice vector.
        :type matrix: numpy.ndarray

        :return:
            The cell.
        :rtype: Cell
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        basis = np.array(Lattice(matrix).get_lll_reduced_lattice().matrix, dtype=np.float64)
        steps = range(-IMAGE_RANGE, IMAGE_RANGE + 1)
        offsets = np.array(sorted(itertools.product(steps, repeat=3), key=lambda o: np.abs(o).sum()), dtype=np.float64)
        return Cell(matrix=matrix, basis=basis, inverse=np.linalg.inv(basis), images=offsets @ basis)

    def minimum_image(self, vectors: np.ndarray) -> np.ndarray:
        """
        Reduce Cartesian displacements to their shortest periodic image.

        :param vectors:
            Cartesian displacements of shape (..., 3).
        :type vectors: numpy.ndarray

        :return:
            The shortest equivalent displacements, of the same shape.
        :rtype: numpy.ndarray
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        fractional = vectors @ self.inverse
        folded = (fractional - np.round(fractional)) @ self.basis
        candidates = folded[..., None, :] + self.images
        chosen = np.argmin(np.sum(candidates * candidates, axis=-1), axis=-1)
        return np.take_along_axis(candidates, chosen[..., None, None], axis=-2)[..., 0, :]

    def distances(self, points: np.ndarray, others: np.ndarray = None) -> np.ndarray:
        """
        Measure every minimum-image distance between two sets of points, or within one set.

        :param points:
            Cartesian positions of shape (n, 3).
        :type points: numpy.ndarray
        :param others:
            Cartesian positions of shape (m, 3). Defaults to the first set, giving the symmetric distance matrix with a
            zero diagonal.
        :type others: numpy.ndarray

        :return:
            Distances in Angstrom, of shape (n, m).
        :rtype: numpy.ndarray
        """
        others = points if others is None else others
        return np.linalg.norm(self.minimum_image(np.asarray(points)[:, None, :] - np.asarray(others)[None, :, :]),
                              axis=-1)

    def mean(self, points: np.ndarray) -> np.ndarray:
        """
        Average positions that may sit in different periodic images of the same neighbourhood.

        Averaging coordinates directly would place the mean of two sites straddling a cell boundary in the middle of the
        cell rather than between them. Taking the mean of the displacements from an arbitrary member and adding it back
        keeps every contribution in one image, which is correct whenever the points are closer to each other than to
        their own images, and that is what a cluster of votes for one atom is.

        :param points:
            Cartesian positions of shape (n, 3), assumed to lie in one neighbourhood.
        :type points: numpy.ndarray

        :return:
            The mean position, of shape (3,).
        :rtype: numpy.ndarray
        """
        points = np.asarray(points, dtype=np.float64)
        return points[0] + np.mean(self.minimum_image(points - points[0]), axis=0)
