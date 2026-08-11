"""
Dense periodic neighbourhood utilities for the grouping network.

Structures in the benchmarks used here hold at most 52 atoms, so the full pairwise distance matrix of a padded batch is
small enough to build directly. Working densely keeps the grouping network short and obviously permutation-equivariant,
and the neighbour lists are then just a top-k over that matrix.

Separations use the minimum image of the fractional difference mapped through the cell. That is the same convention the
periodic interpolant uses, and for the unit cells in these datasets it agrees with the true periodic distance for the
cutoffs of interest.
"""

import torch
from torch_geometric.utils import to_dense_batch


def dense_positions(frac: torch.Tensor, batch: torch.Tensor,
                    num_structures: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad per-atom fractional coordinates into a dense batch.

    :param frac:
        Fractional coordinates of shape (N, 3).
    :type frac: torch.Tensor
    :param batch:
        Structure index of every atom, of shape (N,).
    :type batch: torch.Tensor
    :param num_structures:
        Number of structures B in the batch.
    :type num_structures: int

    :return:
        Padded coordinates of shape (B, Nmax, 3) and a validity mask of shape (B, Nmax).
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    return to_dense_batch(frac, batch, batch_size=num_structures)


def min_image_displacements(frac_dense: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    Compute minimum-image Cartesian displacement vectors between every pair of atoms.

    :param frac_dense:
        Padded fractional coordinates of shape (B, Nmax, 3).
    :type frac_dense: torch.Tensor
    :param cell:
        Cell vectors of shape (B, 3, 3), with row i the i-th lattice vector.
    :type cell: torch.Tensor

    :return:
        Displacements of shape (B, Nmax, Nmax, 3), where entry [b, i, j] points from atom i to atom j.
    :rtype: torch.Tensor
    """
    diff = frac_dense.unsqueeze(1) - frac_dense.unsqueeze(2)  # (B, Nmax, Nmax, 3), [b, i, j] = frac_j - frac_i
    diff = diff - torch.round(diff)
    return torch.einsum('bijk,bkl->bijl', diff, cell)


def pair_distances(frac_dense: torch.Tensor, cell: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute minimum-image Cartesian distances between every pair of atoms.

    Pairs involving a padded slot, and the diagonal, are set to infinity so that they are never selected as neighbours.

    :param frac_dense:
        Padded fractional coordinates of shape (B, Nmax, 3).
    :type frac_dense: torch.Tensor
    :param cell:
        Cell vectors of shape (B, 3, 3).
    :type cell: torch.Tensor
    :param mask:
        Validity mask of shape (B, Nmax).
    :type mask: torch.Tensor

    :return:
        Distances of shape (B, Nmax, Nmax).
    :rtype: torch.Tensor
    """
    distances = min_image_displacements(frac_dense, cell).norm(dim=-1)
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    eye = torch.eye(distances.shape[1], dtype=torch.bool, device=distances.device).unsqueeze(0)
    return distances.masked_fill(~pair_mask | eye, float('inf'))


def neighbour_lists(distances: torch.Tensor, cutoff: float,
                    max_neighbours: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select the nearest neighbours of every atom within a cutoff.

    :param distances:
        Pairwise distances of shape (B, Nmax, Nmax) with invalid pairs set to infinity.
    :type distances: torch.Tensor
    :param cutoff:
        Neighbour cutoff in Angstrom.
    :type cutoff: float
    :param max_neighbours:
        Largest number of neighbours kept per atom.
    :type max_neighbours: int

    :return:
        Neighbour indices of shape (B, Nmax, M) and a validity mask of the same shape, where M is the smaller of
        max_neighbours and Nmax.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    keep = min(max_neighbours, distances.shape[-1])
    neighbour_distances, neighbour_index = torch.topk(distances, k=keep, dim=-1, largest=False)
    return neighbour_index, torch.isfinite(neighbour_distances) & (neighbour_distances <= cutoff)


def gaussian_basis(distances: torch.Tensor, num_basis: int, cutoff: float) -> torch.Tensor:
    """
    Expand distances in equally spaced Gaussian radial basis functions.

    :param distances:
        Distances of any shape.
    :type distances: torch.Tensor
    :param num_basis:
        Number of basis functions.
    :type num_basis: int
    :param cutoff:
        Distance at which the last basis function is centred, in Angstrom.
    :type cutoff: float

    :return:
        Basis expansion with one extra trailing dimension of size num_basis.
    :rtype: torch.Tensor
    """
    centres = torch.linspace(0.0, cutoff, num_basis, dtype=distances.dtype, device=distances.device)
    width = cutoff / max(num_basis - 1, 1)
    return torch.exp(-((distances.unsqueeze(-1) - centres) / width) ** 2)


def atom_group_extents(distances: torch.Tensor, labels_dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute, for every atom, the distance to the furthest other atom of its own group.

    Taking the maximum of this quantity over the atoms of a group recovers that group's diameter, so the mean over
    atoms measures typical group extent and the maximum over the batch is the largest group diameter.

    :param distances:
        Pairwise distances of shape (B, Nmax, Nmax) with invalid pairs set to infinity.
    :type distances: torch.Tensor
    :param labels_dense:
        Padded structure-local group labels of shape (B, Nmax).
    :type labels_dense: torch.Tensor
    :param mask:
        Validity mask of shape (B, Nmax).
    :type mask: torch.Tensor

    :return:
        Flat tensor of one extent per real atom, in Angstrom. Atoms in singleton groups contribute zero.
    :rtype: torch.Tensor
    """
    same_group = labels_dense.unsqueeze(1) == labels_dense.unsqueeze(2)
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2) & same_group
    # The diagonal and padded pairs are infinite in distances, so they are replaced by zero before the maximum. A
    # singleton group therefore yields an extent of zero.
    within_group = torch.where(pair_mask & torch.isfinite(distances), distances, torch.zeros_like(distances))
    return within_group.max(dim=-1).values[mask]
