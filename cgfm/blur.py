"""
Group-mean (blur) operator on batched periodic point clouds.

A partition of a structure's atoms into groups defines a linear operator S that replaces each atom's value by the mean
value of its group. Writing the assignment as a row-stochastic matrix A of shape (N, K), where A[i, m] is the weight
with which atom i belongs to group m, the operator is

    S = A diag(A^T 1)^-1 A^T,

which is symmetric, row-stochastic and mean-preserving. For a hard partition (A is 0/1) S is the block-averaging matrix
and is idempotent; for a soft A it is a smooth relaxation of one.

The coarse-to-fine path only needs the single combination

    Delta = (2 S - I) d = (coarse part of d) - (fine part of d),

centred so that its per-structure mean vanishes. Centring keeps the whole construction inside the translation quotient
that the translation-invariant encoder and the centre-of-mass velocity correction already work in, so switching the
path on never introduces a spurious rigid translation relative to the atomwise baseline.

Both hard and soft assignments go through the same code path. With N <= 52 atoms and K <= 11 groups the padded (N, K)
form costs nothing, and one-hot assignments reproduce hard partitions exactly, so there is no reason to maintain a
separate scatter-based implementation.
"""

from typing import Optional
import torch


_WEIGHT_FLOOR = 1.0e-12
"""Lower bound on a group's total weight, guarding the division for padded (all-zero) group columns."""


def structure_mean(values: torch.Tensor, batch: torch.Tensor, num_structures: int) -> torch.Tensor:
    """
    Average the given per-atom values over the atoms of each structure.

    :param values:
        Per-atom values of shape (N, D).
    :type values: torch.Tensor
    :param batch:
        Structure index of every atom, of shape (N,).
    :type batch: torch.Tensor
    :param num_structures:
        Number of structures B in the batch.
    :type num_structures: int

    :return:
        Per-structure means of shape (B, D).
    :rtype: torch.Tensor
    """
    totals = values.new_zeros((num_structures, values.shape[-1])).index_add(0, batch, values)
    counts = values.new_zeros((num_structures,)).index_add(0, batch, torch.ones_like(batch, dtype=values.dtype))
    return totals / counts.clamp_min(1.0).unsqueeze(-1)


def apply_group_mean(values: torch.Tensor, assignment: torch.Tensor, batch: torch.Tensor,
                     num_structures: int) -> torch.Tensor:
    """
    Apply the group-mean operator S to per-atom values.

    :param values:
        Per-atom values of shape (N, D).
    :type values: torch.Tensor
    :param assignment:
        Row-stochastic assignment of shape (N, K). Columns beyond a structure's number of groups must be zero for every
        atom of that structure.
    :type assignment: torch.Tensor
    :param batch:
        Structure index of every atom, of shape (N,).
    :type batch: torch.Tensor
    :param num_structures:
        Number of structures B in the batch.
    :type num_structures: int

    :return:
        Group means broadcast back to the atoms, of shape (N, D).
    :rtype: torch.Tensor
    """
    num_groups = assignment.shape[1]
    dim = values.shape[-1]
    # Weighted sums and total weights per (structure, group). Group columns are local to a structure, so the scatter
    # index is the structure index and the group axis is kept dense.
    weighted = assignment.unsqueeze(-1) * values.unsqueeze(1)  # (N, K, D)
    numerator = values.new_zeros((num_structures, num_groups, dim)).index_add(0, batch, weighted)
    denominator = values.new_zeros((num_structures, num_groups)).index_add(0, batch, assignment)
    centroids = numerator / denominator.clamp_min(_WEIGHT_FLOOR).unsqueeze(-1)  # (B, K, D)
    return (assignment.unsqueeze(-1) * centroids[batch]).sum(dim=1)


def coarse_fine_delta(displacement: torch.Tensor, assignment: torch.Tensor, batch: torch.Tensor,
                      num_structures: Optional[int] = None) -> torch.Tensor:
    """
    Compute the centred coarse-minus-fine displacement Delta = (2 S - I) d.

    :param displacement:
        Per-atom tangent-space displacement d of shape (N, 3).
    :type displacement: torch.Tensor
    :param assignment:
        Row-stochastic assignment of shape (N, K).
    :type assignment: torch.Tensor
    :param batch:
        Structure index of every atom, of shape (N,).
    :type batch: torch.Tensor
    :param num_structures:
        Number of structures B in the batch. Inferred from batch if None.
    :type num_structures: Optional[int]

    :return:
        Centred Delta of shape (N, 3).
    :rtype: torch.Tensor

    :raises ValueError:
        If the shapes of the arguments are inconsistent.
    """
    if displacement.shape[0] != assignment.shape[0] or displacement.shape[0] != batch.shape[0]:
        raise ValueError("The displacement, assignment and batch tensors must agree in their number of atoms.")
    if num_structures is None:
        num_structures = int(batch.max()) + 1 if batch.numel() > 0 else 0
    coarse = apply_group_mean(displacement, assignment, batch, num_structures)
    delta = 2.0 * coarse - displacement
    return delta - structure_mean(delta, batch, num_structures)[batch]


def fine_energy_fraction(displacement: torch.Tensor, assignment: torch.Tensor, batch: torch.Tensor,
                         num_structures: Optional[int] = None) -> torch.Tensor:
    """
    Compute the fraction of squared displacement that lives within groups, ||(I - S) d||^2 / ||d||^2.

    This is the quantity a grouping network is tempted to shrink in order to lower the flow-matching loss without
    finding better groups, so it is the primary collapse diagnostic.

    :param displacement:
        Per-atom tangent-space displacement d of shape (N, 3).
    :type displacement: torch.Tensor
    :param assignment:
        Row-stochastic assignment of shape (N, K).
    :type assignment: torch.Tensor
    :param batch:
        Structure index of every atom, of shape (N,).
    :type batch: torch.Tensor
    :param num_structures:
        Number of structures B in the batch. Inferred from batch if None.
    :type num_structures: Optional[int]

    :return:
        Scalar fraction in [0, 1].
    :rtype: torch.Tensor
    """
    if num_structures is None:
        num_structures = int(batch.max()) + 1 if batch.numel() > 0 else 0
    coarse = apply_group_mean(displacement, assignment, batch, num_structures)
    fine = displacement - coarse
    return (fine ** 2).sum() / (displacement ** 2).sum().clamp_min(_WEIGHT_FLOOR)


def one_hot_assignment(group: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Build a one-hot assignment matrix from per-atom group labels that are local to each structure.

    :param group:
        Structure-local group label of every atom, of shape (N,).
    :type group: torch.Tensor
    :param num_groups:
        Number of columns K of the returned matrix, at least the largest label plus one.
    :type num_groups: int

    :return:
        One-hot assignment of shape (N, K).
    :rtype: torch.Tensor
    """
    return torch.nn.functional.one_hot(group, num_classes=num_groups).to(torch.get_default_dtype())
