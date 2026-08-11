"""
Producers of atom-to-group assignments for the coarse-to-fine interpolant.

A grouping turns a batch of clean target structures into a row-stochastic assignment matrix of shape (N, K), where N is
the total number of atoms in the batch and K is the largest number of groups of any structure in the batch. Structures
with fewer groups leave the trailing columns at zero.

Groupings are only used to build training paths. They are never evaluated during generation, so a grouping that needs
expensive chemistry (such as the CrystalNN coordination shells) can be precomputed offline without making the resulting
model any harder to sample from.
"""

from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data
from .blur import one_hot_assignment


GROUP_FIELD = "cg_group"
"""Name of the per-atom OMGData attribute holding structure-local group labels."""

NUM_GROUPS_FIELD = "cg_n_groups"
"""Name of the per-structure OMGData attribute holding the number of groups."""

REFERENCE_METHODS = ("kmedoids", "shells")
"""
Fixed partition methods that every batch carries for comparison.

The learned arm is only interesting if its partition can be told apart from geometric clustering, which is what the
flow-matching objective rewards on its own, and from coordination shells, which is the hypothesis under test. Both
comparisons need both partitions present on the same batch, so both are attached regardless of which one defines the
path of the arm being trained.
"""


def reference_field(method: str) -> str:
    """
    Return the name of the per-atom attribute holding a fixed reference partition.

    :param method:
        Name of the partition method.
    :type method: str

    :return:
        Name of the OMGData attribute.
    :rtype: str
    """
    return f"cg_ref_{method}"


class Grouping(ABC):
    """
    Abstract producer of atom-to-group assignments.
    """

    @abstractmethod
    def assignment(self, x_1: Data) -> torch.Tensor:
        """
        Return the row-stochastic assignment matrix for a batch of clean target structures.

        :param x_1:
            Batch of clean target structures.
        :type x_1: torch_geometric.data.Data

        :return:
            Assignment of shape (N, K) whose rows sum to one.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def is_hard(self) -> bool:
        """
        Whether this grouping always returns one-hot assignments.

        :return:
            Whether the assignment is a hard partition.
        :rtype: bool
        """
        raise NotImplementedError


def read_num_groups(x_1: Data) -> torch.Tensor:
    """
    Read the per-structure number of groups from a batch, raising a helpful error when it is absent.

    :param x_1:
        Batch of clean target structures.
    :type x_1: torch_geometric.data.Data

    :return:
        Number of groups of every structure, of shape (B,).
    :rtype: torch.Tensor

    :raises ValueError:
        If the batch does not carry the number of groups.
    """
    num_groups = getattr(x_1, NUM_GROUPS_FIELD, None)
    if num_groups is None:
        raise ValueError(
            f"The batch does not carry the '{NUM_GROUPS_FIELD}' attribute. Coarse-to-fine arms must be run with "
            f"cgfm.data.CGDataModule and a group file produced by cgfm/scripts/precompute_groups.py.")
    return num_groups.reshape(-1)


class PrecomputedGrouping(Grouping):
    """
    Hard grouping that reads structure-local group labels attached to the batch by cgfm.data.CGOMGDataset.

    This backs both fixed arms of the experiment. Which partition is used (periodic k-medoids or CrystalNN coordination
    shells) is decided by the group file that the data module loads, not by this class, so the two arms differ only in
    their data configuration.
    """

    def assignment(self, x_1: Data) -> torch.Tensor:
        """
        Build a one-hot assignment from the precomputed labels on the batch.

        :param x_1:
            Batch of clean target structures.
        :type x_1: torch_geometric.data.Data

        :return:
            One-hot assignment of shape (N, K).
        :rtype: torch.Tensor

        :raises ValueError:
            If the batch does not carry group labels, or a label exceeds its structure's number of groups.
        """
        group = getattr(x_1, GROUP_FIELD, None)
        if group is None:
            raise ValueError(
                f"The batch does not carry the '{GROUP_FIELD}' attribute. Coarse-to-fine arms with precomputed "
                f"groups must be run with cgfm.data.CGDataModule and a group file produced by "
                f"cgfm/scripts/precompute_groups.py.")
        group = group.reshape(-1).long()
        num_groups = read_num_groups(x_1)
        if torch.any(group >= num_groups[x_1.batch]) or torch.any(group < 0):
            raise ValueError("Precomputed group labels must lie in [0, number of groups) of their own structure.")
        return one_hot_assignment(group, int(num_groups.max()))

    def is_hard(self) -> bool:
        """
        Whether this grouping always returns one-hot assignments.

        :return:
            Always True.
        :rtype: bool
        """
        return True
