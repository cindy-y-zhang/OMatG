"""
On-disk format for precomputed partitions.

One file holds the partition of every structure of one dataset split under one method, stored in ragged form as a flat
label array plus offsets. Files are written by cgfm/scripts/precompute_groups.py and read by cgfm.data.CGOMGDataset.

Entries are positional: entry i belongs to structure i of the split, in the order the StructureDataset iterates. A
partition silently paired with the wrong split would not crash, it would quietly train every coarse-to-fine arm on
meaningless groups and invalidate the comparison, so the file also stores the material identifier of every structure.
The identifiers are checked against the dataset on load, which catches a reordered, filtered or differently
preprocessed split rather than only a differently sized one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class GroupTable:
    """
    Partitions of every structure of one dataset split.

    :param labels:
        Structure-local group label of every atom of every structure, concatenated, of shape (total atoms,).
    :type labels: numpy.ndarray
    :param offsets:
        Start index of every structure within labels, of shape (number of structures + 1,).
    :type offsets: numpy.ndarray
    :param num_groups:
        Number of groups of every structure, of shape (number of structures,).
    :type num_groups: numpy.ndarray
    :param identifiers:
        Material identifier of every structure, of shape (number of structures,).
    :type identifiers: numpy.ndarray
    :param method:
        Name of the method that produced the partitions.
    :type method: str
    """

    labels: np.ndarray
    offsets: np.ndarray
    num_groups: np.ndarray
    identifiers: np.ndarray
    method: str

    def __len__(self) -> int:
        """
        Return the number of structures covered by this table.

        :return:
            The number of structures.
        :rtype: int
        """
        return len(self.num_groups)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """
        Return the labels and group count of one structure.

        :param index:
            Index of the structure within the split.
        :type index: int

        :return:
            Labels of shape (number of atoms,) and the number of groups.
        :rtype: tuple[numpy.ndarray, int]
        """
        return self.labels[self.offsets[index]:self.offsets[index + 1]], int(self.num_groups[index])

    def validate(self) -> None:
        """
        Check the internal consistency of the table.

        :raises ValueError:
            If the offsets are not monotonic, the identifiers do not cover every structure, or a structure's labels do
            not cover exactly its declared groups.
        """
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.labels):
            raise ValueError("The offsets must start at zero and end at the number of labels.")
        if np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Every structure must contribute at least one atom.")
        if len(self.identifiers) != len(self.num_groups):
            raise ValueError("There must be exactly one identifier per structure.")
        for index in range(len(self)):
            labels, num_groups = self[index]
            if labels.min() < 0 or labels.max() >= num_groups:
                raise ValueError(f"Structure {index} has labels outside [0, {num_groups}).")
            if len(np.unique(labels)) != num_groups:
                raise ValueError(f"Structure {index} declares {num_groups} groups but does not use all of them.")

    def save(self, path: Path) -> None:
        """
        Write the table to a compressed numpy archive.

        :param path:
            Destination path.
        :type path: Path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=self.labels, offsets=self.offsets, num_groups=self.num_groups,
                            identifiers=self.identifiers, method=np.array(self.method))

    @staticmethod
    def load(path: Path) -> "GroupTable":
        """
        Read a table from a compressed numpy archive.

        :param path:
            Path of the archive.
        :type path: Path

        :return:
            The table.
        :rtype: GroupTable

        :raises FileNotFoundError:
            If the archive does not exist.
        :raises ValueError:
            If the archive predates the identifier field and so cannot be checked for alignment.
        """
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No group file at {path}. Produce it with cgfm/scripts/precompute_groups.py.")
        with np.load(path, allow_pickle=False) as archive:
            if "identifiers" not in archive:
                raise ValueError(
                    f"The group file {path} carries no material identifiers, so it cannot be checked against the "
                    f"dataset it is paired with. Regenerate it with cgfm/scripts/precompute_groups.py.")
            return GroupTable(labels=archive["labels"].astype(np.int64), offsets=archive["offsets"].astype(np.int64),
                              num_groups=archive["num_groups"].astype(np.int64), identifiers=archive["identifiers"],
                              method=str(archive["method"]))

    def check_against(self, identifiers: Sequence[str], source: str) -> None:
        """
        Check that this table describes exactly the given structures, in the same order.

        :param identifiers:
            Material identifier of every structure of the dataset, in dataset order.
        :type identifiers: Sequence[str]
        :param source:
            Description of the dataset, used in the error message.
        :type source: str

        :raises ValueError:
            If the table covers a different number of structures, or any identifier disagrees.
        """
        if len(identifiers) != len(self):
            raise ValueError(
                f"The group file covers {len(self)} structures but {source} holds {len(identifiers)}. Group files are "
                f"positional, so they must be regenerated whenever the split or its preprocessing changes.")
        mismatched = np.flatnonzero(self.identifiers != np.asarray(identifiers, dtype=self.identifiers.dtype))
        if mismatched.size > 0:
            first = int(mismatched[0])
            raise ValueError(
                f"The group file disagrees with {source} at {mismatched.size} of {len(self)} structures, first at "
                f"index {first}: the file has '{self.identifiers[first]}' and the dataset has "
                f"'{identifiers[first]}'. The file was built from a different or differently ordered split.")

    @staticmethod
    def from_labels(labels_per_structure: Sequence[np.ndarray], identifiers: Sequence[str],
                    method: str) -> "GroupTable":
        """
        Build a table from the per-structure label arrays.

        :param labels_per_structure:
            Group labels of every structure, in dataset order.
        :type labels_per_structure: Sequence[numpy.ndarray]
        :param identifiers:
            Material identifier of every structure, in the same order.
        :type identifiers: Sequence[str]
        :param method:
            Name of the method that produced the partitions.
        :type method: str

        :return:
            The table.
        :rtype: GroupTable

        :raises ValueError:
            If the number of identifiers does not match the number of structures.
        """
        if len(identifiers) != len(labels_per_structure):
            raise ValueError("There must be exactly one identifier per structure.")
        sizes = np.array([len(labels) for labels in labels_per_structure], dtype=np.int64)
        return GroupTable(
            labels=np.concatenate(labels_per_structure).astype(np.int64) if len(labels_per_structure) > 0
            else np.zeros(0, dtype=np.int64),
            offsets=np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(sizes)]),
            num_groups=np.array([len(np.unique(labels)) for labels in labels_per_structure], dtype=np.int64),
            identifiers=np.asarray(identifiers, dtype=np.str_),
            method=method)
