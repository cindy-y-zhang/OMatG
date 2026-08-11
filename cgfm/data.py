"""
Data plumbing that attaches precomputed partitions to OMatG batches.

Every coarse-to-fine arm needs the per-structure group count, and the two fixed arms additionally need the group labels
themselves. Both are attached to each OMGData object as ordinary tensor attributes, which PyTorch Geometric then batches
without any custom collation: cg_group is a per-atom tensor that is concatenated, and cg_n_groups mirrors n_atoms as a
scalar per structure that is stacked into shape (B,).

Group labels are local to their structure and deliberately not offset during batching, because the assignment matrix is
built with the structure index as the scatter index and a dense group axis. That keeps the group axis at the size of the
largest structure in the batch rather than the total number of groups in the batch.
"""

from pathlib import Path
from typing import Any, List, Optional
import torch
from omg.datamodule import OMGData, OMGDataModule, OMGDataset, StructureDataset
from .groupfile import GroupTable
from .grouping import GROUP_FIELD, NUM_GROUPS_FIELD, REFERENCE_METHODS, reference_field


def reference_paths(group_file: Path) -> dict[str, Path]:
    """
    Locate the fixed partitions of the same split that sit beside the given group file.

    Group files are named <split>.<method>.npz and cgfm/scripts/precompute_groups.py always writes both methods of a
    split together, so the siblings of a group file can be derived from its name rather than configured separately.
    Deriving them keeps the arm overlays down to the single path that actually selects the arm.

    :param group_file:
        Path of the group file that defines the path of this arm.
    :type group_file: Path

    :return:
        Mapping from method name to the path of its group file.
    :rtype: dict[str, Path]
    """
    split = group_file.name.split(".")[0]
    return {method: group_file.parent / f"{split}.{method}.npz" for method in REFERENCE_METHODS}


def dataset_identifiers(dataset: StructureDataset) -> List[str]:
    """
    Read the material identifier of every structure of a split, in dataset order.

    :param dataset:
        The split to read.
    :type dataset: StructureDataset

    :return:
        Identifier of every structure, falling back to the positional index where a structure carries none.
    :rtype: List[str]
    """
    return [str(dataset[index].metadata.get("identifier", index)) for index in range(len(dataset))]


class CGOMGDataset(OMGDataset):
    """
    OMGDataset that attaches a precomputed partition to every structure.

    :param dataset:
        StructureDataset holding the structures.
    :type dataset: StructureDataset
    :param group_file:
        Path of the group file produced by cgfm/scripts/precompute_groups.py for this split.
    :type group_file: str

    :raises ValueError:
        If the group file does not cover exactly the structures of the dataset, in the same order.
    """

    def __init__(self, dataset: StructureDataset, group_file: str) -> None:
        """Construct the dataset."""
        super().__init__(dataset)
        identifiers = dataset_identifiers(dataset)
        self._group_table = GroupTable.load(Path(group_file))
        self._group_table.check_against(identifiers, f"the dataset {group_file} was paired with")
        self._references = {}
        for method, path in reference_paths(Path(group_file)).items():
            table = GroupTable.load(path)
            table.check_against(identifiers, f"the dataset {group_file} was paired with")
            self._references[method] = table

    def get(self, idx: int) -> OMGData:
        """
        Return the structure at the given index with its partition attached.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: OMGData

        :raises ValueError:
            If the partition does not have one label per atom of the structure.
        """
        data = super().get(idx)
        labels, num_groups = self._group_table[idx]
        if len(labels) != int(data.n_atoms):
            raise ValueError(
                f"The group file gives {len(labels)} labels for structure {idx}, which has {int(data.n_atoms)} atoms. "
                f"The group file was most likely built from a different split or a differently preprocessed copy.")
        setattr(data, GROUP_FIELD, torch.from_numpy(labels).long())
        # A zero-dimensional tensor, like n_atoms, so that batching stacks it into shape (batch size,).
        setattr(data, NUM_GROUPS_FIELD, torch.tensor(num_groups, dtype=torch.long))
        for method, table in self._references.items():
            setattr(data, reference_field(method), torch.from_numpy(table[idx][0]).long())
        return data

    def get_group_table(self) -> GroupTable:
        """
        Return the partitions attached by this dataset.

        :return:
            The group table.
        :rtype: GroupTable
        """
        return self._group_table


class CGDataModule(OMGDataModule):
    """
    OMGDataModule whose training and validation datasets carry precomputed partitions.

    All coarse-to-fine arms use this data module, including the learned one, which needs the per-structure group count
    from the same file so that every arm compresses each structure by the same factor. Having the learned arm also see
    the reference labels lets the diagnostics report how far the learned partition drifts from them, which is the check
    that separates a genuine result from collapse onto geometric clustering.

    The prediction dataset is left untouched: sampling never evaluates a grouping.

    :param train_dataset:
        StructureDataset for training.
    :type train_dataset: StructureDataset
    :param val_dataset:
        StructureDataset for validation.
    :type val_dataset: StructureDataset
    :param pred_dataset:
        StructureDataset for prediction.
    :type pred_dataset: StructureDataset
    :param train_group_file:
        Path of the group file for the training split.
    :type train_group_file: str
    :param val_group_file:
        Path of the group file for the validation split.
    :type val_group_file: str
    :param kwargs:
        Additional keyword arguments forwarded to OMGDataModule.
    :type kwargs: Any
    """

    def __init__(self, train_dataset: StructureDataset, val_dataset: StructureDataset,
                 pred_dataset: StructureDataset, train_group_file: str, val_group_file: str,
                 train_batch_size: Optional[int] = None, val_batch_size: Optional[int] = None,
                 pred_batch_size: Optional[int] = None, **kwargs: Any) -> None:
        """Construct the data module."""
        super().__init__(train_dataset=train_dataset, val_dataset=val_dataset, pred_dataset=pred_dataset,
                         train_batch_size=train_batch_size, val_batch_size=val_batch_size,
                         pred_batch_size=pred_batch_size, **kwargs)
        # LightningDataset reads these attributes when it builds a dataloader, so replacing them here is enough.
        self.train_dataset = CGOMGDataset(train_dataset, train_group_file)
        self.val_dataset = CGOMGDataset(val_dataset, val_group_file)
