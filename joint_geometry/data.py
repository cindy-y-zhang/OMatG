"""Leakage-safe tables and datasets for per-site joint geometry endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import hashlib
import json

import numpy as np
import torch

from cgfm.blockdata import dataset_identifiers, file_sha256, settings_hash
from omg.datamodule import OMGData, OMGDataModule, OMGDataset, StructureDataset

from .descriptor import DescriptorTransform


GEOMETRY_FIELD = "geometry"
"""Field jointly interpolated with positions and cells."""

GEOMETRY_TABLE_NAME = "geometry_{split}.npz"
"""Filename of one split's transformed per-site endpoints."""

TRANSFORM_NAME = "transform.pkl"
"""Filename of the train-only descriptor transform."""

MANIFEST_NAME = "manifest.json"
"""Filename that binds artifacts to settings and source datasets."""


@dataclass
class GeometryTable:
    """Transformed geometry endpoints aligned to one structure split."""

    identifiers: np.ndarray
    atom_offsets: np.ndarray
    numbers: np.ndarray
    values: np.ndarray
    settings_digest: str
    atom_order_hashes: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.atom_offsets) - 1

    @property
    def dimension(self) -> int:
        return int(self.values.shape[1])

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            identifiers=self.identifiers,
            atom_offsets=self.atom_offsets.astype(np.int64),
            numbers=self.numbers.astype(np.int64),
            values=self.values.astype(np.float32),
            settings_digest=np.asarray(self.settings_digest),
            atom_order_hashes=(
                self.atom_order_hashes
                if self.atom_order_hashes is not None
                else atom_order_hashes(self.numbers, self.atom_offsets)
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "GeometryTable":
        with np.load(path, allow_pickle=False) as archive:
            values = archive["values"].astype(np.float32)
            if values.ndim != 2:
                raise ValueError(f"Geometry values in {path} must be two-dimensional, got {values.shape}.")
            return cls(
                identifiers=archive["identifiers"],
                atom_offsets=archive["atom_offsets"].astype(np.int64),
                numbers=archive["numbers"].astype(np.int64),
                values=values,
                settings_digest=str(archive["settings_digest"]),
                atom_order_hashes=(
                    archive["atom_order_hashes"]
                    if "atom_order_hashes" in archive.files
                    else atom_order_hashes(
                        archive["numbers"].astype(np.int64),
                        archive["atom_offsets"].astype(np.int64),
                    )
                ),
            )

    def check_against(
        self,
        identifiers: Sequence[str],
        numbers: Sequence[np.ndarray],
        settings: dict[str, Any],
        source: str,
    ) -> None:
        """Reject stale, reordered, truncated, or differently configured artifacts."""
        expected_digest = settings_hash(settings)
        if self.settings_digest != expected_digest:
            raise ValueError(f"The geometry table paired with {source} uses different descriptor settings.")
        if len(self) != len(identifiers):
            raise ValueError(
                f"The geometry table covers {len(self)} structures but {source} holds {len(identifiers)}."
            )
        stored_identifiers = np.asarray(self.identifiers)
        expected_identifiers = np.asarray(identifiers, dtype=stored_identifiers.dtype)
        mismatched = np.flatnonzero(stored_identifiers != expected_identifiers)
        if mismatched.size:
            first = int(mismatched[0])
            raise ValueError(
                f"The geometry table disagrees with {source} at structure {first}: "
                f"{self.identifiers[first]!r} != {identifiers[first]!r}."
            )
        expected_numbers = (
            np.concatenate([np.asarray(row, dtype=np.int64) for row in numbers])
            if numbers
            else np.zeros(0, dtype=np.int64)
        )
        if expected_numbers.shape != self.numbers.shape or not np.array_equal(
            expected_numbers, self.numbers
        ):
            raise ValueError(
                f"The geometry table paired with {source} was built from a different atom ordering."
            )
        expected_hashes = atom_order_hashes(
            expected_numbers,
            np.concatenate(
                [[0], np.cumsum([len(row) for row in numbers], dtype=np.int64)]
            ),
        )
        if self.atom_order_hashes is None or not np.array_equal(
            self.atom_order_hashes, expected_hashes
        ):
            raise ValueError(
                f"The geometry table paired with {source} has stale atom-order hashes."
            )
        if self.atom_offsets.shape != (len(self) + 1,):
            raise ValueError(f"The geometry table paired with {source} has invalid atom offsets.")
        if int(self.atom_offsets[-1]) != len(self.numbers) or len(self.values) != len(self.numbers):
            raise ValueError(f"The geometry table paired with {source} has inconsistent row counts.")
        if not np.isfinite(self.values).all():
            raise ValueError(f"The geometry table paired with {source} contains non-finite values.")


class GeometryDataset(OMGDataset):
    """Standard atomwise OMatG graphs carrying a clean per-site geometry field."""

    def __init__(
        self,
        dataset: StructureDataset,
        table: GeometryTable,
        settings: dict[str, Any],
        shuffle_within_element: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        super().__init__(dataset)
        identifiers = dataset_identifiers(dataset)
        numbers = [dataset[index].atomic_numbers.numpy() for index in range(len(dataset))]
        table.check_against(identifiers, numbers, settings, "the paired structure dataset")
        self._table = table
        self.settings = settings
        self._dtype = _dataset_floating_dtype(dataset)
        self._permutation = (
            within_element_permutation(table.numbers, shuffle_seed)
            if shuffle_within_element
            else None
        )

    def get(self, idx: int) -> OMGData:
        data = super().get(idx)
        start = int(self._table.atom_offsets[idx])
        stop = int(self._table.atom_offsets[idx + 1])
        rows = np.arange(start, stop)
        if self._permutation is not None:
            rows = self._permutation[rows]
        geometry = torch.from_numpy(np.ascontiguousarray(self._table.values[rows])).to(self._dtype)
        setattr(data, GEOMETRY_FIELD, geometry)
        return data


class GeometryDataModule(OMGDataModule):
    """OMatG data module backed by precomputed joint-geometry endpoint tables."""

    def __init__(
        self,
        train_dataset: StructureDataset,
        val_dataset: StructureDataset,
        pred_dataset: StructureDataset,
        geometry_dir: str,
        shuffle_within_element: bool = False,
        shuffle_seed: int = 0,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        pred_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            pred_dataset=pred_dataset,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            pred_batch_size=pred_batch_size,
            **kwargs,
        )
        root = Path(geometry_dir)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No joint-geometry manifest at {manifest_path}. "
                "Run joint_geometry.scripts.build_descriptor first."
            )
        manifest = json.loads(manifest_path.read_text())
        settings = manifest.get("settings")
        source_hashes = manifest.get("source_hashes")
        artifact_hashes = manifest.get("artifact_sha256")
        if (
            not isinstance(settings, dict)
            or not isinstance(source_hashes, dict)
            or not isinstance(artifact_hashes, dict)
        ):
            raise ValueError(f"The joint-geometry manifest at {manifest_path} is incomplete.")
        artifact_names = [TRANSFORM_NAME] + [
            GEOMETRY_TABLE_NAME.format(split=split) for split in ("train", "val", "test")
        ]
        for name in artifact_names:
            path = root / name
            if not path.is_file() or file_sha256(path) != artifact_hashes.get(name):
                raise ValueError(f"The geometry artifact {path} is missing or differs from its manifest.")

        datasets = {"train": train_dataset, "val": val_dataset, "test": pred_dataset}
        for split, dataset in datasets.items():
            source_path = getattr(dataset, "_file_path", None)
            if source_path is None:
                raise ValueError(f"Cannot identify the source file for the {split} split.")
            actual = file_sha256(Path(source_path))
            if actual != source_hashes.get(split):
                raise ValueError(
                    f"The {split} dataset differs from the source used for the geometry artifacts."
                )

        transform_path = root / TRANSFORM_NAME
        self.transform = DescriptorTransform.load(transform_path)
        if self.transform.settings.as_dict() != settings:
            raise ValueError(f"The transform and manifest at {root} encode different settings.")
        self.geometry_dir = root
        self.settings = settings
        self.geometry_dimension = self.transform.dimension
        self.train_dataset = GeometryDataset(
            train_dataset,
            GeometryTable.load(root / GEOMETRY_TABLE_NAME.format(split="train")),
            settings,
            shuffle_within_element=shuffle_within_element,
            shuffle_seed=shuffle_seed,
        )
        self.val_dataset = GeometryDataset(
            val_dataset,
            GeometryTable.load(root / GEOMETRY_TABLE_NAME.format(split="val")),
            settings,
            shuffle_within_element=shuffle_within_element,
            shuffle_seed=shuffle_seed + 1,
        )
        self.pred_dataset = GeometryDataset(
            pred_dataset,
            GeometryTable.load(root / GEOMETRY_TABLE_NAME.format(split="test")),
            settings,
            shuffle_within_element=shuffle_within_element,
            shuffle_seed=shuffle_seed + 2,
        )


def within_element_permutation(numbers: np.ndarray, seed: int) -> np.ndarray:
    """Return a deterministic permutation that preserves every element marginal."""
    numbers = np.asarray(numbers, dtype=np.int64)
    generator = np.random.default_rng(seed)
    permutation = np.arange(len(numbers))
    for number in np.unique(numbers):
        rows = np.flatnonzero(numbers == number)
        permutation[rows] = generator.permutation(rows)
    return permutation


def atom_order_hashes(numbers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Hash each structure's ordered atomic-number sequence."""
    numbers = np.asarray(numbers, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    return np.asarray(
        [
            hashlib.sha256(numbers[int(offsets[index]) : int(offsets[index + 1])].tobytes()).hexdigest()
            for index in range(len(offsets) - 1)
        ],
        dtype="<U64",
    )


def _dataset_floating_dtype(dataset: StructureDataset) -> torch.dtype:
    dtype = getattr(dataset, "_torch_precision", None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError("The wrapped StructureDataset has no recorded floating-point precision.")
    return dtype
