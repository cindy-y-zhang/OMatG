"""Leakage-safe tables and datasets for per-structure motif censuses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import json

import numpy as np
import torch

from cgfm.blockdata import dataset_identifiers, file_sha256, settings_hash
from joint_geometry.data import atom_order_hashes
from omg.datamodule import OMGData, OMGDataModule, OMGDataset, StructureDataset

from .census import MotifCensusTransform


CENSUS_FIELD = "motif_census"
"""Per-structure conditioning field, constant in time and never diffused."""

CENSUS_TABLE_NAME = "census_{split}.npz"
"""Filename of one split's per-structure censuses."""

TRANSFORM_NAME = "prototypes.pkl"
"""Filename of the train-only motif prototypes and standardisations."""

MANIFEST_NAME = "manifest.json"
"""Filename that binds artifacts to settings and source datasets."""

SPLITS = ("train", "val", "test")


@dataclass
class CensusTable:
    """Per-structure motif censuses aligned to one structure split."""

    identifiers: np.ndarray
    atom_order_digests: np.ndarray
    values: np.ndarray
    fractions: np.ndarray
    settings_digest: str

    def __len__(self) -> int:
        return len(self.values)

    @property
    def dimension(self) -> int:
        return int(self.values.shape[1])

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            identifiers=self.identifiers,
            atom_order_digests=self.atom_order_digests,
            values=self.values.astype(np.float32),
            fractions=self.fractions.astype(np.float32),
            settings_digest=np.asarray(self.settings_digest),
        )

    @classmethod
    def load(cls, path: Path) -> "CensusTable":
        with np.load(path, allow_pickle=False) as archive:
            values = archive["values"].astype(np.float32)
            if values.ndim != 2:
                raise ValueError(f"Census values in {path} must be two-dimensional, got {values.shape}.")
            return cls(
                identifiers=archive["identifiers"],
                atom_order_digests=archive["atom_order_digests"],
                values=values,
                fractions=archive["fractions"].astype(np.float32),
                settings_digest=str(archive["settings_digest"]),
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
            raise ValueError(f"The census table paired with {source} uses different settings.")
        if len(self) != len(identifiers):
            raise ValueError(
                f"The census table covers {len(self)} structures but {source} holds {len(identifiers)}."
            )
        stored = np.asarray(self.identifiers)
        expected = np.asarray(identifiers, dtype=stored.dtype)
        mismatched = np.flatnonzero(stored != expected)
        if mismatched.size:
            first = int(mismatched[0])
            raise ValueError(
                f"The census table disagrees with {source} at structure {first}: "
                f"{self.identifiers[first]!r} != {identifiers[first]!r}."
            )
        flat = (
            np.concatenate([np.asarray(row, dtype=np.int64) for row in numbers])
            if numbers
            else np.zeros(0, dtype=np.int64)
        )
        offsets = np.concatenate([[0], np.cumsum([len(row) for row in numbers], dtype=np.int64)])
        if not np.array_equal(self.atom_order_digests, atom_order_hashes(flat, offsets)):
            raise ValueError(
                f"The census table paired with {source} was built from a different atom ordering."
            )
        if self.fractions.shape != self.values.shape:
            raise ValueError(f"The census table paired with {source} has inconsistent row counts.")
        if not np.isfinite(self.values).all():
            raise ValueError(f"The census table paired with {source} contains non-finite values.")


class CensusDataset(OMGDataset):
    """Standard atomwise OMatG graphs carrying one motif census per structure."""

    def __init__(
        self,
        dataset: StructureDataset,
        table: CensusTable,
        settings: dict[str, Any],
        shuffle_structures: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        super().__init__(dataset)
        identifiers = dataset_identifiers(dataset)
        numbers = [dataset[index].atomic_numbers.numpy() for index in range(len(dataset))]
        table.check_against(identifiers, numbers, settings, "the paired structure dataset")
        self._table = table
        self.settings = settings
        self._dtype = _dataset_floating_dtype(dataset)
        # The mismatched control pairs each structure with another structure's census, so
        # the conditioning channel keeps its exact marginal distribution and loses only its
        # correspondence to the crystal being generated.
        self._permutation = (
            derangement(len(table), shuffle_seed) if shuffle_structures else None
        )

    def get(self, idx: int) -> OMGData:
        data = super().get(idx)
        row = int(self._permutation[idx]) if self._permutation is not None else idx
        census = torch.from_numpy(np.ascontiguousarray(self._table.values[row])).to(self._dtype)
        setattr(data, CENSUS_FIELD, census.unsqueeze(0))
        return data


class CensusDataModule(OMGDataModule):
    """OMatG data module backed by precomputed per-structure motif censuses."""

    def __init__(
        self,
        train_dataset: StructureDataset,
        val_dataset: StructureDataset,
        pred_dataset: StructureDataset,
        census_dir: str,
        shuffle_structures: bool = False,
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
        root = Path(census_dir)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No motif-census manifest at {manifest_path}. "
                "Run motif_conditioning.scripts.build_census first."
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
            raise ValueError(f"The motif-census manifest at {manifest_path} is incomplete.")
        artifact_names = [TRANSFORM_NAME] + [
            CENSUS_TABLE_NAME.format(split=split) for split in SPLITS
        ]
        for name in artifact_names:
            path = root / name
            if not path.is_file() or file_sha256(path) != artifact_hashes.get(name):
                raise ValueError(f"The census artifact {path} is missing or differs from its manifest.")

        datasets = {"train": train_dataset, "val": val_dataset, "test": pred_dataset}
        for split, dataset in datasets.items():
            source_path = getattr(dataset, "_file_path", None)
            if source_path is None:
                raise ValueError(f"Cannot identify the source file for the {split} split.")
            if file_sha256(Path(source_path)) != source_hashes.get(split):
                raise ValueError(
                    f"The {split} dataset differs from the source used for the census artifacts."
                )

        self.transform = MotifCensusTransform.load(root / TRANSFORM_NAME)
        if self.transform.settings.as_dict() != settings:
            raise ValueError(f"The transform and manifest at {root} encode different settings.")
        self.census_dir = root
        self.settings = settings
        self.census_dimension = self.transform.dimension
        self.train_dataset = CensusDataset(
            train_dataset,
            CensusTable.load(root / CENSUS_TABLE_NAME.format(split="train")),
            settings,
            shuffle_structures=shuffle_structures,
            shuffle_seed=shuffle_seed,
        )
        self.val_dataset = CensusDataset(
            val_dataset,
            CensusTable.load(root / CENSUS_TABLE_NAME.format(split="val")),
            settings,
            shuffle_structures=shuffle_structures,
            shuffle_seed=shuffle_seed + 1,
        )
        self.pred_dataset = CensusDataset(
            pred_dataset,
            CensusTable.load(root / CENSUS_TABLE_NAME.format(split="test")),
            settings,
            shuffle_structures=shuffle_structures,
            shuffle_seed=shuffle_seed + 2,
        )


def derangement(count: int, seed: int) -> np.ndarray:
    """Return a deterministic permutation that leaves no structure paired with itself.

    Built by conjugating a single full cycle with a random permutation, which is a
    derangement by construction rather than by rejection or repair.
    """
    if count < 2:
        raise ValueError("A mismatched control needs at least two structures.")
    order = np.random.default_rng(seed).permutation(count)
    permutation = np.empty(count, dtype=np.int64)
    permutation[order] = np.roll(order, -1)
    return permutation


def _dataset_floating_dtype(dataset: StructureDataset) -> torch.dtype:
    dtype = getattr(dataset, "_torch_precision", None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError("The wrapped StructureDataset has no recorded floating-point precision.")
    return dtype
