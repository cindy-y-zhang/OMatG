"""Decide whether a global motif census can possibly help before training anything.

Composition is given in crystal structure prediction, so a census only carries usable
signal to the extent that it is *not* predictable from composition. This script measures
that residual directly, and separately measures how much the census reveals about cell
volume beyond composition, since volume is the part of the answer a CSP model must guess.

Everything here is fitted on the training split and scored on validation, so a high score
cannot come from memorising the fit set.

Example:

    python -m motif_conditioning.scripts.precheck_census --prototypes 2,4,8,16,32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from joint_geometry.data import GEOMETRY_TABLE_NAME, GeometryTable
from motif_conditioning.census import MotifCensusSettings, MotifCensusTransform


RIDGE_PENALTY = 1.0e-3
"""Ridge penalty on standardised features, large enough to keep the solve conditioned."""


def composition_matrix(numbers: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-structure element fractions and the element ordering behind them."""
    numbers = np.asarray(numbers, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    elements = np.unique(numbers)
    lookup = {int(element): index for index, element in enumerate(elements)}
    counts = np.zeros((len(offsets) - 1, len(elements)), dtype=np.float64)
    for structure in range(len(offsets) - 1):
        rows = numbers[int(offsets[structure]) : int(offsets[structure + 1])]
        for atomic_number in rows:
            counts[structure, lookup[int(atomic_number)]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True), elements


def align_composition(
    numbers: np.ndarray, offsets: np.ndarray, elements: np.ndarray
) -> np.ndarray:
    """Build composition features for a split using the training element ordering."""
    numbers = np.asarray(numbers, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    lookup = {int(element): index for index, element in enumerate(elements)}
    counts = np.zeros((len(offsets) - 1, len(elements)), dtype=np.float64)
    for structure in range(len(offsets) - 1):
        rows = numbers[int(offsets[structure]) : int(offsets[structure + 1])]
        for atomic_number in rows:
            index = lookup.get(int(atomic_number))
            if index is not None:
                counts[structure, index] += 1.0
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0.0)


def ridge_r2(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    """Variance-weighted R^2 of a ridge fit, trained and scored on disjoint splits."""
    mean = train_features.mean(axis=0)
    scale = np.maximum(train_features.std(axis=0), 1.0e-8)
    train_scaled = (train_features - mean) / scale
    test_scaled = (test_features - mean) / scale

    design = np.hstack([train_scaled, np.ones((len(train_scaled), 1))])
    query = np.hstack([test_scaled, np.ones((len(test_scaled), 1))])
    penalty = RIDGE_PENALTY * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_targets)

    prediction = query @ coefficients
    residual = float(((test_targets - prediction) ** 2).sum())
    total = float(((test_targets - train_targets.mean(axis=0)) ** 2).sum())
    return 1.0 - residual / total if total > 0.0 else 0.0


def load_split(root: Path, split: str) -> GeometryTable:
    return GeometryTable.load(root / GEOMETRY_TABLE_NAME.format(split=split))


def log_volume_per_atom(data_dir: Path, split: str, expected: int) -> np.ndarray:
    """Cell volume per atom, the scale a CSP model must infer from composition alone."""
    from omg.datamodule import StructureDataset

    dataset = StructureDataset(
        file_path=str(data_dir / f"{split}.lmdb"), lazy_storage=False, niggli_reduce=False
    )
    if len(dataset) != expected:
        raise ValueError(
            f"The {split} structure dataset holds {len(dataset)} entries "
            f"but its census table covers {expected}."
        )
    volumes = np.empty(len(dataset), dtype=np.float64)
    for index in range(len(dataset)):
        structure = dataset[index]
        cell = structure.cell.detach().cpu().numpy().reshape(3, 3)
        volumes[index] = abs(float(np.linalg.det(cell))) / len(structure.atomic_numbers)
    return np.log(volumes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--descriptor-dir", default="joint_geometry/artifacts/mpts_52/radial17")
    parser.add_argument("--data-dir", default="omg/data/mpts_52")
    parser.add_argument("--prototypes", default="2,4,8,16,32")
    parser.add_argument("--fit-sites", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="motif_conditioning/reports/CENSUS-PRECHECK.json")
    arguments = parser.parse_args()

    root = Path(arguments.descriptor_dir)
    data_dir = Path(arguments.data_dir)
    train = load_split(root, "train")
    validation = load_split(root, "val")

    train_composition, elements = composition_matrix(train.numbers, train.atom_offsets)
    validation_composition = align_composition(
        validation.numbers, validation.atom_offsets, elements
    )
    train_volume = log_volume_per_atom(data_dir, "train", len(train))
    validation_volume = log_volume_per_atom(data_dir, "val", len(validation))

    volume_from_composition = ridge_r2(
        train_composition, train_volume, validation_composition, validation_volume
    )

    generator = np.random.default_rng(arguments.seed)
    sample = generator.choice(
        len(train.values), size=min(arguments.fit_sites, len(train.values)), replace=False
    )

    records = []
    for prototypes in (int(token) for token in arguments.prototypes.split(",")):
        settings = MotifCensusSettings(
            prototypes=prototypes, fit_sites=arguments.fit_sites, seed=arguments.seed
        )
        transform = MotifCensusTransform.fit(train.values[sample], settings)
        train_census = transform.fractions(train.values, train.atom_offsets)
        validation_census = transform.fractions(validation.values, validation.atom_offsets)
        transform.finalise(train_census)

        census_from_composition = ridge_r2(
            train_composition, train_census, validation_composition, validation_census
        )
        # If composition plus a single length scale already reproduces the census, the
        # motif framing is decoration and a scalar volume hint would do the same work.
        census_from_composition_and_volume = ridge_r2(
            np.hstack([train_composition, train_volume[:, None]]),
            train_census,
            np.hstack([validation_composition, validation_volume[:, None]]),
            validation_census,
        )
        volume_from_census = ridge_r2(
            np.hstack([train_composition, train_census]),
            train_volume,
            np.hstack([validation_composition, validation_census]),
            validation_volume,
        )
        occupancy = float((train_census.mean(axis=0) > 0.01).sum())
        records.append(
            {
                "prototypes": prototypes,
                "census_variance_explained_by_composition": census_from_composition,
                "census_information_beyond_composition": 1.0 - census_from_composition,
                "census_variance_explained_by_composition_and_volume": (
                    census_from_composition_and_volume
                ),
                "census_information_beyond_composition_and_volume": (
                    1.0 - census_from_composition_and_volume
                ),
                "log_volume_r2_composition_only": volume_from_composition,
                "log_volume_r2_composition_plus_census": volume_from_census,
                "log_volume_r2_gain_from_census": volume_from_census - volume_from_composition,
                "residual_log_volume_variance_removed": (
                    (volume_from_census - volume_from_composition) / (1.0 - volume_from_composition)
                    if volume_from_composition < 1.0
                    else 0.0
                ),
                "occupied_prototypes": occupancy,
                "assignment_temperature": transform.temperature,
            }
        )
        print(
            f"K={prototypes:3d}"
            f"  beyond composition {1.0 - census_from_composition:6.1%}"
            f"  beyond composition+volume {1.0 - census_from_composition_and_volume:6.1%}"
            f"   log-volume R2 {volume_from_composition:.3f} -> {volume_from_census:.3f}"
            f"   occupied {occupancy:.0f}/{prototypes}"
        )

    report = {
        "descriptor_dir": str(root),
        "n_structures": {"train": len(train), "val": len(validation)},
        "n_elements": int(len(elements)),
        "records": records,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {destination}.")


if __name__ == "__main__":
    main()
