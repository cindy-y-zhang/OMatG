"""Build leakage-safe per-structure motif-census artifacts.

Prototypes are fitted on a sample of training sites only, then every structure in every
split is summarised as a distribution over those prototypes.

Example:

    python -m motif_conditioning.scripts.build_census \
        --prototypes 16 \
        --out-dir motif_conditioning/artifacts/mpts_52/motif16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from cgfm.blockdata import dataset_identifiers, file_sha256, settings_hash
from joint_geometry.data import atom_order_hashes
from motif_conditioning.census import MotifCensusSettings, MotifCensusTransform, raw_site_descriptors
from motif_conditioning.data import (
    CENSUS_TABLE_NAME,
    MANIFEST_NAME,
    SPLITS,
    TRANSFORM_NAME,
    CensusTable,
)
from omg.datamodule import OMGDataset, StructureDataset


def load_split(path: Path) -> tuple[StructureDataset, OMGDataset]:
    """Return the structure dataset and its graph wrapper, both needed downstream."""
    structures = StructureDataset(file_path=str(path), lazy_storage=True, niggli_reduce=False)
    return structures, OMGDataset(structures)


def raw_for_split(
    dataset: OMGDataset,
    settings: MotifCensusSettings,
    device: torch.device,
    dtype: torch.dtype,
    batch_structures: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-site descriptors for a whole split, with atom offsets and atomic numbers."""
    loader = DataLoader(dataset, batch_size=batch_structures, shuffle=False)
    blocks: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    numbers: list[np.ndarray] = []
    for batch in tqdm(loader, desc="descriptors", leave=False):
        batch = batch.to(device)
        with torch.no_grad():
            raw = raw_site_descriptors(
                batch.pos.to(dtype), batch.cell.to(dtype), batch.n_atoms, settings
            )
        blocks.append(raw.detach().cpu().double().numpy())
        counts.append(batch.n_atoms.detach().cpu().numpy().astype(np.int64))
        numbers.append(batch.species.detach().cpu().numpy().astype(np.int64))
    values = np.concatenate(blocks, axis=0)
    atom_counts = np.concatenate(counts)
    offsets = np.concatenate([[0], np.cumsum(atom_counts, dtype=np.int64)])
    return values, offsets, np.concatenate(numbers)


def load_or_compute_raw(
    datasets: dict[str, OMGDataset],
    settings: MotifCensusSettings,
    device: torch.device,
    dtype: torch.dtype,
    batch_structures: int,
    cache: Path | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-site descriptors for every split, reused across a sweep when a cache is given."""
    key = settings_hash(
        {"cutoff": settings.cutoff, "shells": settings.shells, "version": settings.version}
    )
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as archive:
            if str(archive["descriptor_digest"]) == key:
                print(f"Reusing cached per-site descriptors from {cache}.")
                return {
                    split: (
                        archive[f"{split}_values"],
                        archive[f"{split}_offsets"],
                        archive[f"{split}_numbers"],
                    )
                    for split in SPLITS
                }
        print(f"Ignoring {cache}: it was built for different descriptor settings.")

    raw = {
        split: raw_for_split(dataset, settings, device, dtype, batch_structures)
        for split, dataset in datasets.items()
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            descriptor_digest=np.asarray(key),
            **{
                f"{split}_{name}": array
                for split, arrays in raw.items()
                for name, array in zip(("values", "offsets", "numbers"), arrays)
            },
        )
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="omg/data/mpts_52")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prototypes", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--shells", type=int, default=16)
    parser.add_argument("--temperature-scale", type=float, default=1.0)
    parser.add_argument("--fit-sites", type=int, default=100_000)
    parser.add_argument("--batch-structures", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # The per-site descriptors do not depend on the vocabulary size, so a sweep over
    # prototypes should pay for them once rather than once per arm.
    parser.add_argument("--raw-cache", default=None)
    arguments = parser.parse_args()

    settings = MotifCensusSettings(
        cutoff=arguments.cutoff,
        shells=arguments.shells,
        prototypes=arguments.prototypes,
        temperature_scale=arguments.temperature_scale,
        fit_sites=arguments.fit_sites,
        seed=arguments.seed,
    )
    data_dir = Path(arguments.data_dir)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(arguments.device)
    dtype = torch.float64

    sources = {split: data_dir / f"{split}.lmdb" for split in SPLITS}
    loaded = {split: load_split(path) for split, path in sources.items()}
    structures = {split: pair[0] for split, pair in loaded.items()}
    datasets = {split: pair[1] for split, pair in loaded.items()}
    raw = load_or_compute_raw(
        datasets,
        settings,
        device,
        dtype,
        arguments.batch_structures,
        Path(arguments.raw_cache) if arguments.raw_cache else None,
    )

    train_values, train_offsets, _ = raw["train"]
    generator = np.random.default_rng(arguments.seed)
    sample = generator.choice(
        len(train_values), size=min(settings.fit_sites, len(train_values)), replace=False
    )
    transform = MotifCensusTransform.fit(train_values[sample], settings)
    transform.finalise(transform.fractions(train_values, train_offsets))
    transform.save(out_dir / TRANSFORM_NAME)

    counts = {}
    for split in SPLITS:
        values, offsets, numbers = raw[split]
        fractions = transform.fractions(values, offsets)
        table = CensusTable(
            identifiers=np.asarray(dataset_identifiers(structures[split])),
            atom_order_digests=atom_order_hashes(numbers, offsets),
            values=transform.standardise(fractions),
            fractions=fractions.astype(np.float32),
            settings_digest=settings_hash(settings.as_dict()),
        )
        table.save(out_dir / CENSUS_TABLE_NAME.format(split=split))
        counts[split] = {"structures": len(table), "atoms": int(offsets[-1])}

    artifact_names = [TRANSFORM_NAME] + [
        CENSUS_TABLE_NAME.format(split=split) for split in SPLITS
    ]
    manifest = {
        "settings": settings.as_dict(),
        "data_dir": str(data_dir),
        "dimension": transform.dimension,
        "assignment_temperature": transform.temperature,
        "counts": counts,
        "source_hashes": {split: file_sha256(path) for split, path in sources.items()},
        "artifact_sha256": {name: file_sha256(out_dir / name) for name in artifact_names},
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {transform.dimension}-channel census artifacts to {out_dir}.")


if __name__ == "__main__":
    main()
