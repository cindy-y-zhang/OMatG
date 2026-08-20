"""
Build leakage-safe CN-RDF endpoint artifacts for MPTS-52.

Example:

    python -m joint_geometry.scripts.build_descriptor \
        --representation cn-rdf4 \
        --out-dir joint_geometry/artifacts/mpts_52/cn-rdf4
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cgfm.blockdata import file_sha256, settings_hash
from direct_geometry.batches import collate, load_split
from joint_geometry.data import (
    GEOMETRY_TABLE_NAME,
    MANIFEST_NAME,
    TRANSFORM_NAME,
    GeometryTable,
)
from joint_geometry.descriptor import (
    DEFAULT_REPRESENTATION,
    DescriptorTransform,
    JointGeometryDescriptorSettings,
    clean_radial_descriptor,
)


SPLITS = ("train", "val", "test")


def raw_for_batch(
    batch,
    settings: JointGeometryDescriptorSettings,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Compute a raw radial descriptor and return it on CPU."""
    batch = batch.to(device)
    positions = batch.pos.to(dtype)
    cells = batch.cell.to(dtype)
    with torch.no_grad():
        raw = clean_radial_descriptor(positions, cells, batch.n_atoms, settings)
    return raw.detach().cpu().double().numpy()


def fit_transform(
    train_path: Path,
    settings: JointGeometryDescriptorSettings,
    sample_atoms: int,
    batch_structures: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    limit: int | None = None,
) -> DescriptorTransform:
    """Fit the train-only transform on a random structure order."""
    dataset = load_split(str(train_path))
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    order = torch.randperm(
        count,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    rows: list[np.ndarray] = []
    total = 0
    progress = tqdm(total=sample_atoms, desc="Fitting descriptor transform", unit=" atoms")
    for start in range(0, len(order), batch_structures):
        chosen = order[start : start + batch_structures]
        raw = raw_for_batch(collate(dataset, chosen), settings, device, dtype)
        take = min(len(raw), sample_atoms - total)
        rows.append(raw[:take])
        total += take
        progress.update(take)
        if total >= sample_atoms:
            break
    progress.close()
    if total < settings.raw_dimension:
        raise ValueError(
            f"Only collected {total} training atoms; need at least {settings.raw_dimension}."
        )
    return DescriptorTransform.fit(np.concatenate(rows), settings, seed=seed)


def build_table(
    source: Path,
    transform: DescriptorTransform,
    settings_digest: str,
    batch_structures: int,
    device: torch.device,
    dtype: torch.dtype,
    limit: int | None = None,
) -> GeometryTable:
    """Build one transformed split table in stored structure order."""
    dataset = load_split(str(source))
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    identifiers: list[str] = []
    numbers: list[np.ndarray] = []
    values: list[np.ndarray] = []
    offsets = [0]
    for start in tqdm(
        range(0, count, batch_structures),
        desc=f"Describing {source.stem}",
        unit=" batches",
    ):
        stop = min(start + batch_structures, count)
        indices = list(range(start, stop))
        batch = collate(dataset, indices)
        raw = raw_for_batch(batch, transform.settings, device, dtype)
        values.append(transform.transform(raw))
        for index in indices:
            structure = dataset._dataset[index]
            atomic_numbers = structure.atomic_numbers.detach().cpu().numpy().astype(np.int64)
            identifiers.append(str(structure.metadata.get("identifier", index)))
            numbers.append(atomic_numbers)
            offsets.append(offsets[-1] + len(atomic_numbers))
    return GeometryTable(
        identifiers=np.asarray(identifiers),
        atom_offsets=np.asarray(offsets, dtype=np.int64),
        numbers=np.concatenate(numbers) if numbers else np.zeros(0, dtype=np.int64),
        values=np.concatenate(values) if values else np.zeros((0, transform.dimension), dtype=np.float32),
        settings_digest=settings_digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="omg/data/mpts_52")
    parser.add_argument(
        "--out-dir",
        default=f"joint_geometry/artifacts/mpts_52/{DEFAULT_REPRESENTATION}",
    )
    parser.add_argument(
        "--representation",
        choices=("cn-rdf4", "cn-rdf8", "radial17"),
        default=DEFAULT_REPRESENTATION,
    )
    parser.add_argument("--sample-atoms", type=int, default=100_000)
    parser.add_argument("--batch-structures", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("32", "64"), default="32")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug-only number of leading structures per split",
    )
    arguments = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    data_dir = Path(arguments.data_dir)
    output = Path(arguments.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = JointGeometryDescriptorSettings(representation=arguments.representation)
    settings_dict = settings.as_dict()
    digest = settings_hash(settings_dict)
    sources = {split: data_dir / f"{split}.lmdb" for split in SPLITS}
    for split, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"No {split} split at {source}.")
    source_hashes = {split: file_sha256(path) for split, path in sources.items()}

    manifest_path = output / MANIFEST_NAME
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("settings_digest") != digest
            or previous.get("source_hashes") != source_hashes
            or previous.get("debug_limit") != arguments.limit
        ):
            raise ValueError(
                f"{output} already contains artifacts for different settings or source data."
            )

    device = torch.device(arguments.device)
    dtype = torch.float32 if arguments.precision == "32" else torch.float64
    transform_path = output / TRANSFORM_NAME
    if transform_path.exists():
        transform = DescriptorTransform.load(transform_path)
        if transform.settings != settings:
            raise ValueError(f"The existing transform at {transform_path} has different settings.")
        print(f"Using existing transform {transform_path}.")
    else:
        transform = fit_transform(
            sources["train"],
            settings,
            arguments.sample_atoms,
            arguments.batch_structures,
            arguments.seed,
            device,
            dtype,
            limit=arguments.limit,
        )
        transform.save(transform_path)
        print(
            f"Wrote {transform_path}: {transform.dimension} dimensions, "
            f"{transform.explained_variance_ratio:.4f} shell-profile variance retained."
        )

    tables: dict[str, GeometryTable] = {}
    for split, source in sources.items():
        destination = output / GEOMETRY_TABLE_NAME.format(split=split)
        if destination.exists():
            tables[split] = GeometryTable.load(destination)
            expected = (
                len(load_split(str(source)))
                if arguments.limit is None
                else min(arguments.limit, len(load_split(str(source))))
            )
            if len(tables[split]) != expected:
                raise ValueError(
                    f"The existing table {destination} covers {len(tables[split])} structures; "
                    f"this invocation expects {expected}."
                )
            print(f"Using existing table {destination}.")
        else:
            tables[split] = build_table(
                source,
                transform,
                digest,
                arguments.batch_structures,
                device,
                dtype,
                limit=arguments.limit,
            )
            tables[split].save(destination)
            print(
                f"Wrote {destination}: {len(tables[split])} structures, "
                f"{len(tables[split].numbers)} atoms."
            )

    manifest = {
        "settings": settings_dict,
        "settings_digest": digest,
        "data_dir": str(data_dir),
        "splits": {split: str(path) for split, path in sources.items()},
        "source_hashes": source_hashes,
        "n_structures": {split: len(table) for split, table in tables.items()},
        "n_atoms": {split: int(len(table.numbers)) for split, table in tables.items()},
        "dimension": transform.dimension,
        "explained_variance_ratio": transform.explained_variance_ratio,
        "fit_sample_atoms": arguments.sample_atoms,
        "fit_seed": arguments.seed,
        "debug_limit": arguments.limit,
        "artifact_sha256": {
            TRANSFORM_NAME: file_sha256(transform_path),
            **{
                GEOMETRY_TABLE_NAME.format(split=split): file_sha256(
                    output / GEOMETRY_TABLE_NAME.format(split=split)
                )
                for split in SPLITS
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}.")


if __name__ == "__main__":
    main()
