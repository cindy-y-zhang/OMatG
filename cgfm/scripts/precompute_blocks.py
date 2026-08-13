"""
Precompute train-only rigid-block tables for every split of a dataset.

Templates and the CN validity mask are fitted on the training split, then train, validation and test are transformed
with identical centre, cap, orphan and fallback policies. The run is resumable: cached decompositions are reused, and a
finished split is not rewritten.

Usage:

    python -m cgfm.scripts.precompute_blocks --data-dir omg/data/mpts_52 --out-dir cgfm/blocks/mpts_52
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional
import json
import pickle
import numpy as np
from tqdm import tqdm
from omg.datamodule import StructureDataset
from ..blockdata import (TemplateLibrary, cn_validity_mask, default_settings, file_sha256, record_structure,
                         settings_hash, table_from_records)
from ..blocks import Decomposition, centre_elements, decompose, fit_templates
from ..readout import orphan_free


_DATASET: Optional[StructureDataset] = None
"""Per-worker dataset handle, since LMDB environments cannot be shared across processes."""

_TYPE_KEY_MODE: str = "centre-cn"
"""Per-worker block type mode."""


def _initialise_worker(file_path: str, type_key_mode: str) -> None:
    """
    Open the dataset inside a worker process.

    :param file_path:
        Path of the dataset split.
    :type file_path: str
    :param type_key_mode:
        How block types are defined.
    :type type_key_mode: str
    """
    global _DATASET, _TYPE_KEY_MODE
    _DATASET = StructureDataset(file_path=file_path, lazy_storage=True, niggli_reduce=False,
                                convert_to_fractional=True, floating_point_precision="64-true")
    _TYPE_KEY_MODE = type_key_mode


def _decompose_structure(index: int) -> tuple[Optional[Decomposition], str]:
    """
    Decompose one structure and repair it so its blocks match its centre atoms one for one.

    :param index:
        Index of the structure within the split.
    :type index: int

    :return:
        The repaired decomposition, or None if neighbour analysis failed, and the material identifier.
    :rtype: tuple[Optional[Decomposition], str]
    """
    assert _DATASET is not None
    structure = _DATASET[index]
    identifier = str(structure.metadata.get("identifier", index))
    pymatgen = structure.get_pymatgen_structure()
    decomposition = decompose(pymatgen, identifier=identifier, type_key_mode=_TYPE_KEY_MODE)
    if decomposition is None:
        coords = np.array(pymatgen.cart_coords, dtype=np.float64)
        numbers = np.array([site.specie.Z for site in pymatgen], dtype=np.int64)
        symbols = tuple(site.specie.symbol for site in pymatgen)
        centre_species, _ = centre_elements(symbols)
        from ..blocks import Block
        blocks = tuple(
            Block(centre=index_, ligands=np.empty(0, dtype=np.int64), offsets=np.empty((0, 3)),
                  species=(), type_key=(pymatgen[index_].specie.symbol, 0))
            for index_ in range(len(pymatgen)) if symbols[index_] in centre_species)
        decomposition = Decomposition(identifier=identifier, lattice=np.array(pymatgen.lattice.matrix, dtype=np.float64),
                                      coords=coords, numbers=numbers, blocks=blocks, num_singletons=len(blocks),
                                      centre_rule="crystalnn-failed")
        return orphan_free(decomposition, type_key_mode=_TYPE_KEY_MODE), identifier
    return orphan_free(decomposition, type_key_mode=_TYPE_KEY_MODE), identifier


def _collect_instances(decompositions: list[Decomposition],
                       fine: bool) -> dict[tuple, list[tuple[np.ndarray, tuple[str, ...]]]]:
    """
    Group every observed polyhedron by its type.

    :param decompositions:
        Decompositions of the training split.
    :type decompositions: list[Decomposition]
    :param fine:
        If True, types additionally carry the sorted vertex composition.
    :type fine: bool

    :return:
        Vertex offsets and vertex species of every block, grouped by type.
    :rtype: dict[tuple, list[tuple[numpy.ndarray, tuple[str, ...]]]]
    """
    instances = defaultdict(list)
    for decomposition in decompositions:
        for block in decomposition.blocks:
            key = (block.type_key[0], len(block.species))
            instances[key + (tuple(sorted(block.species)),) if fine else key].append((block.offsets, block.species))
    return instances


def _decompose_split(file_path: str, source_hash: str, workers: int, type_key_mode: str, cache_path: Path,
                     description: str) -> list[tuple[Decomposition, str]]:
    """
    Decompose a split, reusing a pickle cache when it matches.

    :param file_path:
        Path of the LMDB split.
    :type file_path: str
    :param source_hash:
        SHA-256 digest of the source file.
    :type source_hash: str
    :param workers:
        Number of worker processes.
    :type workers: int
    :param type_key_mode:
        How block types are defined.
    :type type_key_mode: str
    :param cache_path:
        Path of the decomposition cache.
    :type cache_path: Path
    :param description:
        Label of the progress bar.
    :type description: str

    :return:
        Repaired decompositions paired with identifiers, in dataset order.
    :rtype: list[tuple[Decomposition, str]]
    """
    dataset = StructureDataset(file_path=file_path, lazy_storage=True, niggli_reduce=False,
                               convert_to_fractional=True, floating_point_precision="64-true")
    count = len(dataset)
    del dataset
    if cache_path.exists():
        payload = pickle.loads(cache_path.read_bytes())
        if (payload.get("file_path") == file_path and payload.get("source_hash") == source_hash
                and payload.get("count") == count and payload.get("type_key") == type_key_mode):
            print(f"Loaded {count} decompositions from {cache_path}.")
            return payload["rows"]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise_worker,
                             initargs=(file_path, type_key_mode)) as executor:
        for decomposition, identifier in tqdm(executor.map(_decompose_structure, range(count), chunksize=16),
                                              total=count, desc=description, unit=" structures"):
            rows.append((decomposition, identifier))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps({
        "file_path": file_path, "source_hash": source_hash, "count": count,
        "type_key": type_key_mode, "rows": rows}))
    return rows


def main() -> None:
    """Fit train-only templates and write a block table for every split."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="omg/data/mpts_52", help="directory holding train.lmdb, val.lmdb, test.lmdb")
    parser.add_argument("--out-dir", default="cgfm/blocks/mpts_52", help="directory to write tables and the template library")
    parser.add_argument("--workers", type=int, default=8, help="number of worker processes")
    parser.add_argument("--type-key", choices=("centre-cn", "centre-cn-ligands"), default="centre-cn",
                        help="what a rigid template is shared across")
    arguments = parser.parse_args()

    data_dir = Path(arguments.data_dir)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = default_settings(arguments.type_key)
    digest = settings_hash(settings)
    cache_dir = out_dir / "cache"

    splits = ("train", "val", "test")
    lmdb_paths = {}
    source_hashes = {}
    for split in splits:
        lmdb_path = data_dir / f"{split}.lmdb"
        if not lmdb_path.exists():
            raise FileNotFoundError(f"No LMDB at {lmdb_path}.")
        lmdb_paths[split] = lmdb_path
        source_hashes[split] = file_sha256(lmdb_path)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("settings_digest") != digest or previous.get("source_hashes") != source_hashes:
            raise ValueError(
                f"{out_dir} contains block artifacts for different sources or settings; "
                "choose a new output directory or remove the stale artifacts explicitly.")
    else:
        stale = [out_dir / name for name in ("templates.pkl", "train.npz", "val.npz", "test.npz")
                 if (out_dir / name).exists()]
        if stale:
            names = ", ".join(path.name for path in stale)
            raise ValueError(
                f"{out_dir} has provenance-free block artifacts ({names}); "
                "choose a new output directory or remove the stale artifacts explicitly.")

    rows_by_split = {}
    for split in splits:
        rows_by_split[split] = _decompose_split(str(lmdb_paths[split]), source_hashes[split],
                                                arguments.workers, arguments.type_key,
                                                cache_dir / f"{split}.decompositions.pkl", f"Decomposing {split}")

    library_path = out_dir / "templates.pkl"
    if library_path.exists():
        library = TemplateLibrary.load(library_path)
        library.check_settings(settings)
        print(f"Loaded templates from {library_path}.")
    else:
        print("Fitting train-only templates.")
        training = [decomposition for decomposition, _ in rows_by_split["train"]]
        fine = fit_templates(_collect_instances(training, fine=True), species_aware=True)
        coarse = fit_templates(_collect_instances(training, fine=False), species_aware=False)
        library = TemplateLibrary(coarse=coarse, fine=fine, cn_valid=cn_validity_mask(coarse),
                                  settings=settings, settings_digest=digest)
        library.save(library_path)

    for split in splits:
        table_path = out_dir / f"{split}.npz"
        if table_path.exists():
            print(f"{table_path} already exists, leaving it alone.")
            continue
        print(f"Writing {table_path}.")
        identifiers = []
        records = []
        unseen = 0
        for decomposition, identifier in tqdm(rows_by_split[split], desc=f"Recording {split}", unit=" structures"):
            record = record_structure(decomposition, library, type_key_mode=arguments.type_key)
            identifiers.append(identifier)
            records.append(record)
            unseen += int(record["fallback_count"])
        table = table_from_records(identifiers, records, digest)
        table.save(table_path)
        print(f"  {split}: {len(table)} structures, {int(table.n_blocks.sum())} blocks, "
              f"{unseen} fallback placements.")

    manifest = {
        "settings": settings,
        "settings_digest": digest,
        "data_dir": str(data_dir),
        "splits": {split: str(data_dir / f"{split}.lmdb") for split in splits},
        "source_hashes": source_hashes,
        "n_coarse_types": len(library.coarse),
        "n_fine_types": len(library.fine),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}.")


if __name__ == "__main__":
    main()
