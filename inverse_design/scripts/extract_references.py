"""
Pull the handful of real thiophosphates out of MPTS-52 and write them as CIF fixtures.

The polyhedra census is the instrument the whole search depends on, so it is checked
against structures whose motifs are known from the literature rather than against
synthetic geometry: Li7P3S11's P2S7 dimers, Li4GeS4's isolated tetrahedra, Li2SnS3's
edge-sharing octahedra, and LiAlP2S6, whose phosphorus is bonded to another phosphorus
and must therefore not be counted as a tetrahedron.

Run from the repository root::

    PYTHONPATH=. .venv/bin/python inverse_design/scripts/extract_references.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pymatgen.core import Composition

from omg.datamodule.structure_dataset import StructureDataset

WANTED = {
    "Li7P3S11": "corner-sharing P2S7 dimers alongside isolated PS4",
    "Li4GeS4": "isolated GeS4 tetrahedra",
    "Li2SnS3": "edge-sharing SnS6 octahedra, not tetrahedra at all",
    "Li4MnGe2S7": "a Ge2S7 corner-sharing pair",
    "Li6PS5I": "argyrodite, isolated PS4",
    "LiAl(PS3)2": "P2S6 units whose phosphorus carries a P-P bond",
}
"""Reduced formula to the motif the literature reports, which the census must reproduce."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="omg/data/mpts_52",
                        help="Directory holding the MPTS-52 LMDB splits.")
    parser.add_argument("--out", default="inverse_design/tests/data",
                        help="Directory to write the CIF fixtures into.")
    arguments = parser.parse_args()

    destination = Path(arguments.out)
    destination.mkdir(parents=True, exist_ok=True)

    remaining = dict(WANTED)
    for split in ("train", "val", "test"):
        if not remaining:
            break
        dataset = StructureDataset(f"{arguments.data_dir}/{split}.lmdb", lazy_storage=True)
        for index in range(len(dataset)):
            structure = dataset[index].get_pymatgen_structure()
            formula = Composition(structure.composition).reduced_formula
            if formula not in remaining:
                continue
            path = destination / f"{formula.replace('(', '').replace(')', '')}.cif"
            structure.to(filename=str(path))
            print(f"{formula:14s} {len(structure):3d} atoms  from {split}  -> {path}"
                  f"   [{remaining.pop(formula)}]")

    if remaining:
        raise SystemExit(f"Not found in MPTS-52: {sorted(remaining)}")


if __name__ == "__main__":
    main()
