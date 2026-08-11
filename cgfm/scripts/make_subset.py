"""
Write the first N structures of a dataset split to a new LMDB file.

Group files are positional and must cover a whole split, so smoke tests need a genuinely small split rather than a
truncated view of a large one. This produces one.

Usage:

    python -m cgfm.scripts.make_subset --data omg/data/mpts_52/train.lmdb --out cgfm/smoke_data/train.lmdb --count 200
"""

import argparse
from pathlib import Path
import pickle
import lmdb
from tqdm import tqdm
from omg.datamodule import StructureDataset


def main() -> None:
    """Write a prefix of a dataset split to a new LMDB file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path of the source dataset split")
    parser.add_argument("--out", required=True, help="path of the LMDB file to write")
    parser.add_argument("--count", type=int, required=True, help="number of structures to copy")
    arguments = parser.parse_args()

    out_path = Path(arguments.out)
    if out_path.exists():
        print(f"{out_path} already exists, leaving it alone.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = StructureDataset(file_path=arguments.data, lazy_storage=True, niggli_reduce=False,
                               convert_to_fractional=False, floating_point_precision="64-true")
    count = min(arguments.count, len(dataset))
    with (lmdb.Environment(str(out_path), subdir=False, map_size=int(1e11), lock=False) as env,
          env.begin(write=True) as transaction):
        for index in tqdm(range(count), desc=f"Writing {out_path}", unit=" structures"):
            transaction.put(str(index).encode(), pickle.dumps(dataset[index].to_dictionary()))
    print(f"Wrote {count} structures to {out_path}.")


if __name__ == "__main__":
    main()
