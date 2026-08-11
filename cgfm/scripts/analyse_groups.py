"""
Describe the partition a trained grouping network produces.

The learned arm is only interesting if what it learned can be named. This runs the trained grouping network over a split
and reports what its groups look like: how many, how large, how far across, and how they compare with the two fixed
partitions. The comparison with periodic k-medoids is the collapse check, since geometric clustering is what minimises
the within-group residual at a fixed group count; the comparison with the CrystalNN coordination shells is the actual
research question.

Only the grouping network is loaded from the checkpoint, not the denoiser, because nothing here needs to generate.

Usage:

    python -m cgfm.scripts.analyse_groups \\
        --checkpoint $(find runs/learned/seed0 -name 'best_match_rate*.ckpt' | head -1) \\
        --data omg/data/mpts_52/val.lmdb \\
        --groups cgfm/groups/mpts_52 --split val
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from omg.datamodule import StructureDataset
from ..data import CGOMGDataset
from ..diagnostics import adjusted_rand_index, group_statistics
from ..grouper import AnchorMembershipGrouper
from ..grouping import reference_field


GROUPER_PREFIX = "cg_grouper."
"""Prefix under which the grouping network's parameters are stored in a checkpoint."""


def load_grouper(checkpoint_path: Path, temperature: float) -> AnchorMembershipGrouper:
    """
    Rebuild the trained grouping network from a checkpoint.

    The network's shape is inferred from the stored tensors rather than from a configuration file, so the analysis
    cannot be run against a checkpoint it does not actually match.

    :param checkpoint_path:
        Path of the Lightning checkpoint.
    :type checkpoint_path: Path
    :param temperature:
        Membership temperature to evaluate at.
    :type temperature: float

    :return:
        The grouping network, in evaluation mode.
    :rtype: AnchorMembershipGrouper

    :raises ValueError:
        If the checkpoint does not hold a grouping network.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key[len(GROUPER_PREFIX):]: value for key, value in checkpoint["state_dict"].items()
             if key.startswith(GROUPER_PREFIX)}
    if not state:
        raise ValueError(f"{checkpoint_path} holds no grouping network, so it is not a run of the learned arm.")

    grouper = AnchorMembershipGrouper(
        hidden_dim=state["species_embedding.weight"].shape[1],
        num_layers=sum(1 for key in state if key.startswith("node_networks.") and key.endswith(".0.weight")),
        num_basis=state["edge_networks.0.0.weight"].shape[1] - 2 * state["species_embedding.weight"].shape[1],
        temperature=temperature)
    grouper.load_state_dict(state)
    grouper.eval()
    return grouper


def species_patterns(species: torch.Tensor, labels: torch.Tensor, offset: torch.Tensor,
                     top: int = 15) -> list[tuple[str, int]]:
    """
    Count the most common chemical compositions of the learned groups.

    :param species:
        Atomic number of every atom in the batch.
    :type species: torch.Tensor
    :param labels:
        Structure-local group label of every atom.
    :type labels: torch.Tensor
    :param offset:
        Structure index of every atom.
    :type offset: torch.Tensor
    :param top:
        Number of patterns to report.
        Defaults to 15.
    :type top: int

    :return:
        The most common compositions and their counts.
    :rtype: list[tuple[str, int]]
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for atom in range(len(species)):
        groups.setdefault((int(offset[atom]), int(labels[atom])), []).append(int(species[atom]))
    counter = Counter("-".join(str(number) for number in sorted(members)) for members in groups.values())
    return counter.most_common(top)


def main() -> None:
    """Describe the partition of a trained grouping network and compare it with the fixed partitions."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint of a learned-arm run")
    parser.add_argument("--data", required=True, help="dataset split to analyse")
    parser.add_argument("--groups", required=True, help="directory holding the group files of the split")
    parser.add_argument("--split", required=True, help="split name, used to find the group files")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="membership temperature, matching the end of the annealing schedule")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for the analysis pass")
    parser.add_argument("--json", default=None, help="write the summary to this JSON file")
    arguments = parser.parse_args()

    grouper = load_grouper(Path(arguments.checkpoint), arguments.temperature)
    dataset = CGOMGDataset(
        StructureDataset(arguments.data, lazy_storage=True, niggli_reduce=False, floating_point_precision="32-true"),
        str(Path(arguments.groups) / f"{arguments.split}.kmedoids.npz"))

    totals: dict[str, float] = {}
    weight = 0
    patterns: Counter = Counter()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False):
            num_structures = len(batch.n_atoms)
            assignment = grouper.assignment(batch)
            labels = assignment.argmax(dim=1)

            # group_statistics already reports the adjusted Rand index against both fixed partitions, which the batch
            # carries. Comparing the two fixed partitions with each other as well calibrates the scale: it says how far
            # apart two reasonable partitions of the same structures already are, without which a learned value is
            # hard to read as high or low.
            statistics = group_statistics(assignment, batch.batch, num_structures, batch)
            statistics["reference_ari_kmedoids_vs_shells"] = float(adjusted_rand_index(
                getattr(batch, reference_field("kmedoids")).long(),
                getattr(batch, reference_field("shells")).long(), batch.batch, num_structures))

            for name, value in statistics.items():
                totals[name] = totals.get(name, 0.0) + value * num_structures
            weight += num_structures
            patterns.update(dict(species_patterns(batch.species, labels, batch.batch, top=64)))

    summary = {name: value / weight for name, value in totals.items()}
    summary["structures"] = weight
    print(f"Learned partition over {weight} structures of {arguments.split} at temperature {arguments.temperature}:")
    for name in sorted(summary):
        print(f"  {name}: {summary[name]:.4f}")
    print("\nMost common group compositions, as sorted atomic numbers:")
    for pattern, count in patterns.most_common(15):
        print(f"  {pattern}: {count}")

    if arguments.json is not None:
        Path(arguments.json).write_text(json.dumps(
            {"summary": summary, "patterns": patterns.most_common(64)}, indent=2))
        print(f"\nWrote {arguments.json}.")


if __name__ == "__main__":
    main()
