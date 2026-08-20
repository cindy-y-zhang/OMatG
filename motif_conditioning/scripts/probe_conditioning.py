"""Measure whether a trained census arm can influence its own prediction.

A conditioning arm that scores like its baseline has two very different explanations: the
conditioning carries nothing useful, or the conditioning never reached a magnitude where it
could act. Those demand opposite responses, and the aggregate match rate cannot tell them
apart. This probe separates them by measuring the conditioning residual against the node
features it is added to, and by measuring how much the prediction moves when the census is
swapped for another structure's or removed altogether.

Example:

    python -m motif_conditioning.scripts.probe_conditioning \
        --checkpoint motif_conditioning/runs/M_k32_seed0/checkpoints/last.ckpt \
        --census-dir motif_conditioning/artifacts/mpts_52/motif32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from motif_conditioning.data import CENSUS_FIELD, CENSUS_TABLE_NAME, CensusTable
from motif_conditioning.encoder import MotifConditionedCSPNet
from omg.datamodule import OMGDataset, StructureDataset


ENCODER_PREFIX = "model.encoder."


def load_encoder(checkpoint: Path, prototypes: int) -> MotifConditionedCSPNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        name[len(ENCODER_PREFIX) :]: value
        for name, value in payload["state_dict"].items()
        if name.startswith(ENCODER_PREFIX)
    }
    if not state:
        raise ValueError(f"No encoder parameters in {checkpoint}.")
    encoder = MotifConditionedCSPNet(
        census_dimension=prototypes, feature_mode="none", message_graph="fc"
    )
    encoder.load_state_dict(state)
    return encoder.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--census-dir", required=True)
    parser.add_argument("--data", default="omg/data/mpts_52/val.lmdb")
    parser.add_argument("--structures", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="motif_conditioning/reports/CONDITIONING-PROBE.json")
    arguments = parser.parse_args()

    census_dir = Path(arguments.census_dir)
    table = CensusTable.load(census_dir / CENSUS_TABLE_NAME.format(split="val"))
    count = arguments.structures
    if len(table) < 2 * count:
        raise ValueError("The split is too small to supply a mismatched census.")
    encoder = load_encoder(Path(arguments.checkpoint), table.dimension)

    dataset = OMGDataset(
        StructureDataset(file_path=arguments.data, lazy_storage=True, niggli_reduce=False)
    )
    batch = next(iter(DataLoader(dataset, batch_size=count, shuffle=False)))
    batch.pos = batch.pos.float()
    batch.cell = batch.cell.float()
    real = torch.from_numpy(table.values[:count]).float()
    mismatched = torch.from_numpy(table.values[count : 2 * count]).float()
    torch.manual_seed(arguments.seed)
    time = torch.rand((count, 256))

    incoming: list[torch.Tensor] = []
    residual: list[torch.Tensor] = []
    for adapter in encoder.adapters:
        adapter.register_forward_pre_hook(lambda _m, args: incoming.append(args[0].detach()))
        adapter.mixin.register_forward_hook(lambda _m, _a, out: residual.append(out.detach()))

    with torch.no_grad():
        setattr(batch, CENSUS_FIELD, real)
        conditioned = encoder(batch, time)
        setattr(batch, CENSUS_FIELD, mismatched)
        swapped = encoder(batch, time)
        encoder.guidance_scale = 0.0
        setattr(batch, CENSUS_FIELD, real)
        removed = encoder(batch, time)
        encoder.guidance_scale = 1.0

    layers = [
        {
            "layer": index,
            "node_feature_norm": float(node.norm(dim=1).mean()),
            "residual_spread": float(res.std(dim=0).norm()),
            "conditioning_share_of_node_features": float(res.std(dim=0).norm() / node.norm(dim=1).mean()),
        }
        # Only the first pass' hooks describe the conditional branch.
        for index, (node, res) in enumerate(zip(incoming[: encoder.num_layers], residual))
    ]

    def relative(first: torch.Tensor, second: torch.Tensor) -> float:
        return float((first - second).norm() / first.norm())

    sensitivity = {
        field: {
            "swapping_the_census": relative(conditioned[field], swapped[field]),
            "removing_the_census": relative(conditioned[field], removed[field]),
        }
        for field in ("pos_b", "cell_b")
    }
    mixin_norms = [float(adapter.mixin.weight.detach().norm()) for adapter in encoder.adapters]

    report = {
        "checkpoint": arguments.checkpoint,
        "census_dir": str(census_dir),
        "prototypes": table.dimension,
        "structures": count,
        "mixin_weight_norms": mixin_norms,
        "pathway_left_its_zero_initialisation": min(mixin_norms) > 1.0e-3,
        "layers": layers,
        "largest_conditioning_share": max(
            layer["conditioning_share_of_node_features"] for layer in layers
        ),
        "prediction_sensitivity": sensitivity,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")

    print(f"mixin norms {min(mixin_norms):.3f} to {max(mixin_norms):.3f} (zero at initialisation)")
    print(f"{'layer':6s} {'node features':>14s} {'conditioning':>13s} {'share':>9s}")
    for layer in layers:
        print(
            f"{layer['layer']:<6d} {layer['node_feature_norm']:14.1f}"
            f" {layer['residual_spread']:13.4f}"
            f" {layer['conditioning_share_of_node_features']:8.3%}"
        )
    for field, values in sensitivity.items():
        print(
            f"{field}: swapping the census moves it {values['swapping_the_census']:.2e}, "
            f"removing it moves it {values['removing_the_census']:.2e}"
        )
    print(f"Wrote {destination}.")


if __name__ == "__main__":
    main()
