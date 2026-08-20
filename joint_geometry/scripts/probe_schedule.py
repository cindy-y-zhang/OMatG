"""
Measure when each interpolated field actually converges during inference.

The oracle arm can only raise the match rate if its leaked geometry state is
informative *while the structural fields are still moving*.  Velocity annealing
reparameterises the structural trajectory but leaves the geometry trajectory on
the nominal linear schedule, so the two can converge at very different times.
This probe integrates one paired batch, records the remaining distance to the
final value for every field at every step, and reports the times at which each
field has covered fixed fractions of its total displacement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from omg.globals import BIG_TIME, SMALL_TIME
from omg.omg_cli import OMGCLI
from omg.omg_trainer import OMGTrainer

from ..data import GeometryDataModule
from ..lightning import JointGeometryLightning


CONVERGENCE_FRACTIONS = (0.5, 0.9, 0.95, 0.99)
"""Fractions of total displacement whose crossing times are reported."""

DEFAULT_CONFIGS = (
    "joint_geometry/configs/mpts52.yaml",
    "joint_geometry/configs/O.yaml",
    "joint_geometry/configs/local.yaml",
)
"""Config chain that reproduces the oracle arm without its training callbacks."""


def wrapped_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the minimum-image difference between fractional coordinates."""
    difference = left - right
    return difference - difference.round()


def remaining_distance(state: Data, final: Data) -> dict[str, float]:
    """Return the root-mean-square distance from each field to its final value."""
    return {
        "pos": float(wrapped_difference(state.pos, final.pos).square().mean().sqrt()),
        "cell": float((state.cell - final.cell).square().mean().sqrt()),
        "geometry": float((state.geometry - final.geometry).square().mean().sqrt()),
    }


def crossing_times(times: list[float], progress: list[float]) -> dict[str, float | None]:
    """Return the first time at which each convergence fraction is reached."""
    crossings: dict[str, float | None] = {}
    for fraction in CONVERGENCE_FRACTIONS:
        crossings[f"t_at_{int(100 * fraction):02d}pc"] = next(
            (time for time, value in zip(times, progress) if value >= fraction),
            None,
        )
    return crossings


def build_cli(configs: list[str], overrides: list[str]) -> OMGCLI:
    """Instantiate the oracle model and datamodule without running a subcommand."""
    arguments: list[str] = []
    for config in configs:
        arguments.extend(["--config", config])
    arguments.extend(overrides)
    return OMGCLI(
        model_class=JointGeometryLightning,
        datamodule_class=GeometryDataModule,
        trainer_class=OMGTrainer,
        save_config_callback=None,
        run=False,
        args=arguments,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="joint_geometry/runs/oracle/O_seed0/checkpoints/last.ckpt",
        help="Trained oracle-arm checkpoint whose inference schedule is probed.",
    )
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--structures", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="joint_geometry/reports/SCHEDULE-PROBE.json")
    arguments = parser.parse_args()

    configs = list(arguments.config or DEFAULT_CONFIGS)
    # One dataloader batch is one probe sample, so the batch size sets the size.
    overrides: list[str] = ["--data.batch_size", str(arguments.structures)]
    for override in arguments.overrides:
        name, _, value = override.partition("=")
        overrides.extend([f"--{name}", value])

    cli = build_cli(configs, overrides)
    model = cli.model
    datamodule = cli.datamodule

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.on_load_checkpoint(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    datamodule.setup("validate")
    batch = next(iter(datamodule.val_dataloader())).to(device)

    torch.manual_seed(arguments.seed)
    x_0 = model.sampler.sample_p_0(batch).to(device)
    with torch.no_grad():
        final, intermediate = model.si.integrate(x_0, model.model, save_intermediate=True)

    times = torch.linspace(SMALL_TIME, BIG_TIME, model.si._integration_time_steps).tolist()
    distances = [remaining_distance(state, final) for state in intermediate]
    clean_geometry_distance = [
        float((state.geometry - batch.geometry).square().mean().sqrt())
        for state in intermediate
    ]

    trajectory = {}
    for field in ("pos", "cell", "geometry"):
        initial = distances[0][field]
        progress = [
            1.0 - (entry[field] / initial) if initial > 0.0 else 1.0 for entry in distances
        ]
        trajectory[field] = {
            "initial_distance": initial,
            "progress": progress,
            **crossing_times(times, progress),
        }

    clean_initial = clean_geometry_distance[0]
    geometry_to_clean = [
        1.0 - (value / clean_initial) if clean_initial > 0.0 else 1.0
        for value in clean_geometry_distance
    ]
    trajectory["geometry_to_clean_endpoint"] = {
        "initial_distance": clean_initial,
        "progress": geometry_to_clean,
        **crossing_times(times, geometry_to_clean),
    }

    report = {
        "checkpoint": arguments.checkpoint,
        "configs": configs,
        "overrides": arguments.overrides,
        "structures": int(len(batch.n_atoms)),
        "integration_time_steps": int(model.si._integration_time_steps),
        "times": times,
        "trajectory": trajectory,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))

    print(f"{'field':32s} " + "  ".join(f"{int(100 * f):>3d}%" for f in CONVERGENCE_FRACTIONS))
    for field, entry in trajectory.items():
        cells = "  ".join(
            (
                f"{entry[f't_at_{int(100 * f):02d}pc']:.3f}"
                if entry[f"t_at_{int(100 * f):02d}pc"] is not None
                else "  -  "
            )
            for f in CONVERGENCE_FRACTIONS
        )
        print(f"{field:32s} {cells}")
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
