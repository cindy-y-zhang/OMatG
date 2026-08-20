"""
Measure sampler and training throughput, and how it scales with concurrency.

The gate pipeline is dominated by 209-step sequential integration, which is a
long chain of small kernels rather than a few large ones.  Whether more
accelerators would help therefore depends on how much of one device a single
job actually occupies, which this benchmark answers by reporting per-process
throughput that can be compared across simultaneously launched copies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from omg.omg_cli import OMGCLI
from omg.omg_trainer import OMGTrainer

from ..data import GeometryDataModule
from ..lightning import JointGeometryLightning


DEFAULT_CONFIGS = (
    "joint_geometry/configs/mpts52.yaml",
    "joint_geometry/configs/J.yaml",
    "joint_geometry/configs/local.yaml",
)


def build_cli(configs: list[str], overrides: list[str]) -> OMGCLI:
    """Instantiate the joint model and datamodule without running a subcommand."""
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
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    configs = list(arguments.config or DEFAULT_CONFIGS)
    overrides = ["--data.batch_size", str(arguments.batch_size)]
    for override in arguments.overrides:
        name, _, value = override.partition("=")
        overrides.extend([f"--{name}", value])

    cli = build_cli(configs, overrides)
    model = cli.model
    datamodule = cli.datamodule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    datamodule.setup("validate")
    batch = next(iter(datamodule.val_dataloader())).to(device)
    x_0 = model.sampler.sample_p_0(batch).to(device)

    # One model call per interpolated field advances the whole batch one step,
    # so timing that call is timing the sampler's inner loop directly.
    time_step = torch.full((len(batch.n_atoms),), 0.5, device=device)
    with torch.no_grad():
        for _ in range(arguments.warmup):
            model.model(x_0, time_step)
        torch.cuda.synchronize() if device.type == "cuda" else None
        started = time.perf_counter()
        for _ in range(arguments.steps):
            model.model(x_0, time_step)
        torch.cuda.synchronize() if device.type == "cuda" else None
        forward_seconds = (time.perf_counter() - started) / arguments.steps

    # A forward and backward over the same graph, without the Lightning loop's
    # logging, which needs an attached Trainer and is not part of the cost.
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    model.train()

    def optimisation_step() -> None:
        result = model.model(x_0, time_step)
        loss = sum(value.square().mean() for value in result.to_dict().values())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(arguments.warmup):
        optimisation_step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    started = time.perf_counter()
    for _ in range(arguments.steps):
        optimisation_step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    train_seconds = (time.perf_counter() - started) / arguments.steps

    integration_steps = int(model.si._integration_time_steps) - 1
    fields = 3
    report = {
        "label": arguments.label,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "concurrent_peers": os.environ.get("JG_BENCH_PEERS"),
        "structures": int(len(batch.n_atoms)),
        "atoms": int(batch.n_atoms.sum()),
        "forward_seconds": forward_seconds,
        "forwards_per_second": 1.0 / forward_seconds,
        "training_step_seconds": train_seconds,
        "projected_sampling_seconds_per_batch": forward_seconds * integration_steps * fields,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
    }
    print(json.dumps(report, indent=2))
    if arguments.out:
        destination = Path(arguments.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
