"""
Sweep geometry-state trajectories through one trained joint-geometry checkpoint.

The oracle arm answers "does perfect local geometry help?" only if the leaked
state is both informative and delivered while the structural fields are still
being decided.  This probe holds the network, the compositions and the paired
structural priors fixed and varies only the geometry trajectory, so the failure
can be attributed to the descriptor, the injection or the delivery schedule.

Policies
--------
``zero``          geometry input is identically zero for the whole trajectory.
``prior_noise``   the sampled Gaussian prior, frozen; a pure-noise control.
``chord``         the deployed oracle: constant velocity ``g_1 - g_0``.
``feedback``      exact denoiser velocity ``(g_1 - g_t) / (1 - t)``.
``annealed``      chord velocity scaled like the position field's annealing.
``early``         reaches the clean endpoint by ``--early-time`` and holds.
``clean``         the clean endpoint at every step; the information ceiling.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data

from direct_geometry.scripts.evaluate import ANGLE_TOL, LTOL, STOL
from omg.analysis import ValidAtoms, match_rmsds
from omg.datamodule import OMGData
from omg.omg_cli import OMGCLI
from omg.omg_trainer import OMGTrainer

from ..data import GEOMETRY_FIELD
from ..data import GeometryDataModule, within_element_permutation
from ..interpolants import OracleGeometryInterpolants
from ..lightning import JointGeometryLightning
from .score_outcomes import target_errors


DEFAULT_CONFIGS = (
    "joint_geometry/configs/mpts52.yaml",
    # The deployable J overlay, not O: ``OracleGeometryInterpolants.integrate``
    # overwrites ``geometry_b`` after the model returns, which would silently
    # discard every policy below and integrate the chord instead.  J shares the
    # architecture and the prior draw order, so an O checkpoint loads unchanged.
    "joint_geometry/configs/J.yaml",
    "joint_geometry/configs/local.yaml",
)

DEFAULT_POLICIES = (
    "zero",
    "prior_noise",
    "chord",
    "chord#repeat",
    "feedback",
    "annealed",
    "early",
    "clean",
)
"""Policies to sweep; ``#`` suffixes label repeats that measure the noise floor."""

MINIMUM_REMAINING_TIME = 1.0e-3
"""Floor on ``1 - t`` in the feedback velocity."""

SHUFFLED_SUFFIX = "_shuffled"
"""
Policy-name suffix that permutes clean endpoints within element.

This keeps the trajectory and every marginal identical while destroying the
correspondence between an atom and its own environment, which is the control
that separates "the network reads the descriptor's content" from "the network
only depends on the channel's distribution".
"""


GeometryVelocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def geometry_policy(
    name: str,
    prior: torch.Tensor,
    clean: torch.Tensor,
    annealing: float,
    early_time: float,
) -> tuple[torch.Tensor, GeometryVelocity]:
    """Return the initial geometry state and its velocity rule for a policy."""
    chord = clean - prior
    if name == "zero":
        return torch.zeros_like(prior), lambda state, t: torch.zeros_like(state)
    if name == "prior_noise":
        return prior, lambda state, t: torch.zeros_like(state)
    if name == "clean":
        return clean, lambda state, t: torch.zeros_like(state)
    if name == "chord":
        return prior, lambda state, t: chord
    if name == "feedback":
        return prior, lambda state, t: (clean - state) / (1.0 - t).clamp(
            min=MINIMUM_REMAINING_TIME
        )
    if name == "annealed":
        # Matches the position field's (1 + k t) reparameterisation so that the
        # geometry and structural trajectories cover the same fraction of their
        # displacement at the same nominal time.
        scale = 1.0 + 0.5 * annealing
        return prior, lambda state, t: ((1.0 + annealing * t) / scale) * chord
    if name == "early":
        return prior, lambda state, t: torch.where(
            t < early_time, chord / early_time, torch.zeros_like(chord)
        )
    raise ValueError(f"Unknown geometry policy {name!r}.")


def check_interpolants(si: object) -> None:
    """
    Refuse to sweep policies through an interpolant that overrides them.

    ``OracleGeometryInterpolants.integrate`` reassigns ``geometry_b`` after the
    model returns, so a policy expressed as a model wrapper would be discarded
    and every arm would silently integrate the same chord.
    """
    if isinstance(si, OracleGeometryInterpolants):
        raise ValueError(
            "OracleGeometryInterpolants overwrites geometry_b after the model returns, so "
            "every geometry policy would silently integrate the chord instead. Use the "
            "deployable J overlay, which shares the architecture and the prior draw order."
        )


class GeometryPathModel(torch.nn.Module):
    """Wrap a joint model so the geometry velocity follows a fixed policy."""

    def __init__(self, model: torch.nn.Module, velocity: GeometryVelocity) -> None:
        super().__init__()
        self.model = model
        self.velocity = velocity

    def forward(self, state: Data, time: torch.Tensor) -> Data:
        result = self.model(state, time)
        scalar_time = time.reshape(-1)[0]
        geometry_velocity = self.velocity(getattr(state, GEOMETRY_FIELD), scalar_time)
        result.geometry_b = geometry_velocity
        result.geometry_eta = torch.zeros_like(geometry_velocity)
        return result


def as_atoms(structure: OMGData, index: int) -> Atoms:
    """Return one structure of a CPU batch as an ASE object."""
    lower, upper = structure.ptr[index], structure.ptr[index + 1]
    return Atoms(
        numbers=structure.species[lower:upper],
        scaled_positions=structure.pos[lower:upper],
        cell=structure.cell[index],
        pbc=(1, 1, 1),
    )


def seed_batch(seed: int, batch_index: int) -> None:
    """Reproduce ``PairedInferenceSeed`` so every policy shares its priors."""
    value = seed + 1_000_003 * int(batch_index)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def score(generated: list[Atoms], reference: list[Atoms], workers: int) -> dict:
    """Return the gate's match statistics and endpoint errors for one policy."""
    valid_generated = ValidAtoms.get_valid_atoms(
        generated, desc="generated", skip_validation=True, number_cpus=1
    )
    valid_reference = ValidAtoms.get_valid_atoms(
        reference, desc="reference", skip_validation=True, number_cpus=1
    )
    match_rate, mean_rmsd, _, _, rmsds, _, corrected_rmsd, _ = match_rmsds(
        valid_generated,
        valid_reference,
        ltol=LTOL,
        stol=STOL,
        angle_tol=ANGLE_TOL,
        number_cpus=workers,
        enable_progress_bar=False,
    )
    outcomes = []
    for index, rmsd in enumerate(rmsds):
        position_mse, cell_mse = target_errors(generated[index], reference[index])
        outcomes.append(
            {
                "index": index,
                "match": rmsd is not None,
                "rmsd": None if rmsd is None else float(rmsd),
                "position_target_mse": position_mse,
                "cell_target_mse": cell_mse,
            }
        )
    return {
        "num_compositions": len(outcomes),
        "match_rate": float(match_rate),
        "mean_rmsd": float(mean_rmsd) if np.isfinite(mean_rmsd) else None,
        "corrected_rmsd": float(corrected_rmsd),
        "position_target_mse": float(np.mean([e["position_target_mse"] for e in outcomes])),
        "cell_target_mse": float(np.mean([e["cell_target_mse"] for e in outcomes])),
        "outcomes": outcomes,
    }


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
    parser.add_argument(
        "--checkpoint", default="joint_geometry/runs/oracle/O_seed0/checkpoints/last.ckpt"
    )
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-time", type=float, default=0.3)
    parser.add_argument("--position-annealing", type=float, default=10.182659004291072)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="joint_geometry/reports/GEOMETRY-PATHS.json")
    arguments = parser.parse_args()

    configs = list(arguments.config or DEFAULT_CONFIGS)
    overrides = ["--data.batch_size", str(arguments.batch_size)]
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
    model.si._enable_progress_bar = False

    datamodule.setup("validate")
    loader = datamodule.val_dataloader()
    batches = []
    for index, batch in enumerate(loader):
        if index >= arguments.batches:
            break
        batches.append(batch)

    check_interpolants(model.si)

    results = {}
    for label in arguments.policies.split(","):
        policy = label.partition("#")[0]
        shuffled = policy.endswith(SHUFFLED_SUFFIX)
        if shuffled:
            policy = policy[: -len(SHUFFLED_SUFFIX)]
        generated_atoms: list[Atoms] = []
        reference_atoms: list[Atoms] = []
        for batch_index, batch in enumerate(batches):
            x_1 = batch.to(device)
            seed_batch(arguments.seed, batch_index)
            x_0 = model.sampler.sample_p_0(x_1).to(device)
            prior = getattr(x_0, GEOMETRY_FIELD).clone()
            clean = getattr(x_1, GEOMETRY_FIELD).clone()
            if shuffled:
                order = within_element_permutation(
                    x_1.species.detach().cpu().numpy(), arguments.seed + batch_index
                )
                clean = clean[torch.as_tensor(order, device=clean.device)]
            initial, velocity = geometry_policy(
                policy, prior, clean, arguments.position_annealing, arguments.early_time
            )
            setattr(x_0, GEOMETRY_FIELD, initial.clone())
            wrapped = GeometryPathModel(model.model, velocity)
            with torch.no_grad():
                generated = model.si.integrate(x_0, wrapped, save_intermediate=False)
            generated = generated.to("cpu")
            reference = x_1.clone().to("cpu")
            for index in range(len(batch.n_atoms)):
                generated_atoms.append(as_atoms(generated, index))
                reference_atoms.append(as_atoms(reference, index))
        results[label] = score(generated_atoms, reference_atoms, arguments.workers)
        print(
            f"{label:14s} match {100.0 * results[label]['match_rate']:6.2f}%  "
            f"pos_mse {results[label]['position_target_mse']:.5f}  "
            f"cell_mse {results[label]['cell_target_mse']:.4f}",
            flush=True,
        )

    report = {
        "checkpoint": arguments.checkpoint,
        "configs": configs,
        "overrides": arguments.overrides,
        "seed": arguments.seed,
        "batches": arguments.batches,
        "batch_size": arguments.batch_size,
        "early_time": arguments.early_time,
        "position_annealing": arguments.position_annealing,
        "tolerances": {"ltol": LTOL, "stol": STOL, "angle_tol": ANGLE_TOL},
        "policies": results,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
