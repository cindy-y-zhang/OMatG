"""
The actual denoising error of a trained arm, at fixed denoising times.

WHY THE TRAINING LOSS WILL NOT DO

OMatG's ODE objective for a velocity field is

    L(b) = mean(b^2) - 2 mean(b . v)                                                        (1)

for the target velocity v, which is the mean squared error shifted by a constant:

    L(b) = mean((b - v)^2) - mean(v^2).                                                     (2)

The shift is independent of the model, so at one fixed time the loss does rank two arms. But it depends on the time and on
the data, it is usually negative, and its magnitude is dominated by mean(v^2), which at the noisy end is enormous. A "5 per
cent better loss" is therefore not 5 per cent better denoising, and losses at different times cannot be pooled. Gate DG3 is
stated on the mean squared error, so the mean squared error is what is measured here.

HOW THE TARGET IS OBTAINED

Not by reimplementing it. The target velocity involves the periodic interpolant's wrapping convention and a per-structure
centre-of-mass projection, and a second implementation of those would be a second thing to be wrong. Instead the model's
prediction is handed to OMatG's own loss as a leaf tensor and the loss is differentiated with respect to it. From (1),

    dL/db = (2/N) (b - v)      so      v = b - (N/2) dL/db,                                 (3)

with N the number of elements the mean runs over. This recovers whatever target OMatG actually regresses against --
including the centre-of-mass correction, and including the antithetic branch, where it correctly returns the average of the
two -- and it stays correct if that code changes.

The identity (2) is then checked directly: the reported error must equal the loss plus mean(v^2) to numerical tolerance, and
a zero prediction must score exactly mean(v^2), which is the denominator every relative number here is quoted against. A
relative error of 1.0 is the do-nothing predictor and 0.0 is exact.

Usage:

    python -m direct_geometry.scripts.target_mse --run-dir direct_geometry/runs/paired/A_seed0
    python -m direct_geometry.scripts.target_mse --run-dir <dir> --checkpoint <path> --out mse.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import torch
from torch_geometric.data import Data

from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer
from omg.datamodule import OMGDataModule


DEFAULT_TIMES = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
"""
The fixed time buckets the error is reported in.

Spread over the path rather than concentrated at either end. A descriptor of the *current* state is worth most where that
state resembles a crystal, so a gain that appears only near t=1 is still a real gain but a different claim from one that
holds throughout; separating them is the point of bucketing at all.
"""

FIELDS = ("pos", "cell")
"""
The two fields with a velocity to predict.

Species are given in this task and carried by an identity interpolant with weight zero, so there is no species error to
report.
"""

TOLERANCE = 5.0e-4
"""
Relative tolerance for the identity error = loss + mean(target^2).

Loose enough for float32 accumulation over a large batch, tight enough that a wrong target reconstruction cannot pass.
"""


def arm_of(run_dir: Path) -> dict[str, str]:
    """
    Recover which arm a run was, from the command it recorded.

    Read from the run rather than passed in. Every arm has identical parameter shapes -- that is deliberate, so that a
    contrast is not confounded with model size -- which means loading arm D's weights under arm A's configuration would
    succeed silently and produce a meaningless number. The recorded command is the only thing that distinguishes them.

    :param run_dir:
        Directory of the run, containing the ``COMMAND`` file its launcher wrote.
    :type run_dir: pathlib.Path

    :return:
        The ``--config`` paths in order, and the encoder's two factor settings.
    :rtype: dict[str, str]

    :raises SystemExit:
        If the command file is missing, or does not pin both factors.
    """
    command_file = run_dir / "COMMAND"
    if not command_file.is_file():
        raise SystemExit(f"{command_file} does not exist, so the arm this checkpoint belongs to is unknown. Every arm "
                         f"here has the same parameter shapes, so guessing would load fine and mean nothing.")
    tokens = command_file.read_text().split()
    configs, settings = [], {}
    prefix = "--model.model.init_args.encoder.init_args."
    for index, token in enumerate(tokens[:-1]):
        if token == "--config":
            configs.append(tokens[index + 1])
        elif token.startswith(prefix):
            settings[token[len(prefix):]] = tokens[index + 1]
    missing = {"message_graph", "feature_mode"} - settings.keys()
    if missing or not configs:
        raise SystemExit(f"{command_file} does not pin {', '.join(sorted(missing)) or 'any config'}.")
    return {"configs": configs, **settings}


def newest_checkpoint(run_dir: Path) -> Path:
    """
    Find the checkpoint to evaluate: the one the run selected on match rate.

    :param run_dir:
        Directory of the run.
    :type run_dir: pathlib.Path

    :return:
        Path to the checkpoint.
    :rtype: pathlib.Path

    :raises SystemExit:
        If no checkpoint is present.
    """
    for name in ("checkpoints/best_match_rate.ckpt", "checkpoints/last.ckpt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    found = sorted(run_dir.glob("**/*.ckpt"))
    if not found:
        raise SystemExit(f"no checkpoint under {run_dir}")
    return found[-1]


def build(arm: dict[str, str], data_dir: str, batch_size: int, device: torch.device
          ) -> tuple[OMGLightning, OMGDataModule]:
    """
    Instantiate the arm's module and data module through OMatG's own CLI.

    Through the CLI rather than by hand because the parser links arguments that matter here -- the batch size into the base
    sampler, the trainer precision into every dataset -- and a hand-built module would differ from the trained one in
    exactly those places.

    :param arm:
        Output of :func:`arm_of`.
    :type arm: dict[str, str]
    :param data_dir:
        Directory holding the split LMDBs.
    :type data_dir: str
    :param batch_size:
        Structures per batch.
    :type batch_size: int
    :param device:
        Device to move the module to.
    :type device: torch.device

    :return:
        The module and its data module.
    :rtype: tuple[omg.omg_lightning.OMGLightning, omg.datamodule.OMGDataModule]
    """
    arguments = ["validate"]
    for config in arm["configs"]:
        arguments += ["--config", config]
    prefix = "--model.model.init_args.encoder.init_args."
    for key in ("message_graph", "feature_mode"):
        arguments += [prefix + key, arm[key]]
    for split, field in (("train", "train_dataset"), ("val", "val_dataset"), ("test", "pred_dataset")):
        arguments += [f"--data.{field}.init_args.file_path", f"{data_dir}/{split}.lmdb"]
    arguments += ["--data.batch_size", str(batch_size), "--data.num_workers", "0",
                  "--data.persistent_workers", "false", "--data.prefetch_factor", "null",
                  "--trainer.logger", "null", "--trainer.callbacks", "null"]
    cli = OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, trainer_class=OMGTrainer,
                 save_config_callback=None, run=False, args=arguments)
    return cli.model.to(device).eval(), cli.datamodule


def errors_for_batch(module: OMGLightning, x_1: Data, time_value: float, seed: int) -> dict[str, torch.Tensor]:
    """
    Sum of squared errors and of squared targets for one batch at one fixed time.

    Sums rather than means, so that batches of different sizes can be pooled afterwards without weighting a small last
    batch as heavily as a full one.

    :param module:
        The trained module.
    :type module: omg.omg_lightning.OMGLightning
    :param x_1:
        A batch of reference structures, already on the module's device.
    :type x_1: torch_geometric.data.Data
    :param time_value:
        The denoising time to evaluate at, in ``[0, 1]``.
    :type time_value: float
    :param seed:
        Seed of the base-distribution draw. Shared across arms so that every arm denoises the same corruptions.
    :type seed: int

    :return:
        Per field, the squared-error sum, the squared-target sum, the element count, and the loss OMatG would report.
    :rtype: dict[str, torch.Tensor]
    """
    device = x_1.pos.device
    # Forked so that the draw depends on the batch and the time and nothing else -- not on how many batches came before,
    # which would make a partial evaluation incomparable with a complete one.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        x_0 = module.sampler.sample_p_0(x_1).to(device)
    t = torch.full((len(x_1.n_atoms),), float(time_value), device=device)

    captured: dict[str, torch.Tensor] = {}

    def model_function(x_t: Data, time: torch.Tensor) -> Data:
        """The real model, with each velocity field replaced by a differentiable leaf holding the same values."""
        prediction = module.model(x_t, time)
        for field in FIELDS:
            key = f"{field}_b"
            leaf = prediction[key].detach().requires_grad_(True)
            captured[key] = leaf
            prediction[key] = leaf
        return prediction

    losses = module.si.losses(model_function, t, x_0, x_1)

    summary = {}
    for field in FIELDS:
        leaf = captured[f"{field}_b"]
        loss = losses[f"{field}_loss_b"]
        # retain_graph because the two fields' losses share one forward pass through the trunk.
        gradient, = torch.autograd.grad(loss, leaf, retain_graph=True)
        count = leaf.numel()
        # Equation (3): the error is exactly (N/2) dL/db, so it needs no target to be formed at all.
        error = 0.5 * count * gradient
        target = leaf.detach() - error
        summary[field] = {"squared_error": error.square().sum().detach(),
                          "squared_target": target.square().sum(),
                          "count": torch.tensor(float(count), device=device),
                          "loss": loss.detach() * count}
    return summary


def evaluate(module: OMGLightning, loader, times: tuple[float, ...], structures: Optional[int], seed: int,
             device: torch.device) -> dict[str, dict]:
    """
    Pool the per-batch sums over the split and turn them into the reported errors.

    :param module:
        The trained module.
    :type module: omg.omg_lightning.OMGLightning
    :param loader:
        Iterable of batches.
    :param times:
        Denoising times to report at.
    :type times: tuple[float, ...]
    :param structures:
        Stop after about this many structures, or ``None`` for the whole split.
    :type structures: Optional[int]
    :param seed:
        Seed of the base draws.
    :type seed: int
    :param device:
        Device to evaluate on.
    :type device: torch.device

    :return:
        Per time, per field: mean squared error, the zero predictor's error, their ratio, and the identity check.
    :rtype: dict[str, dict]
    """
    totals = {f"{time:g}": {field: {key: torch.zeros((), device=device)
                                    for key in ("squared_error", "squared_target", "count", "loss")}
                            for field in FIELDS} for time in times}
    seen = 0
    for index, batch in enumerate(loader):
        batch = batch.to(device)
        for time in times:
            # The seed carries the batch index, so each batch gets its own corruption, and the time, so the buckets are
            # independent draws rather than one draw viewed at several times.
            summary = errors_for_batch(module, batch, time, seed + 1_000 * index + int(round(time * 100)))
            for field in FIELDS:
                for key, value in summary[field].items():
                    totals[f"{time:g}"][field][key] += value
        seen += len(batch.n_atoms)
        if structures is not None and seen >= structures:
            break

    report = {}
    for time in times:
        label = f"{time:g}"
        entry = {}
        for field in FIELDS:
            sums = totals[label][field]
            count = sums["count"].item()
            mean_error = sums["squared_error"].item() / count
            zero = sums["squared_target"].item() / count
            loss = sums["loss"].item() / count
            # Equation (2). If the reconstruction were wrong this is where it shows.
            residual = abs(mean_error - (loss + zero)) / max(zero, 1.0e-12)
            entry[field] = {"mean_squared_error": mean_error, "zero_predictor_error": zero,
                            "relative_error": mean_error / zero if zero > 0.0 else float("nan"),
                            "omatg_loss": loss, "identity_residual": residual,
                            "identity_holds": residual < TOLERANCE}
        entry["structures"] = seen
        report[label] = entry
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """
    Evaluate one run's checkpoint and write its fixed-time errors.

    :param argv:
        Command line arguments, or ``None`` to read ``sys.argv``.
    :type argv: Optional[list[str]]

    :return:
        Process exit status: nonzero if the loss identity failed anywhere.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument("--run-dir", required=True, help="run directory holding COMMAND and checkpoints")
    parser.add_argument("--checkpoint", default=None, help="checkpoint to evaluate, default the match-rate best")
    parser.add_argument("--data-dir", default="omg/data/mpts_52", help="directory of the split LMDBs")
    parser.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES), help="fixed times to report at")
    parser.add_argument("--structures", type=int, default=1024, help="validation structures to pool over")
    parser.add_argument("--batch-size", type=int, default=64, help="structures per batch")
    parser.add_argument("--seed", type=int, default=0, help="seed of the base draws, shared across arms")
    parser.add_argument("--out", default=None, help="where to write the report, default TARGET-MSE.json in the run")
    settings = parser.parse_args(argv)

    run_dir = Path(settings.run_dir)
    arm = arm_of(run_dir)
    checkpoint_path = Path(settings.checkpoint) if settings.checkpoint else newest_checkpoint(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module, data = build(arm, settings.data_dir, settings.batch_size, device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=True)
    print(f"{run_dir.name}: graph {arm['message_graph']}, features {arm['feature_mode']}, "
          f"from {checkpoint_path.name}")

    data.setup("validate")
    report = evaluate(module, data.val_dataloader(), tuple(settings.times), settings.structures, settings.seed, device)

    print(f"\n{'t':>6}  {'pos MSE':>12} {'rel':>7}   {'cell MSE':>12} {'rel':>7}")
    for label, entry in report.items():
        print(f"{label:>6}  {entry['pos']['mean_squared_error']:12.5g} {entry['pos']['relative_error']:7.4f}   "
              f"{entry['cell']['mean_squared_error']:12.5g} {entry['cell']['relative_error']:7.4f}")

    broken = [(label, field) for label, entry in report.items() for field in FIELDS
              if not entry[field]["identity_holds"]]
    destination = Path(settings.out) if settings.out else run_dir / "TARGET-MSE.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(
        {"run": str(run_dir), "checkpoint": str(checkpoint_path), "arm": arm, "seed": settings.seed,
         "structures": settings.structures, "times": list(settings.times), "errors": report,
         "identity_failures": [f"{label}/{field}" for label, field in broken]}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}")

    if broken:
        print(f"\nthe loss identity failed at {', '.join(f'{a}/{b}' for a, b in broken)}: the reconstructed target does "
              f"not satisfy MSE = loss + mean(target^2), so none of these numbers can be trusted.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
