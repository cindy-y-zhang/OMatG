"""
Gates DG1 and DG2: is the coordination environment *in* the descriptor, before any generative model is trained?

WHY THIS RUNS BEFORE ANY TRAINING RUN

A denoiser cannot use information its input does not contain. The two previous attempts in this project each spent days of
GPU time discovering that after the fact, so the question is asked first and cheaply here: fit a small probe to name each
atom's coordination number, and its coordination geometry, from the descriptor alone, and compare against a probe given
everything *except* the geometry. If the gap is small the pivot stops at a cost of minutes.

The labels are the precomputed CrystalNN tables under ``cgfm/motifs/``. They are read-only diagnostics. Nothing here feeds
the denoiser, appears in a loss, or is available at sampling time -- that was the previous pivot, and it failed.

WHAT MAKES THIS A VALID INSTRUMENT RATHER THAN JUST A CLASSIFIER

The predecessor to this measurement was a six-layer message-passing classifier, and it reported *negative* information
against a plain element prior -- minus 0.489 bits -- while scoring high accuracy. It had memorised the training split. A
two-layer version of the same probe reported about plus 1.16 bits. An instrument that can memorise is not measuring the
thing it is pointed at, so this one is deliberately weak and deliberately fixed:

- the probe reads *per-atom features*, with no message passing, so it cannot recover geometry the descriptor omits;
- depth is not searched: one linear probe and one two-layer probe, both reported, the gate read on the two-layer one with
  the linear one required to agree in sign;
- the budget, the optimiser and the weight decay are the same for every arm, and training early-stops on validation
  cross-entropy;
- the train-minus-validation cross-entropy gap is reported for every arm, so a memorising probe is visible rather than
  quoted.

THE FLOOR

The geometry-ablated arm is handed the descriptor of an *independent draw from the base distribution* -- a different
crystal entirely, sharing only the atom count -- while keeping the true species and structure size. It therefore measures
what the composition alone says about coordination, which on this dataset is a great deal: elements have preferred
coordination numbers. The floor is time-independent by construction, since none of its inputs depends on the denoising
time, so it is fitted once and used as the same denominator at every corruption level.

Probes are trained at fixed denoising times rather than jointly across them, which is what makes the time input the sweep
axis rather than a feature column: both arms are always trained and evaluated at the same time, so neither is handicapped
by not knowing it.

Usage:

    python -m direct_geometry.scripts.probe_features
    python -m direct_geometry.scripts.probe_features --train-structures 4000 --steps 400   # quick look
"""

import argparse
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from direct_geometry.batches import displacement, interpolate, iterate_structures, load_split
from direct_geometry.features import DescriptorSpec, FeatureMode, local_environment_descriptor
from direct_geometry.neighbors import periodic_neighbors


SPLITS = {"train": "omg/data/mpts_52/train.lmdb", "val": "omg/data/mpts_52/val.lmdb"}
"""
The two splits the probes use.

The test split is deliberately absent. The promoted descriptor is selected on validation information gain, and a test
number measured now would be a number available to influence that selection.
"""

TASKS = {"coordination": ("cgfm/motifs/mpts_52_cn", 13), "geometry": ("cgfm/motifs/mpts_52_geom", 28)}
"""
The two label tables, as directory and class count.

Both are needed because the gates ask different questions. Coordination number is a count, and Gate DG1 asks whether the
radial block can count. Coordination geometry is a shape at a given count, and Gate DG2 asks whether the angular block
adds anything once the count is known -- which is exactly the question the 28-class table was built to answer.
"""

PROBE_MODES = (FeatureMode.RADIAL, FeatureMode.ANGULAR, FeatureMode.BOTH)
"""
Descriptor subsets probed. ``none`` is absent because a probe with no descriptor is the ablated arm's floor already.
"""

DEFAULT_TIMES = (0.5, 0.8, 0.9, 0.95, 0.99, 1.0)
"""
Denoising times swept, chosen for what they mean in Angstrom rather than for round numbers.

Measured on MPTS-52, the root-mean-square displacement from the crystal is 2.06 Angstrom at 0.5, 0.45 at 0.9, 0.23 at 0.95
and zero at 1.0. So 1.0 is the clean endpoint both gates are read at, and 0.95 is the "near 0.2 Angstrom" corruption Gate
DG2's retention clause names. The report states the measured corruption at every time, so the times are never quoted as
though they were physical.
"""

MAX_ATOMIC_NUMBER = 100
"""Largest atomic number the species encoding covers, matching the width of the label tables' own element priors."""

DEPTHS = ("linear", "two_layer")
"""
The only two probe architectures. Depth is not searched: an instrument whose capacity is tuned per arm would report its
own tuning as information.
"""

GATE_DEPTH = "two_layer"
"""
Probe the gates are read on.

Predeclared rather than chosen after the fact. The two-layer probe is the stronger of the two instruments and the one the
project's earlier diagnostic found to be well calibrated; the linear probe is required to agree in sign, which is the check
that a gain is in the features rather than in the probe's capacity.
"""

DG1_ACCURACY_POINTS = 20.0
"""Coordination-number accuracy points the radial block must add over the matched floor at the clean endpoint."""

DG1_BITS = 0.25
"""Bits of information the radial block must add over the matched floor at the clean endpoint."""

DG2_SHAPE_POINTS = 5.0
"""Shape-given-correct-coordination points the angular block must add over radial alone at the clean endpoint."""

DG2_BITS = 0.10
"""Bits the angular block must add over radial alone at the clean endpoint."""

DG2_RETENTION = 0.5
"""Fraction of its clean information gain the angular block must keep at about 0.2 Angstrom of corruption."""

RETENTION_TIME = 0.95
"""Denoising time Gate DG2's retention clause is read at, being the sweep point nearest 0.2 Angstrom of corruption."""


@dataclass(frozen=True)
class Labels:
    """
    A precomputed per-atom label table, aligned to a split's stored order.

    :param target:
        Class of every atom, of shape ``(atoms,)``, with ``-1`` where the labeller failed.
    :type target: torch.Tensor
    :param coordination:
        Coordination number of every class, of shape ``(classes,)``. For the coordination task this is the class index
        itself; for the geometry task it is parsed from the class names, several of which share a coordination number.
    :type coordination: torch.Tensor
    :param names:
        Name of every class.
    :type names: tuple[str, ...]
    """

    target: torch.Tensor
    coordination: torch.Tensor
    names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        """
        Return the number of classes.

        :return:
            Number of classes.
        :rtype: int
        """
        return int(self.coordination.numel())


def coordination_of(names: tuple[str, ...]) -> torch.Tensor:
    """
    Read each class's coordination number out of its name.

    Parsed from the manifest rather than imported from ``cgfm.coordination_geometry``, so that this package keeps no
    runtime dependency on the label stack it is a diagnostic consumer of. The names carry the number explicitly -- "octahedral
    CN_6", "CN_12_or_more" -- and the parse is asserted to cover every class rather than defaulting quietly.

    :param names:
        Class names, each containing a ``CN_<number>`` token.
    :type names: tuple[str, ...]

    :raises ValueError:
        If a name carries no coordination number, which would mean the vocabulary has changed shape and the shape metrics
        would silently be measuring something else.

    :return:
        Coordination number of every class, of shape ``(classes,)``.
    :rtype: torch.Tensor
    """
    numbers = []
    for name in names:
        found = re.search(r"CN_(\d+)", name)
        if found is None:
            raise ValueError(f"Class name {name!r} carries no CN_<number> token, so its coordination number is unknown.")
        numbers.append(int(found.group(1)))
    return torch.tensor(numbers, dtype=torch.long)


def load_labels(task: str, split: str, structures: Optional[int] = None) -> Labels:
    """
    Read a label table for one split.

    The two tables encode failure differently and both are normalised to ``-1`` here. The geometry table carries an explicit
    ``labels`` array with ``-1`` for atoms whose order parameters were undefined. The coordination table carries only
    one-hot responsibilities, and a structure whose neighbour finding failed is written as a *uniform* row -- which an
    argmax would silently turn into class zero, a real coordination number, for an atom that has no measured one.

    :param task:
        Task name, a key of ``TASKS``.
    :type task: str
    :param split:
        Split name, a key of ``SPLITS``.
    :type split: str
    :param structures:
        Number of leading structures to keep, or None for all of them. Must match what the descriptors were computed over.
        Defaults to None.
    :type structures: Optional[int]

    :raises FileNotFoundError:
        If the table has not been precomputed.
    :raises ValueError:
        If the table does not describe the requested number of structures.

    :return:
        The labels.
    :rtype: Labels
    """
    directory, classes = TASKS[task]
    table = Path(directory) / f"motifs_K{classes}_{split}.npz"
    if not table.exists():
        raise FileNotFoundError(f"No label table at {table}. It is produced by cgfm/scripts/precompute_geometry.py and "
                                f"cgfm/scripts/precompute_coordination.py, and this script only reads it.")
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    names = tuple(manifest["settings"].get("class_names") or [f"CN_{index}" for index in range(classes)])

    loaded = np.load(table, allow_pickle=False)
    offsets = loaded["atom_offsets"]
    if structures is not None:
        if structures + 1 > offsets.shape[0]:
            raise ValueError(f"{table} covers {offsets.shape[0] - 1} structures, fewer than the {structures} requested.")
        keep = int(offsets[structures])
    else:
        keep = int(offsets[-1])

    if "labels" in loaded.files:
        target = torch.from_numpy(loaded["labels"][:keep].astype(np.int64))
    else:
        responsibilities = loaded["responsibilities"][:keep].astype(np.float32)
        target = torch.from_numpy(responsibilities.argmax(axis=1).astype(np.int64))
        # A uniform row is the coordination table's way of recording a failure, and it has no argmax worth taking.
        target[torch.from_numpy(responsibilities.max(axis=1)) < 0.5] = -1
    return Labels(target=target, coordination=coordination_of(names), names=names)


def descriptors_for_split(split: str, spec: DescriptorSpec, time_value: float, seed: int, batch_size: int,
                          structures: Optional[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor,
                                                                                   torch.Tensor, float]:
    """
    Compute the full descriptor of every atom of a split at one denoising time.

    Streamed in stored order, so that row ``i`` of the result is atom ``i`` of the label table. The full twenty-two channels
    are computed once and masked per mode by the caller, since the blocks share a neighbour list and computing them
    separately would triple the cost for no difference in the answer.

    :param split:
        Split name, a key of ``SPLITS``.
    :type split: str
    :param spec:
        Descriptor specification.
    :type spec: DescriptorSpec
    :param time_value:
        Denoising time in ``[0, 1]``.
    :type time_value: float
    :param seed:
        Seed of the base draws. Combined with the batch index, so every batch gets its own draw and the whole pass is still
        reproducible from this one number.
    :type seed: int
    :param batch_size:
        Number of structures per batch.
    :type batch_size: int
    :param structures:
        Number of leading structures to read, or None for the whole split.
    :type structures: Optional[int]
    :param device:
        Device to compute on.
    :type device: torch.device

    :return:
        Descriptors of shape ``(atoms, spec.dim)``, atomic numbers of shape ``(atoms,)``, structure sizes broadcast to
        atoms of shape ``(atoms,)``, and the root-mean-square displacement from the crystal in Angstrom.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
    """
    dataset = load_split(SPLITS[split])
    blocks, numbers, sizes, squared_displacement, counted = [], [], [], 0.0, 0
    for index, batch in iterate_structures(dataset, batch_size, structures):
        frac, cell = interpolate(batch, time_value, seed + index)
        moved = displacement(batch, frac, cell)
        squared_displacement += float(moved.pow(2).sum())
        counted += moved.numel()
        neighbors = periodic_neighbors(frac.to(device), cell.to(device), batch.n_atoms.to(device), spec.cutoff)
        blocks.append(local_environment_descriptor(neighbors, frac.shape[0], spec, FeatureMode.BOTH).float().cpu())
        numbers.append(batch.species)
        sizes.append(batch.n_atoms.repeat_interleave(batch.n_atoms))
    return (torch.cat(blocks), torch.cat(numbers), torch.cat(sizes),
            (squared_displacement / max(counted, 1)) ** 0.5)


def assemble(descriptor: torch.Tensor, numbers: torch.Tensor, sizes: torch.Tensor, mode: FeatureMode,
             spec: DescriptorSpec) -> torch.Tensor:
    """
    Build the probe's input: the enabled descriptor channels, the species, and the structure size.

    The disabled channels are dropped rather than zeroed, because a column that is exactly constant contributes nothing but
    a bias the probe already has, and carrying it would make the ``angular`` arm look as though it had a twenty-two
    dimensional input when it has five.

    Species enter as a one-hot code rather than a learned embedding, which makes the linear probe's species-only solution
    exactly the element-conditional prior -- the same baseline the project's earlier diagnostic was scored against.

    :param descriptor:
        Full descriptors of shape ``(atoms, spec.dim)``.
    :type descriptor: torch.Tensor
    :param numbers:
        Atomic numbers of shape ``(atoms,)``.
    :type numbers: torch.Tensor
    :param sizes:
        Structure size of every atom, of shape ``(atoms,)``.
    :type sizes: torch.Tensor
    :param mode:
        Which descriptor blocks the probe may see.
    :type mode: FeatureMode
    :param spec:
        Descriptor specification.
    :type spec: DescriptorSpec

    :raises ValueError:
        If an atomic number is outside the range the species encoding covers.

    :return:
        Features of shape ``(atoms, enabled + MAX_ATOMIC_NUMBER + 1)``.
    :rtype: torch.Tensor
    """
    if int(numbers.max()) > MAX_ATOMIC_NUMBER or int(numbers.min()) < 1:
        raise ValueError(f"Atomic numbers span {int(numbers.min())} to {int(numbers.max())}, outside the 1 to "
                         f"{MAX_ATOMIC_NUMBER} the species encoding covers.")
    enabled = spec.channel_mask(mode).to(torch.bool)
    species = functional.one_hot(numbers.long() - 1, num_classes=MAX_ATOMIC_NUMBER).float()
    return torch.cat([descriptor[:, enabled], species, torch.log1p(sizes.float()).unsqueeze(-1)], dim=1)


def standardise(train: torch.Tensor, others: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """
    Centre and scale features by the training split's own statistics.

    Necessary rather than cosmetic: the descriptor mixes a normalised shell distribution against a ``log1p`` mass and
    Legendre moments, whose scales differ by more than an order of magnitude, and a fixed-budget optimiser on unscaled
    inputs would report the badly scaled channels as uninformative. Constant columns -- an element that never appears --
    are left at zero rather than divided by nothing.

    :param train:
        Training features, whose statistics are used.
    :type train: torch.Tensor
    :param others:
        Further feature matrices to transform with the same statistics.
    :type others: tuple[torch.Tensor, ...]

    :return:
        The training features and the others, standardised.
    :rtype: tuple[torch.Tensor, ...]
    """
    centre = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, keepdim=True)
    scale = torch.where(scale > 1.0e-8, scale, torch.ones_like(scale))
    return tuple((matrix - centre) / scale for matrix in (train, *others))


def build_probe(features: int, classes: int, depth: str, hidden: int) -> nn.Module:
    """
    Build one of the two fixed probe architectures.

    :param features:
        Input width.
    :type features: int
    :param classes:
        Number of classes.
    :type classes: int
    :param depth:
        Either "linear" or "two_layer".
    :type depth: str
    :param hidden:
        Hidden width of the two-layer probe.
    :type hidden: int

    :raises ValueError:
        If the depth is not one of the two fixed choices, since searching depth would make the probe a model rather than
        an instrument.

    :return:
        The probe.
    :rtype: torch.nn.Module
    """
    if depth == "linear":
        return nn.Linear(features, classes)
    if depth == "two_layer":
        return nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, classes))
    raise ValueError(f"Unknown probe depth {depth!r}. Only 'linear' and 'two_layer' exist, by design.")


def evaluate(probe: nn.Module, features: torch.Tensor, target: torch.Tensor, labels: Labels,
             chunk: int = 65536) -> dict[str, float]:
    """
    Score a probe on labelled atoms.

    :param probe:
        The probe.
    :type probe: torch.nn.Module
    :param features:
        Features of shape ``(atoms, width)``.
    :type features: torch.Tensor
    :param target:
        Classes of shape ``(atoms,)``, all of them valid.
    :type target: torch.Tensor
    :param labels:
        The label table, for the class-to-coordination map.
    :type labels: Labels
    :param chunk:
        Number of atoms scored per pass.
        Defaults to 65536.
    :type chunk: int

    :return:
        Top-one accuracy, cross-entropy in bits, coordination accuracy, shape accuracy with the coordination number given,
        and the share of atoms the shape metric is defined on.
    :rtype: dict[str, float]
    """
    probe.eval()
    coordination = labels.coordination.to(features.device)
    # Coordination numbers the vocabulary offers more than one shape for. At CN 1, and at 9 and above, there is a single
    # class, so naming the count names the class, and scoring those atoms would report counting under a shape heading.
    populations = torch.bincount(coordination, minlength=int(coordination.max()) + 1)
    ambiguous_class = (populations > 1)[coordination]

    total_loss, correct, counted, shape_correct, shape_seen, seen = 0.0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for start in range(0, features.shape[0], chunk):
            piece = target[start:start + chunk]
            logits = probe(features[start:start + chunk])
            total_loss += float(functional.cross_entropy(logits, piece, reduction="sum"))
            chosen = logits.argmax(dim=1)
            correct += int((chosen == piece).sum())
            counted += int((coordination[chosen] == coordination[piece]).sum())
            seen += int(piece.numel())

            # The coordination number is *given* rather than predicted: the argmax is taken only over the classes sharing
            # the true count. Without that the denominator would move with the arm -- a probe that counts better puts more
            # and harder atoms into a "coordination was right" denominator, and its shape score falls even as it improves,
            # which is the opposite of what Gate DG2 needs to read.
            allowed = coordination.unsqueeze(0) == coordination[piece].unsqueeze(1)
            restricted = logits.masked_fill(~allowed, -float("inf")).argmax(dim=1)
            scored = ambiguous_class[piece]
            shape_correct += int((restricted[scored] == piece[scored]).sum())
            shape_seen += int(scored.sum())

    return {"accuracy": correct / max(seen, 1),
            "cross_entropy_bits": total_loss / max(seen, 1) / float(np.log(2.0)),
            "coordination_accuracy": counted / max(seen, 1),
            "shape_given_coordination": shape_correct / shape_seen if shape_seen else float("nan"),
            "shape_denominator_share": shape_seen / max(seen, 1)}


def train_probe(train_features: torch.Tensor, train_target: torch.Tensor, val_features: torch.Tensor,
                val_target: torch.Tensor, labels: Labels, depth: str, settings: argparse.Namespace,
                seed: int) -> dict[str, float]:
    """
    Fit one probe under the fixed budget and return its validation scores.

    Every arm gets the same optimiser, the same weight decay, the same batch size, the same step ceiling and the same
    early-stopping rule, because the quantity being compared is a difference between arms and any per-arm tuning would
    appear in that difference as information.

    :param train_features:
        Training features of shape ``(atoms, width)``.
    :type train_features: torch.Tensor
    :param train_target:
        Training classes of shape ``(atoms,)``.
    :type train_target: torch.Tensor
    :param val_features:
        Validation features.
    :type val_features: torch.Tensor
    :param val_target:
        Validation classes.
    :type val_target: torch.Tensor
    :param labels:
        The label table.
    :type labels: Labels
    :param depth:
        Probe depth.
    :type depth: str
    :param settings:
        Parsed command-line settings, for the budget.
    :type settings: argparse.Namespace
    :param seed:
        Seed of the initial weights and the batch order.
    :type seed: int

    :return:
        Validation scores, the training cross-entropy for comparison, and the step the best validation score was reached
        at.
    :rtype: dict[str, float]
    """
    device = train_features.device
    torch.manual_seed(seed)
    probe = build_probe(train_features.shape[1], labels.num_classes, depth, settings.hidden).to(device)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    generator = torch.Generator(device=device.type).manual_seed(seed)

    best, best_step, best_state, stale = float("inf"), 0, None, 0
    for step in range(1, settings.steps + 1):
        probe.train()
        pick = torch.randint(0, train_features.shape[0], (settings.batch_size,), generator=generator, device=device)
        loss = functional.cross_entropy(probe(train_features[pick]), train_target[pick])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % settings.evaluate_every and step != settings.steps:
            continue
        current = evaluate(probe, val_features, val_target, labels)["cross_entropy_bits"]
        if current < best - 1.0e-5:
            best, best_step, stale = current, step, 0
            best_state = {name: value.detach().clone() for name, value in probe.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)

    scores = evaluate(probe, val_features, val_target, labels)
    sample = torch.randperm(train_features.shape[0], generator=generator, device=device)[:settings.train_sample]
    scores["train_cross_entropy_bits"] = evaluate(probe, train_features[sample], train_target[sample],
                                                  labels)["cross_entropy_bits"]
    scores["generalisation_gap_bits"] = scores["cross_entropy_bits"] - scores["train_cross_entropy_bits"]
    scores["best_step"] = float(best_step)
    return scores


def run_task(task: str, spec: DescriptorSpec, settings: argparse.Namespace, device: torch.device) -> dict:
    """
    Probe one label table across the descriptor modes, the two depths and the sweep of denoising times.

    :param task:
        Task name, a key of ``TASKS``.
    :type task: str
    :param spec:
        Descriptor specification.
    :type spec: DescriptorSpec
    :param settings:
        Parsed command-line settings.
    :type settings: argparse.Namespace
    :param device:
        Device to fit on.
    :type device: torch.device

    :return:
        The floor scores, the scores at every time, and the measured corruption at every time.
    :rtype: dict
    """
    train_labels = load_labels(task, "train", settings.train_structures)
    val_labels = load_labels(task, "val", settings.val_structures)
    print(f"\n{task}: {train_labels.num_classes} classes, "
          f"{int((train_labels.target >= 0).sum())} labelled training atoms, "
          f"{int((val_labels.target < 0).sum())} unlabelled validation atoms")

    # The ablated floor is fitted once. None of its inputs depends on the denoising time, so refitting it per time would
    # only add sampling noise to the denominator of every information gain.
    ablated = {"train": descriptors_for_split("train", spec, 0.0, settings.seed + 9_000, settings.batch_structures,
                                              settings.train_structures, device),
               "val": descriptors_for_split("val", spec, 0.0, settings.seed + 19_000, settings.batch_structures,
                                            settings.val_structures, device)}

    train_keep = (train_labels.target >= 0).to(device)
    val_keep = (val_labels.target >= 0).to(device)
    train_target = train_labels.target.to(device)[train_keep]
    val_target = val_labels.target.to(device)[val_keep]

    def fit(shown: dict, mode: FeatureMode) -> dict[str, dict[str, float]]:
        """Assemble and standardise one arm's features once, then fit both probe depths on them."""
        train_matrix = assemble(*shown["train"][:3], mode, spec).to(device)
        val_matrix = assemble(*shown["val"][:3], mode, spec).to(device)
        train_matrix, val_matrix = standardise(train_matrix, (val_matrix,))
        train_matrix, val_matrix = train_matrix[train_keep], val_matrix[val_keep]
        return {depth: train_probe(train_matrix, train_target, val_matrix, val_target, val_labels, depth, settings,
                                  settings.seed) for depth in DEPTHS}

    floors = {}
    for mode in PROBE_MODES:
        for depth, scores in fit(ablated, mode).items():
            floors[f"{mode.value}/{depth}"] = scores
            print(f"  floor  {mode.value:>7} {depth:>9}  CE {scores['cross_entropy_bits']:.3f} bits  "
                  f"acc {scores['accuracy']:.3f}  gap {scores['generalisation_gap_bits']:+.3f}")

    per_time, corruption = {}, {}
    for time_value in settings.times:
        shown = {"train": descriptors_for_split("train", spec, time_value, settings.seed, settings.batch_structures,
                                                settings.train_structures, device),
                 "val": descriptors_for_split("val", spec, time_value, settings.seed + 1_000,
                                              settings.batch_structures, settings.val_structures, device)}
        corruption[f"{time_value:g}"] = shown["val"][3]
        entry = {}
        for mode in PROBE_MODES:
            for depth, scores in fit(shown, mode).items():
                scores["information_bits"] = (floors[f"{mode.value}/{depth}"]["cross_entropy_bits"]
                                              - scores["cross_entropy_bits"])
                entry[f"{mode.value}/{depth}"] = scores
        per_time[f"{time_value:g}"] = entry
        best = entry[f"{FeatureMode.BOTH.value}/{GATE_DEPTH}"]
        print(f"  t={time_value:<6g} rms {shown['val'][3]:.3f} A   both/{GATE_DEPTH}: "
              f"acc {best['accuracy']:.3f}  CE {best['cross_entropy_bits']:.3f} bits  "
              f"gain {best['information_bits']:+.3f} bits  gap {best['generalisation_gap_bits']:+.3f}")
    return {"floors": floors, "times": per_time, "corruption": corruption,
            "classes": list(val_labels.names),
            "unlabelled_validation_atoms": int((val_labels.target < 0).sum())}


def read_gates(results: dict[str, dict], clean_time: float) -> tuple[dict, list[str]]:
    """
    Read Gates DG1 and DG2 and decide which descriptor is promoted.

    :param results:
        Output of ``run_task`` for both tasks.
    :type results: dict[str, dict]
    :param clean_time:
        The sweep's clean endpoint, where both gates are read.
    :type clean_time: float

    :return:
        The verdict, and the reasons any clause failed.
    :rtype: tuple[dict, list[str]]
    """
    clean = f"{clean_time:g}"
    radial, both = f"{FeatureMode.RADIAL.value}/{GATE_DEPTH}", f"{FeatureMode.BOTH.value}/{GATE_DEPTH}"
    linear_radial, linear_both = f"{FeatureMode.RADIAL.value}/linear", f"{FeatureMode.BOTH.value}/linear"

    counting = results["coordination"]
    radial_clean = counting["times"][clean][radial]
    radial_floor = counting["floors"][radial]
    # Read on the coordination task's own accuracy, which for that table is the top-one accuracy: every class is a count.
    dg1_points = 100.0 * (radial_clean["accuracy"] - radial_floor["accuracy"])
    dg1_bits = radial_clean["information_bits"]
    dg1_linear = 100.0 * (counting["times"][clean][linear_radial]["accuracy"]
                          - counting["floors"][linear_radial]["accuracy"])
    dg1 = {"accuracy_points": dg1_points, "bits": dg1_bits, "linear_accuracy_points": dg1_linear,
           "passed": dg1_points >= DG1_ACCURACY_POINTS and dg1_bits >= DG1_BITS and dg1_linear > 0.0}

    shapes = results["geometry"]
    shape_clean, shape_radial = shapes["times"][clean][both], shapes["times"][clean][radial]
    dg2_points = 100.0 * (shape_clean["shape_given_coordination"] - shape_radial["shape_given_coordination"])
    dg2_bits = shape_clean["information_bits"] - shape_radial["information_bits"]
    dg2_linear = 100.0 * (shapes["times"][clean][linear_both]["shape_given_coordination"]
                          - shapes["times"][clean][linear_radial]["shape_given_coordination"])
    retention_key = f"{RETENTION_TIME:g}"
    corrupted = shapes["times"].get(retention_key)
    retained = ((corrupted[both]["information_bits"] - corrupted[radial]["information_bits"]) / dg2_bits
                if corrupted is not None and dg2_bits > 0.0 else 0.0)
    dg2 = {"shape_points": dg2_points, "bits": dg2_bits, "linear_shape_points": dg2_linear,
           "retained_fraction": retained, "retention_time": RETENTION_TIME,
           "retention_corruption_angstrom": shapes["corruption"].get(retention_key),
           "passed": (dg2_points >= DG2_SHAPE_POINTS and dg2_bits >= DG2_BITS and dg2_linear > 0.0
                      and retained >= DG2_RETENTION)}

    reasons = []
    if not dg1["passed"]:
        reasons.append(f"DG1 fails: radial adds {dg1_points:+.1f} coordination points (needs "
                       f"{DG1_ACCURACY_POINTS:+.0f}) and {dg1_bits:+.3f} bits (needs {DG1_BITS:+.2f}) over the matched "
                       f"floor, with the linear probe at {dg1_linear:+.1f} points. The descriptor does not carry the "
                       f"coordination environment, so the pivot stops here.")
    if not dg2["passed"]:
        reasons.append(f"DG2 fails: angular adds {dg2_points:+.1f} shape points (needs {DG2_SHAPE_POINTS:+.0f}) and "
                       f"{dg2_bits:+.3f} bits (needs {DG2_BITS:+.2f}) over radial alone, retaining "
                       f"{retained:.2f} of that at {RETENTION_TIME:g} (needs {DG2_RETENTION:.2f}), with the linear probe "
                       f"at {dg2_linear:+.1f} points. Angular channels are dropped and radial is promoted alone.")

    promoted = None if not dg1["passed"] else (FeatureMode.BOTH.value if dg2["passed"] else FeatureMode.RADIAL.value)
    return {"DG1": dg1, "DG2": dg2, "promoted": promoted, "gate_depth": GATE_DEPTH}, reasons


def main(argv: Optional[list[str]] = None) -> int:
    """
    Run the probes and write the Gate DG1/DG2 verdict.

    :param argv:
        Command-line arguments, or None to read them from the process.
        Defaults to None.
    :type argv: Optional[list[str]]

    :return:
        Zero if Gate DG1 passes, since a DG1 failure ends the pivot while a DG2 failure only narrows the descriptor.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES), help="denoising times to sweep")
    parser.add_argument("--train-structures", type=int, default=None,
                        help="number of leading training structures, or all of them")
    parser.add_argument("--val-structures", type=int, default=None,
                        help="number of leading validation structures, or all of them")
    parser.add_argument("--batch-structures", type=int, default=256, help="structures per descriptor batch")
    parser.add_argument("--steps", type=int, default=3000, help="optimiser steps per probe, the same for every arm")
    parser.add_argument("--batch-size", type=int, default=8192, help="atoms per probe step")
    parser.add_argument("--hidden", type=int, default=128, help="hidden width of the two-layer probe")
    parser.add_argument("--learning-rate", type=float, default=3.0e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1.0e-2,
                        help="AdamW weight decay, fixed by the plan rather than tuned")
    parser.add_argument("--evaluate-every", type=int, default=100, help="steps between validation evaluations")
    parser.add_argument("--patience", type=int, default=8, help="evaluations without improvement before stopping")
    parser.add_argument("--train-sample", type=int, default=100_000,
                        help="training atoms scored for the generalisation gap")
    parser.add_argument("--descriptor-cutoff", type=float, default=6.0, help="descriptor radius in Angstrom")
    parser.add_argument("--descriptor-shells", type=int, default=16, help="radial shells")
    parser.add_argument("--descriptor-angular-order", type=int, default=4, help="highest Legendre order")
    parser.add_argument("--seed", type=int, default=0, help="seed of the base draws and the probe weights")
    parser.add_argument("--out", default="direct_geometry/reports/DG1-DG2-PROBES.json", help="where to write the report")
    arguments = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DescriptorSpec(cutoff=arguments.descriptor_cutoff, num_shells=arguments.descriptor_shells,
                          max_angular_order=arguments.descriptor_angular_order)
    print(f"probing a {spec.dim}-channel descriptor on {device}, times {arguments.times}")

    results = {task: run_task(task, spec, arguments, device) for task in TASKS}
    verdict, reasons = read_gates(results, max(arguments.times))

    report = {"gates": ["DG1", "DG2"], "verdict": verdict, "failures": reasons, "arguments": vars(arguments),
              "environment": {"torch": torch.__version__, "device": str(device), "platform": platform.platform(),
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "descriptor": {"cutoff": spec.cutoff, "shells": spec.num_shells,
                             "angular_order": spec.max_angular_order, "dim": spec.dim},
              "results": results}
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for task, result in results.items():
        print(f"\n{task}: information gain over the matched floor, in bits ({GATE_DEPTH} probe)")
        header = "  ".join(f"{mode.value:>9}" for mode in PROBE_MODES)
        print(f"  {'t':>6}  {'rms A':>6}  {header}")
        for label, entry in result["times"].items():
            gains = "  ".join(f"{entry[f'{mode.value}/{GATE_DEPTH}']['information_bits']:>+9.3f}"
                              for mode in PROBE_MODES)
            print(f"  {label:>6}  {result['corruption'][label]:>6.3f}  {gains}")

    print(f"\nGate DG1 (radial accessibility): {'PASS' if verdict['DG1']['passed'] else 'FAIL'}")
    print(f"  {verdict['DG1']['accuracy_points']:+.1f} coordination points, {verdict['DG1']['bits']:+.3f} bits")
    print(f"Gate DG2 (angular value): {'PASS' if verdict['DG2']['passed'] else 'FAIL'}")
    print(f"  {verdict['DG2']['shape_points']:+.1f} shape points, {verdict['DG2']['bits']:+.3f} bits, "
          f"{verdict['DG2']['retained_fraction']:.2f} retained at {RETENTION_TIME:g}")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"promoted descriptor: {verdict['promoted']}")
    print(f"wrote {destination}")
    return 0 if verdict["DG1"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
