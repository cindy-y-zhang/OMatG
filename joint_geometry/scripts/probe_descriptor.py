"""
Compression and conditional-information gate for CN-RDF endpoint states.

The reported bit gains are cross-validated reductions in probe
cross-entropy relative to chemistry-only inputs.  They are usable
information for the fixed probe family, not unrestricted mutual information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from direct_geometry.batches import collate, load_split
from direct_geometry.scripts.probe_features import (
    DEPTHS,
    Labels,
    load_labels,
    standardise,
    train_probe,
)
from joint_geometry.data import GEOMETRY_TABLE_NAME, GeometryTable, within_element_permutation
from joint_geometry.descriptor import DescriptorTransform, clean_radial_descriptor


REPRESENTATIONS = ("cn-rdf4", "cn-rdf8", "radial17")
TASKS = ("coordination", "geometry")
MAX_ATOMIC_NUMBER = 100
RETENTION_THRESHOLD = 0.90
TERMINAL_INFORMATION_CEILING = 0.02
PRIOR_MOMENT_TOLERANCE = 0.02
PRIOR_CORRELATION_CEILING = 0.02


def artifact_manifests(root: Path) -> dict:
    """Bind a gate report to the exact descriptor artifacts it evaluated."""
    manifests = {}
    for representation in REPRESENTATIONS:
        path = root / representation / "manifest.json"
        payload = json.loads(path.read_text())
        manifests[representation] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "settings_digest": payload["settings_digest"],
            "artifact_sha256": payload["artifact_sha256"],
        }
    return manifests


def recomputed_reference(path: Path) -> dict:
    """Read the preregistered probe of the descriptor recomputed from current x_t."""
    report = json.loads(path.read_text())
    summary = {}
    for task in TASKS:
        task_report = report["results"][task]
        summary[task] = {
            "corruption_angstrom": task_report["corruption"],
            "radial_two_layer": {
                time: values["radial/two_layer"]
                for time, values in task_report["times"].items()
            },
        }
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "summary": summary,
    }


def species_matched_distance(
    query: torch.Tensor,
    candidate: torch.Tensor,
    numbers: torch.Tensor,
) -> float:
    """Minimum mean squared descriptor distance under same-element site matching."""
    total = 0.0
    count = 0
    for number in torch.unique(numbers):
        rows = torch.nonzero(numbers == number, as_tuple=False).flatten()
        costs = torch.cdist(query[rows], candidate[rows]).square().detach().cpu().numpy()
        left, right = linear_sum_assignment(costs)
        total += float(costs[left, right].sum())
        count += len(left)
    return total / max(count, 1)


def decoys(
    fractional: torch.Tensor,
    cell: torch.Tensor,
    numbers: torch.Tensor,
    count: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
    """Construct wrapped displacement, strain, swap, and radial-matched decoys."""
    generator = torch.Generator().manual_seed(seed)
    inverse = torch.linalg.inv(cell)
    cartesian = fractional @ cell
    variants, cells, errors = [], [], []
    identity = torch.eye(3, dtype=cell.dtype)
    for index in range(count):
        mode = index % 4
        moved = fractional.clone()
        changed_cell = cell.clone()
        if len(fractional) == 1:
            diagonal = 1.0 + 0.04 * torch.randn(3, generator=generator, dtype=cell.dtype)
            changed_cell = torch.diag(diagonal) @ cell
        elif mode == 0:
            scale = 0.08 + 0.32 * float(torch.rand((), generator=generator))
            displacement = torch.randn(cartesian.shape, generator=generator, dtype=cell.dtype) * scale
            moved = fractional + displacement @ inverse
        elif mode == 1:
            raw = torch.randn((3, 3), generator=generator, dtype=cell.dtype)
            strain = 0.04 * (raw + raw.T) / 2.0
            changed_cell = (identity + strain) @ cell
        elif mode == 2:
            first = index % len(fractional)
            alternatives = torch.nonzero(numbers != numbers[first], as_tuple=False).flatten()
            if len(alternatives):
                second = int(alternatives[index % len(alternatives)])
                moved[[first, second]] = moved[[second, first]]
            else:
                second = (first + 1) % len(fractional)
                displacement = torch.randn((3,), generator=generator, dtype=cell.dtype)
                displacement = 0.20 * displacement / displacement.norm().clamp(min=1.0e-8)
                moved[first] += displacement @ inverse
                moved[second] -= displacement @ inverse
        else:
            centre = cartesian.mean(dim=0, keepdim=True)
            relative = cartesian - centre
            direction = torch.randn(relative.shape, generator=generator, dtype=cell.dtype)
            radial = (direction * relative).sum(dim=-1, keepdim=True)
            tangent = direction - radial * relative / relative.square().sum(dim=-1, keepdim=True).clamp(min=1.0e-8)
            tangent = 0.20 * tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
            moved = fractional + tangent @ inverse
        moved = moved.remainder(1.0)
        variants.append(moved)
        cells.append(changed_cell)
        periodic_delta = (moved - fractional) - torch.round(moved - fractional)
        position_error = (periodic_delta @ cell).square().mean()
        cell_error = (changed_cell - cell).square().mean()
        errors.append(float(torch.sqrt(position_error + cell_error)))
    return variants, cells, errors


def retrieval_gate(
    data_dir: Path,
    root: Path,
    structures: int,
    negatives: int,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Evaluate all nested representations on identical same-composition decoys."""
    dataset = load_split(str(data_dir / "val.lmdb"))
    chosen = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
        : min(structures, len(dataset))
    ].tolist()
    transforms = {
        representation: DescriptorTransform.load(root / representation / "transform.pkl")
        for representation in REPRESENTATIONS
    }
    distances = {representation: [] for representation in REPRESENTATIONS}
    structural_errors: list[float] = []
    per_structure = {representation: [] for representation in REPRESENTATIONS}
    for structure_index in chosen:
        batch = collate(dataset, [structure_index])
        frac = batch.pos.double()
        cell = batch.cell[0].double()
        numbers = batch.species.long()
        variants, variant_cells, errors = decoys(
            frac, cell, numbers, negatives, seed + 1009 * structure_index
        )
        all_fractional = torch.cat([frac, *variants]).to(device=device, dtype=torch.float32)
        all_cells = torch.stack([cell, *variant_cells]).to(device=device, dtype=torch.float32)
        atom_counts = torch.full(
            (negatives + 1,), len(frac), dtype=torch.long, device=device
        )
        with torch.no_grad():
            raw = clean_radial_descriptor(all_fractional, all_cells, atom_counts)
        raw = raw.reshape(negatives + 1, len(frac), -1)
        structural_errors.extend(errors)
        for representation, transform in transforms.items():
            encoded = transform.transform_tensor(raw.reshape(-1, raw.shape[-1])).reshape(
                negatives + 1, len(frac), -1
            )
            query = encoded[0]
            candidate_distances = [
                species_matched_distance(query, encoded[index], numbers.to(device))
                for index in range(negatives + 1)
            ]
            distances[representation].extend(candidate_distances[1:])
            false_collisions = sum(value < 1.0e-6 for value in candidate_distances[1:])
            order = np.argsort(np.asarray(candidate_distances), kind="stable")
            rank = int(np.flatnonzero(order == 0)[0]) + 1
            per_structure[representation].append(
                {
                    "top1": float(rank == 1),
                    "reciprocal_rank": 1.0 / rank,
                    "false_collisions": float(false_collisions),
                }
            )

    report = {}
    for representation in REPRESENTATIONS:
        entries = per_structure[representation]
        correlation = spearmanr(distances[representation], structural_errors).statistic
        report[representation] = {
            "structures": len(entries),
            "negatives_per_structure": negatives,
            "top1": float(np.mean([entry["top1"] for entry in entries])),
            "mrr": float(np.mean([entry["reciprocal_rank"] for entry in entries])),
            "false_collisions": int(sum(entry["false_collisions"] for entry in entries)),
            "structural_error_spearman": float(correlation),
        }
    full = report["radial17"]
    for representation in ("cn-rdf4", "cn-rdf8"):
        report[representation]["top1_retention"] = (
            report[representation]["top1"] / full["top1"] if full["top1"] else 0.0
        )
        report[representation]["mrr_retention"] = (
            report[representation]["mrr"] / full["mrr"] if full["mrr"] else 0.0
        )
    return report


def invariant_audit(root: Path, device: torch.device) -> dict:
    """Numerically certify invariance, continuity, finiteness, and train scaling."""
    generator = torch.Generator().manual_seed(71)
    fractional = torch.rand((6, 3), generator=generator, dtype=torch.float64)
    cell = torch.tensor(
        [[6.0, 0.0, 0.0], [2.6, 5.4, 0.0], [1.9, 1.4, 5.1]],
        dtype=torch.float64,
    )
    atoms = torch.tensor([len(fractional)], dtype=torch.long)
    transforms = {
        representation: DescriptorTransform.load(root / representation / "transform.pkl")
        for representation in REPRESENTATIONS
    }

    def raw_descriptor(
        frac: torch.Tensor,
        lattice: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return clean_radial_descriptor(
                frac.to(device=device, dtype=torch.float32),
                lattice.to(device=device, dtype=torch.float32),
                counts.to(device),
            )

    reference_raw = raw_descriptor(fractional, cell.unsqueeze(0), atoms)
    translated_raw = raw_descriptor(
        fractional + torch.tensor([0.37, -0.91, 1.2], dtype=torch.float64),
        cell.unsqueeze(0),
        atoms,
    )
    permutation = torch.randperm(len(fractional), generator=generator)
    permuted_raw = raw_descriptor(fractional[permutation], cell.unsqueeze(0), atoms)
    matrix = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated_raw = raw_descriptor(fractional, (cell @ rotation).unsqueeze(0), atoms)
    basis = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rebased_raw = raw_descriptor(
        fractional @ torch.linalg.inv(basis),
        (basis @ cell).unsqueeze(0),
        atoms,
    )
    shifts = torch.stack(
        torch.meshgrid(
            torch.arange(2, dtype=torch.float64),
            torch.arange(2, dtype=torch.float64),
            torch.arange(2, dtype=torch.float64),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    tiled_fractional = torch.cat([(fractional + shift) / 2.0 for shift in shifts], dim=0)
    tiled_raw = raw_descriptor(
        tiled_fractional,
        (2.0 * cell).unsqueeze(0),
        torch.tensor([len(tiled_fractional)]),
    )
    displacement = torch.randn(fractional.shape, generator=generator, dtype=torch.float64)
    displacement *= 1.0e-5 / displacement.norm(dim=-1, keepdim=True).clamp(min=1.0e-12)
    nearby_raw = raw_descriptor(
        fractional + displacement @ torch.linalg.inv(cell),
        cell.unsqueeze(0),
        atoms,
    )

    report = {}
    for representation, transform in transforms.items():
        reference = transform.transform_tensor(reference_raw).cpu()
        translated = transform.transform_tensor(translated_raw).cpu()
        permuted = transform.transform_tensor(permuted_raw).cpu()
        rotated = transform.transform_tensor(rotated_raw).cpu()
        rebased = transform.transform_tensor(rebased_raw).cpu()
        tiled = transform.transform_tensor(tiled_raw).cpu()
        nearby = transform.transform_tensor(nearby_raw).cpu()
        table = GeometryTable.load(
            root / representation / GEOMETRY_TABLE_NAME.format(split="train")
        )
        standard_deviation = table.values.std(axis=0)
        report[representation] = {
            "translation_max_abs": float((reference - translated).abs().max()),
            "rotation_max_abs": float((reference - rotated).abs().max()),
            "permutation_max_abs": float((reference[permutation] - permuted).abs().max()),
            "unit_cell_max_abs": float((reference - rebased).abs().max()),
            "supercell_max_abs": float((reference - tiled[: len(fractional)]).abs().max()),
            "continuity_max_change_for_1e-5_angstrom": float(
                (reference - nearby).abs().max()
            ),
            "finite": bool(np.isfinite(table.values).all()),
            "train_channel_mean": table.values.mean(axis=0).tolist(),
            "train_channel_std": standard_deviation.tolist(),
            "unit_variance_max_error": float(
                np.max(np.abs(standard_deviation - 1.0))
            ),
        }
    return report


def rows_for_structures(table: GeometryTable, structures: int | None) -> int:
    if structures is None:
        return len(table.numbers)
    if structures < 1 or structures > len(table):
        raise ValueError(f"Requested {structures} structures from a table of length {len(table)}.")
    return int(table.atom_offsets[structures])


def load_values(
    root: Path,
    representation: str,
    split: str,
    structures: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    table = GeometryTable.load(root / representation / GEOMETRY_TABLE_NAME.format(split=split))
    rows = rows_for_structures(table, structures)
    values = torch.from_numpy(table.values[:rows]).float()
    numbers = torch.from_numpy(table.numbers[:rows]).long()
    counts = np.diff(table.atom_offsets[: (len(table) if structures is None else structures) + 1])
    sizes = torch.from_numpy(np.repeat(counts, counts)).long()
    if len(sizes) != rows:
        raise ValueError(f"Structure sizes in {root / representation} do not align with atom rows.")
    return values, numbers, sizes


def chemistry_features(numbers: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    if numbers.numel() and (int(numbers.min()) < 1 or int(numbers.max()) > MAX_ATOMIC_NUMBER):
        raise ValueError("Atomic numbers fall outside the fixed chemistry encoding.")
    species = functional.one_hot(numbers - 1, num_classes=MAX_ATOMIC_NUMBER).float()
    return torch.cat([species, torch.log1p(sizes.float()).unsqueeze(-1)], dim=-1)


def fit_arm(
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    train_labels: Labels,
    val_labels: Labels,
    depth: str,
    settings: argparse.Namespace,
    device: torch.device,
    seeds: tuple[int, ...],
) -> dict[str, float]:
    train_mask = train_labels.target >= 0
    val_mask = val_labels.target >= 0
    train, val = standardise(train_features[train_mask], (val_features[val_mask],))
    train = train.to(device)
    val = val.to(device)
    train_target = train_labels.target[train_mask].to(device)
    val_target = val_labels.target[val_mask].to(device)
    scores = [
        train_probe(
            train,
            train_target,
            val,
            val_target,
            val_labels,
            depth,
            settings,
            seed,
        )
        for seed in seeds
    ]
    numeric = scores[0].keys()
    result = {key: float(np.mean([entry[key] for entry in scores])) for key in numeric}
    result.update({f"{key}_std": float(np.std([entry[key] for entry in scores])) for key in numeric})
    result["probe_seeds"] = list(seeds)
    return result


def terminal_state(values: torch.Tensor, seed: int) -> torch.Tensor:
    """Return the exact Gaussian descriptor state at the base endpoint."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(values.shape, generator=generator, dtype=values.dtype)


def prior_audit(root: Path, structures: int | None) -> dict:
    """Check that the sampled descriptor endpoint agrees with N(0, I)."""
    report = {}
    for index, representation in enumerate(REPRESENTATIONS):
        values, _, _ = load_values(root, representation, "train", structures)
        noise = terminal_state(values, 70_000 + index)
        clean = values.double().numpy()
        sampled = noise.double().numpy()
        correlations = []
        for channel in range(sampled.shape[1]):
            clean_channel = clean[:, channel]
            sampled_channel = sampled[:, channel]
            if np.std(clean_channel) == 0.0 or np.std(sampled_channel) == 0.0:
                correlations.append(0.0)
            else:
                correlations.append(
                    float(np.corrcoef(clean_channel, sampled_channel)[0, 1])
                )
        report[representation] = {
            "atoms": int(len(sampled)),
            "mean": sampled.mean(axis=0).tolist(),
            "std": sampled.std(axis=0).tolist(),
            "max_abs_mean": float(np.max(np.abs(sampled.mean(axis=0)))),
            "max_std_error": float(np.max(np.abs(sampled.std(axis=0) - 1.0))),
            "max_abs_clean_correlation": float(np.max(np.abs(correlations))),
        }
    return report


def run_task(
    task: str,
    root: Path,
    settings: argparse.Namespace,
    device: torch.device,
    seeds: tuple[int, ...],
) -> dict:
    train_labels = load_labels(task, "train", settings.train_structures)
    val_labels = load_labels(task, "val", settings.val_structures)

    loaded = {
        representation: {
            "train": load_values(root, representation, "train", settings.train_structures),
            "val": load_values(root, representation, "val", settings.val_structures),
        }
        for representation in REPRESENTATIONS
    }
    reference = loaded["radial17"]
    train_chemistry = chemistry_features(reference["train"][1], reference["train"][2])
    val_chemistry = chemistry_features(reference["val"][1], reference["val"][2])
    if len(train_labels.target) != len(train_chemistry) or len(val_labels.target) != len(val_chemistry):
        raise ValueError(f"The {task} labels do not align with the descriptor artifacts.")

    result: dict[str, dict] = {
        "floor": {},
        "representations": {},
        "terminal": {},
        "shuffled": {},
        "diffused": {},
    }
    for depth in DEPTHS:
        result["floor"][depth] = fit_arm(
            train_chemistry,
            val_chemistry,
            train_labels,
            val_labels,
            depth,
            settings,
            device,
            seeds,
        )

    for representation in REPRESENTATIONS:
        train_values, train_numbers, train_sizes = loaded[representation]["train"]
        val_values, val_numbers, val_sizes = loaded[representation]["val"]
        if not torch.equal(train_numbers, reference["train"][1]) or not torch.equal(
            val_numbers, reference["val"][1]
        ):
            raise ValueError(f"{representation} artifacts use a different atom ordering.")
        train_real = torch.cat([train_values, chemistry_features(train_numbers, train_sizes)], dim=-1)
        val_real = torch.cat([val_values, chemistry_features(val_numbers, val_sizes)], dim=-1)
        train_noise = torch.cat(
            [terminal_state(train_values, 10_000), chemistry_features(train_numbers, train_sizes)],
            dim=-1,
        )
        val_noise = torch.cat(
            [terminal_state(val_values, 20_000), chemistry_features(val_numbers, val_sizes)],
            dim=-1,
        )

        train_permutation = within_element_permutation(train_numbers.numpy(), seed=30_000)
        val_permutation = within_element_permutation(val_numbers.numpy(), seed=40_000)
        train_shuffled = torch.cat(
            [train_values[train_permutation], chemistry_features(train_numbers, train_sizes)], dim=-1
        )
        val_shuffled = torch.cat(
            [val_values[val_permutation], chemistry_features(val_numbers, val_sizes)], dim=-1
        )
        result["representations"][representation] = {}
        result["terminal"][representation] = {}
        result["shuffled"][representation] = {}
        result["diffused"][representation] = {}
        for depth in DEPTHS:
            floor = result["floor"][depth]["cross_entropy_bits"]
            for destination, train_features, val_features in (
                ("representations", train_real, val_real),
                ("terminal", train_noise, val_noise),
                ("shuffled", train_shuffled, val_shuffled),
            ):
                scores = fit_arm(
                    train_features,
                    val_features,
                    train_labels,
                    val_labels,
                    depth,
                    settings,
                    device,
                    seeds,
                )
                scores["information_bits"] = floor - scores["cross_entropy_bits"]
                result[destination][representation][depth] = scores
        for time_value in settings.diffusion_times:
            train_diffused = (
                time_value * train_values
                + (1.0 - time_value) * terminal_state(train_values, 50_000)
            )
            val_diffused = (
                time_value * val_values
                + (1.0 - time_value) * terminal_state(val_values, 60_000)
            )
            train_features = torch.cat(
                [train_diffused, chemistry_features(train_numbers, train_sizes)], dim=-1
            )
            val_features = torch.cat(
                [val_diffused, chemistry_features(val_numbers, val_sizes)], dim=-1
            )
            result["diffused"][representation][str(time_value)] = {}
            for depth in DEPTHS:
                scores = fit_arm(
                    train_features,
                    val_features,
                    train_labels,
                    val_labels,
                    depth,
                    settings,
                    device,
                    seeds,
                )
                scores["information_bits"] = (
                    result["floor"][depth]["cross_entropy_bits"]
                    - scores["cross_entropy_bits"]
                )
                result["diffused"][representation][str(time_value)][depth] = scores
    return result


def verdict(results: dict) -> dict:
    information = {
        representation: {
            task: {
                depth: results["tasks"][task]["representations"][representation][depth][
                    "information_bits"
                ]
                for depth in DEPTHS
            }
            for task in TASKS
        }
        for representation in REPRESENTATIONS
    }
    full = {
        task: information["radial17"][task]["two_layer"]
        for task in TASKS
    }
    retention = {
        representation: {
            task: (
                information[representation][task]["two_layer"] / full[task]
                if full[task] > 0.0
                else 0.0
            )
            for task in TASKS
        }
        for representation in REPRESENTATIONS
    }
    retrieval_retention = {
        representation: {
            "top1": results["retrieval"][representation].get("top1_retention", 1.0),
            "mrr": results["retrieval"][representation].get("mrr_retention", 1.0),
        }
        for representation in REPRESENTATIONS
    }
    eligibility = {}
    for representation in REPRESENTATIONS:
        nonlinear_positive = all(
            information[representation][task]["two_layer"] > 0.0 for task in TASKS
        )
        linear_positive = all(
            information[representation][task]["linear"] > 0.0 for task in TASKS
        )
        information_retained = all(
            retention[representation][task] >= RETENTION_THRESHOLD for task in TASKS
        )
        retrieval_retained = all(
            value >= RETENTION_THRESHOLD
            for value in retrieval_retention[representation].values()
        )
        eligibility[representation] = {
            "nonlinear_information_positive": nonlinear_positive,
            "linear_information_positive": linear_positive,
            "information_retained": information_retained,
            "retrieval_retained": retrieval_retained,
            "passed": (
                nonlinear_positive
                and linear_positive
                and information_retained
                and retrieval_retained
            ),
        }
    promoted = next(
        (
            representation
            for representation in REPRESENTATIONS
            if eligibility[representation]["passed"]
        ),
        None,
    )

    geometry_floor = results["tasks"]["geometry"]["floor"]
    shape_given_coordination = {}
    for representation in REPRESENTATIONS:
        shape_given_coordination[representation] = {}
        for depth in DEPTHS:
            absolute = results["tasks"]["geometry"]["representations"][representation][
                depth
            ]["shape_given_coordination"]
            floor = geometry_floor[depth]["shape_given_coordination"]
            shape_given_coordination[representation][depth] = {
                "accuracy": absolute,
                "chemistry_floor_accuracy": floor,
                "gain_points": 100.0 * (absolute - floor),
            }

    terminal = {
        representation: max(
            abs(
                results["tasks"][task]["terminal"][representation]["two_layer"][
                    "information_bits"
                ]
            )
            for task in TASKS
        )
        for representation in REPRESENTATIONS
    }
    shuffled = {
        representation: max(
            abs(
                results["tasks"][task]["shuffled"][representation]["two_layer"][
                    "information_bits"
                ]
            )
            for task in TASKS
        )
        for representation in REPRESENTATIONS
    }
    terminal_passed = (
        promoted is not None
        and terminal[promoted] <= TERMINAL_INFORMATION_CEILING
    )
    shuffled_passed = (
        promoted is not None
        and shuffled[promoted] <= TERMINAL_INFORMATION_CEILING
    )
    retrieval_passed = (
        promoted is not None and eligibility[promoted]["retrieval_retained"]
    )
    audit = results["invariant_audit"].get(promoted, {}) if promoted else {}
    audit_passed = (
        bool(audit)
        and audit["finite"]
        and max(
            audit["translation_max_abs"],
            audit["rotation_max_abs"],
            audit["permutation_max_abs"],
            audit["unit_cell_max_abs"],
            audit["supercell_max_abs"],
        )
        <= 1.0e-4
        and audit["continuity_max_change_for_1e-5_angstrom"] <= 1.0e-2
        and audit["unit_variance_max_error"] <= 0.15
    )
    prior = results["prior_audit"].get(promoted, {}) if promoted else {}
    prior_passed = (
        bool(prior)
        and prior["max_abs_mean"] <= PRIOR_MOMENT_TOLERANCE
        and prior["max_std_error"] <= PRIOR_MOMENT_TOLERANCE
        and prior["max_abs_clean_correlation"] <= PRIOR_CORRELATION_CEILING
    )
    return {
        "promoted": promoted,
        "eligibility": eligibility,
        "information_bits": information,
        "retention": retention,
        "retrieval_retention": retrieval_retention,
        "shape_given_coordination": shape_given_coordination,
        "retrieval_passed": retrieval_passed,
        "invariant_audit_passed": audit_passed,
        "terminal_information_bits_absolute_max": terminal,
        "terminal_passed": terminal_passed,
        "shuffled_information_bits_absolute_max": shuffled,
        "shuffled_passed": shuffled_passed,
        "prior_passed": prior_passed,
        "passed": bool(
            promoted is not None
            and eligibility[promoted]["passed"]
            and terminal_passed
            and shuffled_passed
            and prior_passed
            and audit_passed
        ),
        "thresholds": {
            "compression_information_retention": RETENTION_THRESHOLD,
            "retrieval_retention": RETENTION_THRESHOLD,
            "terminal_information_bits": TERMINAL_INFORMATION_CEILING,
            "shuffled_information_bits": TERMINAL_INFORMATION_CEILING,
            "prior_moment_tolerance": PRIOR_MOMENT_TOLERANCE,
            "prior_clean_correlation": PRIOR_CORRELATION_CEILING,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="joint_geometry/artifacts/mpts_52",
        help="directory containing cn-rdf4, cn-rdf8 and radial17 artifact directories",
    )
    parser.add_argument("--out", default="joint_geometry/reports/DESCRIPTOR-GATE.json")
    parser.add_argument("--train-structures", type=int, default=4000)
    parser.add_argument("--val-structures", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-sample", type=int, default=100_000)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--diffusion-times", type=float, nargs="+", default=[0.5, 0.95])
    parser.add_argument("--data-dir", default="omg/data/mpts_52")
    parser.add_argument("--retrieval-structures", type=int, default=128)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument(
        "--recomputed-report",
        default="direct_geometry/reports/DG1-DG2-PROBES.json",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    root = Path(arguments.artifact_root)
    results = {
        "gate": "JG0",
        "arguments": vars(arguments),
        "artifacts": artifact_manifests(root),
        "tasks": {
            task: run_task(task, root, arguments, device, tuple(arguments.probe_seeds))
            for task in TASKS
        },
        "retrieval": retrieval_gate(
            Path(arguments.data_dir),
            root,
            arguments.retrieval_structures,
            arguments.hard_negatives,
            device,
        ),
        "invariant_audit": invariant_audit(root, device),
        "prior_audit": prior_audit(root, arguments.train_structures),
        "recomputed_from_current_structure": recomputed_reference(
            Path(arguments.recomputed_report)
        ),
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    results["verdict"] = verdict(results)
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["verdict"], indent=2))
    print(f"Wrote {destination}.")
    if not results["verdict"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
