"""
Measure whether candidate CSPNet backbones can recover clean CN-RDF state.

The probe uses the exact message graph implementation later used by the
generative arms, but a smaller fixed trunk and a per-site regression head.  It
compares held-out MSE with an element-and-size chemistry floor at fixed
physical corruption levels.
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
from sklearn.linear_model import Ridge

from direct_geometry.batches import collate, interpolate, load_split
from joint_geometry.data import GEOMETRY_TABLE_NAME, GeometryTable
from joint_geometry.encoder import JointGeometryCSPNet
from omg.model.model_utils import SinusoidalTimeEmbeddings


GRAPHS = ("fc", "fc_distance", "periodic_distance")


def table_rows(table: GeometryTable, indices: list[int]) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(int(table.atom_offsets[index]), int(table.atom_offsets[index + 1]))
            for index in indices
        ]
    )


def chemistry_features(table: GeometryTable, structures: int) -> np.ndarray:
    stop = min(structures, len(table))
    rows = int(table.atom_offsets[stop])
    numbers = table.numbers[:rows]
    counts = np.diff(table.atom_offsets[: stop + 1])
    one_hot = np.eye(100, dtype=np.float32)[numbers - 1]
    sizes = np.log1p(np.repeat(counts, counts).astype(np.float32))[:, None]
    return np.concatenate([one_hot, sizes], axis=1)


def make_model(graph: str, dimension: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    latent = 64
    model = JointGeometryCSPNet(
        geometry_dimension=dimension,
        geometry_input=False,
        feature_mode="none",
        message_graph=graph,
        hidden_dim=128,
        latent_dim=latent,
        num_layers=2,
        num_freqs=16,
        edge_basis=16,
        pred_type=False,
    ).to(device)
    return model, SinusoidalTimeEmbeddings(latent).to(device)


def corrupted_batch(
    dataset,
    indices: list[int],
    time_value: float,
    seed: int,
    dimension: int,
    device: torch.device,
):
    batch = collate(dataset, indices)
    frac, cell = interpolate(batch, time_value, seed)
    batch.pos = frac.float()
    batch.cell = cell.float()
    batch.geometry = torch.zeros((batch.pos.shape[0], dimension), dtype=batch.pos.dtype)
    return batch.to(device)


def evaluate(
    model,
    time_embedder,
    dataset,
    table: GeometryTable,
    indices: list[int],
    time_value: float,
    batch_structures: int,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    squared, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_structures):
            chosen = indices[start : start + batch_structures]
            batch = corrupted_batch(
                dataset,
                chosen,
                time_value,
                seed + start,
                table.dimension,
                device,
            )
            times = torch.full(
                (len(chosen),), time_value, dtype=batch.pos.dtype, device=device
            )
            prediction = model(batch, time_embedder(times)).geometry_b
            target = torch.from_numpy(table.values[table_rows(table, chosen)]).to(device)
            squared += float(functional.mse_loss(prediction, target, reduction="sum"))
            count += target.numel()
    return squared / max(count, 1)


def fit_one(
    graph: str,
    time_value: float,
    seed: int,
    train_dataset,
    val_dataset,
    train_table: GeometryTable,
    val_table: GeometryTable,
    settings: argparse.Namespace,
    device: torch.device,
) -> float:
    model, time_embedder = make_model(graph, train_table.dimension, seed, device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    upper = min(settings.train_structures, len(train_dataset))
    model.train()
    for step in range(settings.steps):
        chosen = torch.randint(
            0,
            upper,
            (settings.batch_structures,),
            generator=generator,
        ).tolist()
        batch = corrupted_batch(
            train_dataset,
            chosen,
            time_value,
            seed * 1_000_003 + step,
            train_table.dimension,
            device,
        )
        times = torch.full(
            (len(chosen),), time_value, dtype=batch.pos.dtype, device=device
        )
        prediction = model(batch, time_embedder(times)).geometry_b
        target = torch.from_numpy(train_table.values[table_rows(train_table, chosen)]).to(device)
        loss = functional.mse_loss(prediction, target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    validation_indices = list(range(min(settings.val_structures, len(val_dataset))))
    return evaluate(
        model,
        time_embedder,
        val_dataset,
        val_table,
        validation_indices,
        time_value,
        settings.batch_structures,
        device,
        seed * 2_000_003,
    )


def interval(values: list[float]) -> dict[str, float]:
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "values": values,
    }


def select_backbone(clean_endpoint: dict[str, dict]) -> str | None:
    """Choose the cheapest graph whose interval overlaps the best graph."""
    best_low = max(
        clean_endpoint[graph]["chemistry_relative_r2"]["ci95_low"]
        for graph in GRAPHS
    )
    for graph in GRAPHS:
        score = clean_endpoint[graph]["chemistry_relative_r2"]
        if score["mean"] > 0.0 and score["ci95_high"] >= best_low:
            return graph
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="omg/data/mpts_52")
    parser.add_argument(
        "--artifact-dir",
        default=None,
    )
    parser.add_argument(
        "--descriptor-report",
        default="joint_geometry/reports/DESCRIPTOR-GATE.json",
        help="passing descriptor gate used to resolve --artifact-dir when omitted",
    )
    parser.add_argument("--out", default="joint_geometry/reports/BACKBONE-GATE.json")
    parser.add_argument("--times", type=float, nargs="+", default=[0.9, 0.95, 1.0])
    parser.add_argument("--train-structures", type=int, default=4000)
    parser.add_argument("--val-structures", type=int, default=1000)
    parser.add_argument("--batch-structures", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    descriptor_report = Path(arguments.descriptor_report)
    if arguments.artifact_dir is None:
        descriptor = json.loads(descriptor_report.read_text())
        if not descriptor["verdict"]["passed"]:
            raise ValueError(f"The descriptor gate at {descriptor_report} did not pass.")
        promoted = descriptor["verdict"]["promoted"]
        root = Path(descriptor["arguments"]["artifact_root"]) / promoted
    else:
        root = Path(arguments.artifact_dir)
    train_table = GeometryTable.load(root / GEOMETRY_TABLE_NAME.format(split="train"))
    val_table = GeometryTable.load(root / GEOMETRY_TABLE_NAME.format(split="val"))
    train_dataset = load_split(str(Path(arguments.data_dir) / "train.lmdb"))
    val_dataset = load_split(str(Path(arguments.data_dir) / "val.lmdb"))

    train_rows = int(train_table.atom_offsets[min(arguments.train_structures, len(train_table))])
    floor = Ridge(alpha=1.0).fit(
        chemistry_features(train_table, arguments.train_structures),
        train_table.values[:train_rows],
    )
    val_rows = table_rows(
        val_table, list(range(min(arguments.val_structures, len(val_dataset))))
    )
    floor_prediction = floor.predict(chemistry_features(val_table, arguments.val_structures))
    floor_mse = float(np.mean((floor_prediction - val_table.values[val_rows]) ** 2))
    report = {
        "floor_mse": floor_mse,
        "times": {},
        "arguments": vars(arguments),
        "artifact_manifest": {
            "path": str(root / "manifest.json"),
            "sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        },
        "descriptor_gate": (
            {
                "path": str(descriptor_report),
                "sha256": hashlib.sha256(descriptor_report.read_bytes()).hexdigest(),
            }
            if descriptor_report.is_file()
            else None
        ),
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    for time_value in arguments.times:
        time_report = {}
        for graph in GRAPHS:
            values = [
                fit_one(
                    graph,
                    time_value,
                    seed,
                    train_dataset,
                    val_dataset,
                    train_table,
                    val_table,
                    arguments,
                    device,
                )
                for seed in arguments.seeds
            ]
            mse = interval(values)
            r2_values = [1.0 - value / floor_mse for value in values]
            time_report[graph] = {"mse": mse, "chemistry_relative_r2": interval(r2_values)}
        report["times"][str(time_value)] = time_report

    clean = report["times"][str(max(arguments.times))]
    selected = select_backbone(clean)
    report["verdict"] = {
        "selected_backbone": selected,
        "passed": selected is not None,
        "selection_rule": (
            "cheapest graph with positive held-out gain whose 95% interval overlaps "
            "the best graph's interval at the clean endpoint"
        ),
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    print(f"Wrote {destination}.")
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
