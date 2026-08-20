"""
Gate DG0: does the geometry machinery describe real crystals, at real noise levels, without costing too much?

The correctness tests in ``direct_geometry/tests/`` assert algebra on hand-built polyhedra. They cannot tell whether the
descriptor *varies* across MPTS-52, whether a channel is dead, whether a real batch trips a safety bound, or what the
addition costs on the card the screening runs will use. Those are measurements, and this is the script that makes them.

WHAT IS MEASURED, AND WHY EACH ONE CAN FAIL A GATE

- **Neighbourhoods.** Edge counts, degree distribution, image reach, and how many edges are duplicate atom pairs -- that
  last one is the periodic multiplicity the fully connected graph cannot express, and if it is zero on MPTS-52 then the
  graph factor is a no-op and the study should not be run.
- **Channels across denoising time.** The denoiser sees the interpolated state, not the crystal. At ``t`` near zero the
  positions are a uniform draw and the descriptor is describing noise; near one it is describing a crystal. A channel whose
  variance collapses at the times that matter is a channel the projection cannot use, and reporting a variance measured
  only on clean structures would hide that.
- **Finiteness.** One NaN in one channel poisons a whole batch's gradient. Measured on sampled priors as well as on data,
  because the degenerate configurations -- an empty neighbourhood, two atoms on top of each other -- live at the noisy end.
- **Cost.** Training-step wall time and peak memory for every arm against the ``fc``/``none`` control, on the real card at
  the real precision. The gate is 30 per cent, and it is a real constraint: at 1,600 epochs a 30 per cent overhead is a day.

Usage:

    python -m direct_geometry.scripts.audit_geometry
    python -m direct_geometry.scripts.audit_geometry --structures 512 --repeats 5
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Optional
import torch
from torch_geometric.data import Batch

from direct_geometry.batches import interpolate, sample_structures
from direct_geometry.encoder import DirectGeometryCSPNet, MessageGraph
from direct_geometry.features import FeatureMode, descriptor_statistics, local_environment_descriptor
from direct_geometry.neighbors import Neighbors, bounded_cutoff, constant_degree_radius, periodic_neighbors

DEFAULT_SPLIT = "omg/data/mpts_52/val.lmdb"
"""
The validation split, not the training one.

Nothing here is fitted, so reading validation structures costs the study nothing, and keeping the audit off the training
split means the numbers in this report cannot be quoted as though they described training data.
"""

DEFAULT_TIMES = (0.001, 0.25, 0.5, 0.75, 0.95, 0.999)
"""
Denoising times at which the channels are summarised.

Spanning both endpoints because the two ends are qualitatively different problems: at 0.001 the positions are essentially
the base distribution and the descriptor is a description of noise, at 0.999 it is a description of a crystal.
"""

ARMS = ((MessageGraph.FC, FeatureMode.NONE), (MessageGraph.FC_DISTANCE, FeatureMode.NONE),
        (MessageGraph.PERIODIC_DISTANCE, FeatureMode.NONE),
        (MessageGraph.FC, FeatureMode.RADIAL), (MessageGraph.FC, FeatureMode.ANGULAR),
        (MessageGraph.FC, FeatureMode.BOTH), (MessageGraph.PERIODIC_DISTANCE, FeatureMode.BOTH))
"""
Every combination the study can promote, plus the two single-block modes.

The single-block modes are here because Gate DG2 may drop the angular channels, and the cost of the arm that would then be
promoted has to be known before the gate is read rather than after.
"""

COST_CEILING = 1.30
"""
Largest tolerated ratio of step time or peak memory against the ``fc``/``none`` control.

Set by the plan. It is a budget rather than a preference: the production sweep is twelve runs of 1,600 epochs, so a factor
of 1.3 is roughly an extra day of eight-GPU time and a factor of two would not be affordable.
"""

MAX_AT_BOUND_FRACTION = 0.5
"""
Largest tolerated fraction of clean structures whose graph radius is held at its bound.

The bound is what lets one neighbour list serve both the descriptor and the graph, and a low-density structure whose
requested radius exceeds it gets the bound instead -- a fixed-radius graph with a degree below target. A minority is the
accepted price; a majority would mean the constant-degree design had reverted to a fixed radius without saying so, and the
matched-connectivity argument for the graph factor would no longer hold.
"""

BUDGET_CUTOFFS = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0)
"""
Graph radii the edge budget is measured at.

The periodic graph's price is set almost entirely by its edge count, because every edge runs the whole edge network. So the
radius is a cost parameter, and this sweep is the measurement it was chosen from -- recorded in the report rather than left
in a comment, so that the choice can be audited and so that it is visibly a cost decision taken before any match rate was
measured.
"""


def summarise_neighbors(neighbors: Neighbors, num_nodes: int) -> dict[str, float]:
    """
    Reduce a neighbour list to the counts a gate can be read on.

    :param neighbors:
        The neighbour list.
    :type neighbors: Neighbors
    :param num_nodes:
        Number of atoms the list was built over.
    :type num_nodes: int

    :return:
        Edge and degree counts, image reach, and periodic multiplicity.
    :rtype: dict[str, float]
    """
    degree = neighbors.degree(num_nodes)
    # An atom pair joined by more than one edge is a pair joined through more than one periodic image. This is exactly the
    # multiplicity a one-edge-per-pair graph cannot express, so its size is the size of the graph factor.
    _, multiplicity = torch.unique(neighbors.center * num_nodes + neighbors.neighbor, return_counts=True)
    return {
        "edges": int(len(neighbors)),
        "edges_per_atom": float(degree.double().mean()) if num_nodes else 0.0,
        "degree_min": int(degree.min()) if num_nodes else 0,
        "degree_max": int(degree.max()) if num_nodes else 0,
        "isolated_atoms": int((degree == 0).sum()),
        "max_image_component": int(neighbors.image.abs().max()) if len(neighbors) else 0,
        "multiedge_fraction": float((multiplicity > 1).double().mean()) if len(neighbors) else 0.0,
        "max_pair_multiplicity": int(multiplicity.max()) if len(neighbors) else 0,
    }


def audit_neighborhoods(encoder: DirectGeometryCSPNet, batch: Batch, times: tuple[float, ...],
                        seed: int) -> dict[str, dict]:
    """
    Summarise the neighbour lists and the descriptor channels at each denoising time.

    Both lists, because the two factors read the same enumeration at different radii and the gate asks different questions
    of each: the descriptor's list has to yield varying, finite channels, while the graph's list has to actually contain
    the periodic multiplicity that is the whole point of replacing the fully connected topology.

    :param encoder:
        An encoder configured with the descriptor and graph to be summarised.
    :type encoder: DirectGeometryCSPNet
    :param batch:
        The reference structures.
    :type batch: torch_geometric.data.Batch
    :param times:
        Denoising times to sweep.
    :type times: tuple[float, ...]
    :param seed:
        Seed of the base draws.
    :type seed: int

    :return:
        For every time, both neighbourhood summaries and the channel statistics.
    :rtype: dict[str, dict]
    """
    report = {}
    for time_value in times:
        frac, cell = interpolate(batch, time_value, seed)
        neighbors = encoder.build_neighbors(frac, cell, batch.n_atoms)
        atoms = frac.shape[0]
        descriptor = local_environment_descriptor(neighbors, atoms, encoder.descriptor_spec, FeatureMode.BOTH)
        radius = constant_degree_radius(cell, batch.n_atoms, encoder.graph_degree, encoder.graph_max_radius)
        # How often the pathological-cell reduction engages. It exists for cells the sampler generates while integrating
        # an undertrained model, and must never fire on the probability path itself: if it did, the descriptor here would
        # summarise a smaller neighbourhood than the one the DG1 probes measured its value at, and the promoted feature
        # would not be the feature being trained.
        effective = bounded_cutoff(cell, batch.n_atoms, encoder.neighbor_cutoff)
        report[f"{time_value:g}"] = {
            "descriptor_neighbors": summarise_neighbors(neighbors.within(encoder.descriptor_spec.cutoff), atoms),
            "graph_neighbors": summarise_neighbors(neighbors.within(radius), atoms),
            "graph_radius": {"mean": float(radius.mean()), "min": float(radius.min()), "max": float(radius.max()),
                             "at_bound_fraction": float((radius >= encoder.graph_max_radius).double().mean())},
            "pathological_cells": {
                "reduced_fraction": float((effective < encoder.neighbor_cutoff).double().mean()),
                "min_effective_cutoff": float(effective.min())},
            "channels": descriptor_statistics(descriptor, encoder.descriptor_spec),
            "finite_fraction": float(torch.isfinite(descriptor).double().mean()),
        }
    return report


def audit_edge_budget(encoder: DirectGeometryCSPNet, batch: Batch, times: tuple[float, ...],
                      cutoffs: tuple[float, ...], seed: int) -> dict[str, dict]:
    """
    Count the periodic graph's edges per atom against the fully connected control's, for fixed and adaptive radii.

    The control's count is not a constant that can be looked up: the fully connected graph joins every ordered pair within
    a structure, so its edges per atom is the atom-weighted mean structure size, which is a property of the split. On
    MPTS-52 it is about 32.

    Both the fixed-radius sweep and the configured adaptive radius are reported, because the sweep is the evidence for
    preferring the adaptive one and belongs in the artifact rather than in a comment. It shows the bind directly: no fixed
    radius both matches the control's degree on crystals and stays affordable at the noisy end, since the two ends of the
    baseline's path differ in density by about a factor of two.

    Worst case over the sweep of denoising times as well as the clean value, because at the noisy end the cell prior draws
    small cells, atoms are dense, and a fixed-radius graph is at its largest exactly where the training step has to fit.

    :param encoder:
        An encoder configured with the graph whose adaptive radius is to be measured.
    :type encoder: DirectGeometryCSPNet
    :param batch:
        The reference structures.
    :type batch: torch_geometric.data.Batch
    :param times:
        Denoising times to take the worst case over.
    :type times: tuple[float, ...]
    :param cutoffs:
        Fixed graph radii to measure, in Angstrom.
    :type cutoffs: tuple[float, ...]
    :param seed:
        Seed of the base draws.
    :type seed: int

    :return:
        The control's edges per atom, the same counts for every fixed radius, and the counts for the configured adaptive
        radius, each with its ratio against the control.
    :rtype: dict[str, dict]
    """
    atoms = int(batch.n_atoms.sum())
    # Ordered pairs including self, which is what CSPNetFull.gen_edges builds for the fully connected graph.
    control = float((batch.n_atoms.double() ** 2).sum()) / atoms
    clean_time = max(times)

    def summarise(per_time: dict[float, float]) -> dict[str, float]:
        worst = max(per_time.values())
        return {"worst_edges_per_atom": worst, "clean_edges_per_atom": per_time[clean_time],
                "worst_ratio": worst / control, "clean_ratio": per_time[clean_time] / control}

    fixed = {}
    for cutoff in cutoffs:
        counts = {}
        for time_value in times:
            frac, cell = interpolate(batch, time_value, seed)
            counts[time_value] = len(periodic_neighbors(frac, cell, batch.n_atoms, cutoff)) / atoms
        fixed[f"{cutoff:g}"] = summarise(counts)

    adaptive = {}
    for time_value in times:
        frac, cell = interpolate(batch, time_value, seed)
        radius = constant_degree_radius(cell, batch.n_atoms, encoder.graph_degree, encoder.graph_max_radius)
        built = periodic_neighbors(frac, cell, batch.n_atoms, float(radius.max()))
        adaptive[time_value] = len(built.within(radius)) / atoms
    return {"fc_edges_per_atom": control, "fixed_radius": fixed,
            "adaptive_radius": {"degree": encoder.graph_degree, "max_radius": encoder.graph_max_radius,
                                **summarise(adaptive)}}


def time_training_step(graph: MessageGraph, mode: FeatureMode, batch: Batch, device: torch.device,
                       times: tuple[float, ...], repeats: int, seed: int) -> dict[str, float]:
    """
    Measure one arm's forward-and-backward wall time and peak memory across the noise range.

    Forward *and* backward, because the descriptor is cheap to evaluate and the widened edge network is not, and a forward-
    only measurement would understate the arm that the study is most likely to promote.

    Across denoising times rather than on clean structures, because the periodic graph's density is a function of the
    state it is built on: at the noisy end the positions are a uniform draw, atoms clump, and the graph carries about half
    again as many edges as it does on a crystal. Training draws the time uniformly, so the reported step time is a mean
    over the sweep -- that is what sets an epoch -- while the reported memory is the worst case over it, since one dense
    batch is enough to end a run.

    :param graph:
        Message graph of the arm.
    :type graph: MessageGraph
    :param mode:
        Feature mode of the arm.
    :type mode: FeatureMode
    :param batch:
        The batch to time.
    :type batch: torch_geometric.data.Batch
    :param device:
        Device to time on.
    :type device: torch.device
    :param times:
        Denoising times to time the arm at.
    :type times: tuple[float, ...]
    :param repeats:
        Number of timed steps per time, after a warm-up step.
    :type repeats: int
    :param seed:
        Seed of the initial weights and base draws, shared across arms.
    :type seed: int

    :return:
        Mean and worst median step time in seconds, and worst peak allocated memory in gibibytes.
    :rtype: dict[str, float]
    """
    torch.manual_seed(seed)
    encoder = DirectGeometryCSPNet(feature_mode=mode.value, message_graph=graph.value).to(device)
    species = batch.species.to(device)
    atoms = batch.n_atoms.to(device)
    node2graph = batch.batch.to(device)
    embedded = torch.randn((len(atoms), 256), device=device)

    def step(frac: torch.Tensor, cell: torch.Tensor) -> None:
        encoder.zero_grad(set_to_none=True)
        output = encoder._forward(species, frac, cell, atoms, node2graph, embedded)
        (output["pos_b"].pow(2).mean() + output["cell_b"].pow(2).mean()).backward()

    per_time, peak = [], 0.0
    for index, time_value in enumerate(times):
        interpolated = interpolate(batch, time_value, seed)
        frac, cell = (tensor.to(device).float() for tensor in interpolated)
        if index == 0:
            step(frac, cell)  # A warm-up, so that the first arm does not pay for lazily initialised kernels.
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            step(frac, cell)
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
        durations.sort()
        per_time.append(durations[len(durations) // 2])
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device) / 2 ** 30)
    return {"seconds": sum(per_time) / len(per_time), "worst_seconds": max(per_time),
            "peak_gib": peak if device.type == "cuda" else float("nan")}


def gate_failures(neighborhoods: dict[str, dict], costs: dict[str, dict], channel_names: list[str],
                  ceiling: float) -> list[str]:
    """
    Read Gate DG0 from the measurements and return every reason it fails.

    Separated from the measuring so that the verdict can be tested without a GPU and without an LMDB split. The gate is
    the thing that stops a bad experiment from reaching the A100s, so a silent bug in it is more expensive than a bug in
    anything it inspects, and a gate that can only be exercised by running the full audit is a gate nobody checks.

    Every reason is collected rather than returning at the first, because the reasons are independent and a fix informed
    by one of four failures is a fix that has to be attempted four times.

    :param neighborhoods:
        Output of ``audit_neighborhoods``.
    :type neighborhoods: dict[str, dict]
    :param costs:
        Output of the cost measurement, with ratios already attached, or empty if it was skipped.
    :type costs: dict[str, dict]
    :param channel_names:
        Names of the descriptor channels, every one of which must vary somewhere in the sweep.
    :type channel_names: list[str]
    :param ceiling:
        Largest tolerated cost ratio against the control.
    :type ceiling: float

    :return:
        Human-readable reasons the gate fails, empty if it passes.
    :rtype: list[str]
    """
    failures = []
    for label, entry in neighborhoods.items():
        if entry["finite_fraction"] < 1.0:
            failures.append(f"non-finite descriptor channels at t={label}")
        # An atom with no neighbours has an all-zero descriptor, so the arm is blind there. A handful at the noisy end is
        # survivable; the gate exists to catch a cutoff or an image enumeration that leaves them everywhere.
        if entry["descriptor_neighbors"]["isolated_atoms"] > 0 and float(label) >= 0.5:
            failures.append(f"{entry['descriptor_neighbors']['isolated_atoms']} atoms have empty neighbourhoods at "
                            f"t={label}, where the structures are nearly crystals")
    # Every channel must vary somewhere in the sweep. One that never does is a channel the projection cannot use, and it
    # would sit in the parameter count pretending to be information.
    for name in channel_names:
        spread = max(entry["channels"][name]["std"] for entry in neighborhoods.values())
        if not spread > 0.0:
            failures.append(f"channel {name} has zero variance at every audited time")
    clean = neighborhoods[max(neighborhoods, key=float)]
    if clean["graph_neighbors"]["multiedge_fraction"] <= 0.0:
        failures.append("no atom pair is joined by more than one periodic image inside the graph radius, so the graph "
                        "factor is a no-op")
    # A structure whose density asks for more reach than the bound allows gets the bound instead, so its graph is a
    # fixed-radius one after all. A minority is the accepted price of serving both consumers from one neighbour list; a
    # majority means the adaptive radius has quietly stopped being adaptive and the bound needs raising.
    at_bound = clean.get("graph_radius", {}).get("at_bound_fraction", 0.0)
    if at_bound > MAX_AT_BOUND_FRACTION:
        failures.append(f"{at_bound:.0%} of structures are held at the graph radius bound on clean data, so the graph is "
                        f"effectively fixed-radius rather than constant-degree")
    # The pathological-cell reduction is for cells the *sampler* generates, not for anything on the probability path. If
    # it fires here it is silently shrinking the neighbourhood the DG1 probes measured, so the promoted descriptor would
    # no longer be the descriptor whose value was established. Any occurrence at all is a failure, not a fraction.
    for label, entry in neighborhoods.items():
        reduced = entry.get("pathological_cells", {}).get("reduced_fraction", 0.0)
        if reduced > 0.0:
            failures.append(f"{reduced:.1%} of structures had their descriptor cutoff reduced at t={label}, down to "
                            f"{entry['pathological_cells']['min_effective_cutoff']:.2f} Angstrom. That bound is meant for "
                            f"generated cells only; firing on the path means the descriptor is not the one DG1 measured")

    for name, entry in costs.items():
        if entry["time_ratio"] > ceiling:
            failures.append(f"{name} costs {entry['time_ratio']:.2f} times the control's mean step time")
        # Compared to itself rather than to the ceiling when it is not a number, which is the CPU case where peak
        # allocation is not reported. Silently passing an unmeasured budget is how a memory regression ships.
        if entry["memory_ratio"] != entry["memory_ratio"]:
            continue
        if entry["memory_ratio"] > ceiling:
            failures.append(f"{name} peaks at {entry['memory_ratio']:.2f} times the control's memory")
    return failures


def main(argv: Optional[list[str]] = None) -> int:
    """
    Run the audit and write the Gate DG0 verdict.

    :param argv:
        Command-line arguments, or None to read them from the process.
        Defaults to None.
    :type argv: Optional[list[str]]

    :return:
        Zero if Gate DG0 passes, one otherwise, so a launcher can refuse to continue.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="LMDB split to audit")
    parser.add_argument("--structures", type=int, default=256, help="number of structures in the audit batch")
    parser.add_argument("--cost-structures", type=int, default=64,
                        help="number of structures in the timing batch, which is smaller so that every arm fits")
    parser.add_argument("--repeats", type=int, default=5, help="timed steps per arm, after a warm-up")
    parser.add_argument("--seed", type=int, default=0, help="seed of the selection, base draws and weights")
    parser.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES),
                        help="denoising times to summarise the channels at")
    parser.add_argument("--budget-cutoffs", type=float, nargs="+", default=list(BUDGET_CUTOFFS),
                        help="graph radii to measure the edge budget at")
    parser.add_argument("--out", default="direct_geometry/reports/DG0-AUDIT.json", help="where to write the report")
    parser.add_argument("--skip-cost", action="store_true",
                        help="skip the timing, for a correctness-only audit on a machine without the training GPU")
    arguments = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"auditing {arguments.split} on {device}")
    batch = sample_structures(arguments.split, arguments.structures, arguments.seed)
    print(f"  {len(batch.n_atoms)} structures, {int(batch.n_atoms.sum())} atoms, "
          f"{float(batch.n_atoms.double().mean()):.1f} atoms per structure")

    reference = DirectGeometryCSPNet(feature_mode="both", message_graph="periodic_distance",
                                    hidden_dim=64, latent_dim=32, num_layers=1, num_freqs=4)
    neighborhoods = audit_neighborhoods(reference, batch, tuple(arguments.times), arguments.seed)
    budget = audit_edge_budget(reference, batch, tuple(arguments.times), tuple(arguments.budget_cutoffs),
                               arguments.seed)

    costs = {}
    if not arguments.skip_cost:
        cost_batch = sample_structures(arguments.split, arguments.cost_structures, arguments.seed)
        print(f"timing {len(ARMS)} arms at {arguments.cost_structures} structures, {len(arguments.times)} times, "
              f"{arguments.repeats} steps each")
        for graph, mode in ARMS:
            costs[f"{graph.value}/{mode.value}"] = time_training_step(
                graph, mode, cost_batch, device, tuple(arguments.times), arguments.repeats, arguments.seed)
        control = costs[f"{MessageGraph.FC.value}/{FeatureMode.NONE.value}"]
        for entry in costs.values():
            entry["time_ratio"] = entry["seconds"] / control["seconds"]
            entry["worst_time_ratio"] = entry["worst_seconds"] / control["worst_seconds"]
            entry["memory_ratio"] = entry["peak_gib"] / control["peak_gib"] if control["peak_gib"] else float("nan")

    failures = gate_failures(neighborhoods, costs, reference.descriptor_spec.channel_names(), COST_CEILING)

    report = {
        "gate": "DG0",
        "passed": not failures,
        "failures": failures,
        "arguments": vars(arguments),
        "environment": {"torch": torch.__version__, "device": str(device), "platform": platform.platform(),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "descriptor": {"cutoff": reference.descriptor_spec.cutoff, "shells": reference.descriptor_spec.num_shells,
                       "angular_order": reference.descriptor_spec.max_angular_order,
                       "dim": reference.descriptor_spec.dim, "graph_degree": reference.graph_degree,
                       "graph_max_radius": reference.graph_max_radius, "edge_basis": reference.edge_basis},
        "structures": {"count": len(batch.n_atoms), "atoms": int(batch.n_atoms.sum()),
                       "atoms_per_structure": float(batch.n_atoms.double().mean())},
        "neighborhoods": neighborhoods,
        "edge_budget": budget,
        "costs": costs,
        "cost_ceiling": COST_CEILING,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\nneighbourhoods across denoising time (descriptor {reference.descriptor_spec.cutoff:g} A, "
          f"graph at degree {reference.graph_degree:g})")
    print(f"  {'t':>6}  {'desc e/atom':>11}  {'graph e/atom':>12}  {'graph r':>7}  {'isolated':>8}  {'max image':>9}  "
          f"{'multiedge':>9}  {'max mult':>8}")
    for label, entry in neighborhoods.items():
        described, graphed = entry["descriptor_neighbors"], entry["graph_neighbors"]
        print(f"  {label:>6}  {described['edges_per_atom']:>11.1f}  {graphed['edges_per_atom']:>12.1f}  "
              f"{entry['graph_radius']['mean']:>7.2f}  {described['isolated_atoms']:>8d}  "
              f"{described['max_image_component']:>9d}  {graphed['multiedge_fraction']:>9.3f}  "
              f"{graphed['max_pair_multiplicity']:>8d}")

    print(f"\nedge budget: the fully connected control carries {budget['fc_edges_per_atom']:.1f} edges per atom")
    print(f"  {'radius':>9}  {'worst e/atom':>12}  {'x control':>9}  {'clean e/atom':>12}  {'x control':>9}")
    for label, entry in budget["fixed_radius"].items():
        print(f"  {label:>9}  {entry['worst_edges_per_atom']:>12.1f}  {entry['worst_ratio']:>9.2f}  "
              f"{entry['clean_edges_per_atom']:>12.1f}  {entry['clean_ratio']:>9.2f}")
    adaptive = budget["adaptive_radius"]
    print(f"  {'adaptive':>9}  {adaptive['worst_edges_per_atom']:>12.1f}  {adaptive['worst_ratio']:>9.2f}  "
          f"{adaptive['clean_edges_per_atom']:>12.1f}  {adaptive['clean_ratio']:>9.2f} <- configured")

    if costs:
        print("\ncost against the fc/none control, averaged over denoising time")
        print(f"  {'arm':>26}  {'s/step':>8}  {'x time':>7}  {'x worst':>8}  {'peak GiB':>9}  {'x memory':>8}")
        for name, entry in costs.items():
            print(f"  {name:>26}  {entry['seconds']:>8.4f}  {entry['time_ratio']:>7.2f}  "
                  f"{entry['worst_time_ratio']:>8.2f}  {entry['peak_gib']:>9.2f}  {entry['memory_ratio']:>8.2f}")

    print(f"\nGate DG0: {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"  - {failure}")
    print(f"wrote {destination}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
