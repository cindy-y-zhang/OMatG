"""
Tests of the Gate DG0 audit.

The audit is the only thing standing between a broken descriptor and a twelve-run A100 sweep, and it is the kind of code
that is never read again once it has printed a table. Two parts of it are worth pinning:

- the *verdict*, because a gate that cannot fail is not a gate, and every one of its clauses is a claim about what would
  make the pivot not worth running;
- the *probability path*, because everything the audit reports is measured on interpolated structures, and an audit of the
  wrong geometry is worse than no audit. This was not hypothetical: an earlier draft interpolated towards a lognormal
  jitter of each structure's own cell instead of the baseline's fitted lattice prior, and it reported the periodic graph
  as costing 1.7 times the control when the true figure on the real path was 2.5, which is the difference between passing
  the cost gate and failing it.
"""

import math
import pytest
import torch

from direct_geometry.encoder import DirectGeometryCSPNet
from direct_geometry.features import FeatureMode
from direct_geometry.batches import interpolate, sample_structures
from direct_geometry.scripts.audit_geometry import (COST_CEILING, audit_edge_budget, audit_neighborhoods,
                                                    gate_failures, summarise_neighbors)
from direct_geometry.neighbors import periodic_neighbors
from .conftest import CUBIC, random_structure, single

SPLIT = "cgfm/smoke_data/val.lmdb"
"""The sixteen-structure smoke split, which is enough to exercise every code path and fast enough for a unit test."""

TIMES = (0.001, 0.5, 0.999)
"""Three times rather than the audit's six, for the same reason."""


def reference_encoder() -> DirectGeometryCSPNet:
    """
    Build the tiny encoder the audit uses to describe neighbourhoods.

    :return:
        The encoder.
    :rtype: DirectGeometryCSPNet
    """
    return DirectGeometryCSPNet(feature_mode="both", message_graph="periodic_distance", hidden_dim=32, latent_dim=16,
                               num_layers=1, num_freqs=4)


def passing_measurements() -> tuple[dict, dict, list[str]]:
    """
    Build measurements that Gate DG0 should accept.

    :return:
        Neighbourhood summaries, costs, and the channel names they cover.
    :rtype: tuple[dict, dict, list[str]]
    """
    names = ["radial_0", "angular_0"]
    neighborhoods = {}
    for label in ("0.001", "0.999"):
        neighborhoods[label] = {
            "descriptor_neighbors": {"isolated_atoms": 0},
            "graph_neighbors": {"multiedge_fraction": 0.35},
            "graph_radius": {"at_bound_fraction": 0.15},
            "pathological_cells": {"reduced_fraction": 0.0, "min_effective_cutoff": 6.0},
            "channels": {name: {"std": 0.5} for name in names},
            "finite_fraction": 1.0,
        }
    costs = {"fc/none": {"time_ratio": 1.0, "memory_ratio": 1.0},
             "periodic_distance/both": {"time_ratio": 1.1, "memory_ratio": 1.2}}
    return neighborhoods, costs, names


def test_the_gate_passes_measurements_that_should_pass() -> None:
    """The baseline case, so that the failing cases below are known to be failing for the reason they name."""
    assert gate_failures(*passing_measurements(), COST_CEILING) == []


def test_the_gate_fails_any_descriptor_cutoff_reduction_on_the_probability_path() -> None:
    """
    The pathological-cell bound firing on the path at all is a failure, however few structures it touches.

    That bound exists because generation can drive the cell somewhere no crystal is, and a batch containing such a cell
    still has to be scored. It is not licence to shrink the descriptor's reach on ordinary structures: DG1 measured the
    descriptor's +26.1 accuracy points at 6 Angstrom, and a descriptor computed at less than that on some fraction of
    structures is a different feature than the one that evidence is about. So this is not a tolerance with a threshold.
    """
    neighborhoods, costs, names = passing_measurements()
    neighborhoods["0.001"]["pathological_cells"] = {"reduced_fraction": 0.001, "min_effective_cutoff": 4.2}
    failures = gate_failures(neighborhoods, costs, names, COST_CEILING)
    assert any("cutoff reduced" in reason for reason in failures)


def test_the_gate_fails_a_non_finite_channel() -> None:
    """One NaN in one channel poisons a whole batch's gradient, and it must not be reported as a passing audit."""
    neighborhoods, costs, names = passing_measurements()
    neighborhoods["0.001"]["finite_fraction"] = 0.999
    assert any("non-finite" in reason for reason in gate_failures(neighborhoods, costs, names, COST_CEILING))


def test_the_gate_fails_a_channel_that_never_varies() -> None:
    """
    A constant channel is not a feature. It occupies the projection's input width and can carry nothing.

    Failing on the maximum over times rather than on any single time, because a channel is allowed to be flat at the noisy
    end -- describing a uniform draw -- as long as it says something about a crystal.
    """
    neighborhoods, costs, names = passing_measurements()
    for entry in neighborhoods.values():
        entry["channels"]["angular_0"]["std"] = 0.0
    assert any("angular_0" in reason for reason in gate_failures(neighborhoods, costs, names, COST_CEILING))

    neighborhoods["0.001"]["channels"]["angular_0"]["std"] = 0.4
    assert gate_failures(neighborhoods, costs, names, COST_CEILING) == []


def test_the_gate_fails_when_the_graph_factor_would_be_a_no_op() -> None:
    """
    If no atom pair is joined by more than one periodic image, the periodic graph is the fully connected one relabelled.

    The whole arm would then be an expensive null, and two of the four cells of the factorial would be duplicates.
    """
    neighborhoods, costs, names = passing_measurements()
    neighborhoods["0.999"]["graph_neighbors"]["multiedge_fraction"] = 0.0
    assert any("no-op" in reason for reason in gate_failures(neighborhoods, costs, names, COST_CEILING))

    # Read at the clean end, not the noisy one: a uniform draw has multiedges whatever the crystal does.
    neighborhoods["0.999"]["graph_neighbors"]["multiedge_fraction"] = 0.3
    neighborhoods["0.001"]["graph_neighbors"]["multiedge_fraction"] = 0.0
    assert gate_failures(neighborhoods, costs, names, COST_CEILING) == []


def test_the_gate_fails_isolated_atoms_only_where_the_structures_are_crystals() -> None:
    """
    An atom with no neighbours has an all-zero descriptor, which is a blind spot rather than a description.

    Tolerated at the noisy end, where a uniform draw in a large cell can genuinely leave an atom alone, and refused near
    the data end, where it would mean the cutoff or the image enumeration is wrong.
    """
    neighborhoods, costs, names = passing_measurements()
    neighborhoods["0.001"]["descriptor_neighbors"]["isolated_atoms"] = 5
    assert gate_failures(neighborhoods, costs, names, COST_CEILING) == []

    neighborhoods["0.999"]["descriptor_neighbors"]["isolated_atoms"] = 1
    assert any("empty neighbourhood" in reason for reason in gate_failures(neighborhoods, costs, names, COST_CEILING))


def test_the_gate_fails_when_the_adaptive_radius_has_reverted_to_a_fixed_one() -> None:
    """
    If most structures are held at the radius bound, the constant-degree graph is a fixed-radius graph with extra code.

    Worth failing on rather than tolerating, because it would silently remove the matched-connectivity argument that makes
    the graph factor interpretable, and the symptom -- a slightly low degree -- looks like nothing at all.
    """
    neighborhoods, costs, names = passing_measurements()
    neighborhoods["0.999"]["graph_radius"]["at_bound_fraction"] = 0.8
    assert any("fixed-radius" in reason for reason in gate_failures(neighborhoods, costs, names, COST_CEILING))

    # Read at the clean end only: at the noisy end the cells are small and the bound cannot bind anyway.
    neighborhoods["0.999"]["graph_radius"]["at_bound_fraction"] = 0.15
    neighborhoods["0.001"]["graph_radius"]["at_bound_fraction"] = 0.8
    assert gate_failures(neighborhoods, costs, names, COST_CEILING) == []


def test_the_gate_fails_a_cost_over_the_ceiling() -> None:
    """The budget is a real constraint: the production sweep is twelve runs of 1,600 epochs."""
    neighborhoods, costs, names = passing_measurements()
    costs["periodic_distance/both"]["time_ratio"] = COST_CEILING + 0.01
    reasons = gate_failures(neighborhoods, costs, names, COST_CEILING)
    assert any("step time" in reason for reason in reasons)

    costs["periodic_distance/both"]["time_ratio"] = 1.0
    costs["periodic_distance/both"]["memory_ratio"] = COST_CEILING + 0.01
    assert any("memory" in reason for reason in gate_failures(neighborhoods, costs, names, COST_CEILING))


def test_an_unmeasured_memory_ratio_is_not_read_as_a_pass() -> None:
    """
    Peak allocation is not reported on a CPU, and a comparison against a NaN is false, which would pass silently.

    Asserted because that is the failure mode of the obvious implementation, and it would only ever be hit on the machine
    that had no GPU to measure -- exactly where nobody would notice the budget went unchecked.
    """
    neighborhoods, costs, names = passing_measurements()
    costs["periodic_distance/both"]["memory_ratio"] = float("nan")
    assert gate_failures(neighborhoods, costs, names, COST_CEILING) == []
    assert not (float("nan") > COST_CEILING)


def test_the_audited_path_is_the_baselines_own() -> None:
    """
    At the data end the interpolated state must *be* the data, and at the base end it must be a draw from the prior.

    Endpoint checks rather than a comparison against a reimplementation, since the whole point of calling the baseline's
    interpolants is that there is no second implementation to compare against. The cell is the informative one: the prior
    is a fitted lognormal over lengths and a uniform over angles, so a prior cell has nothing to do with the structure's
    own, and an audit that quietly interpolated from a jittered copy of the data cell would understate the density -- and
    with it the cost -- at every time.
    """
    batch = sample_structures(SPLIT, 8, seed=0)
    frac, cell = interpolate(batch, 1.0, seed=0)
    assert torch.allclose(cell, batch.cell)
    # Positions agree up to the wrap the corrector applies, so compare the fractional offset to the nearest image.
    offset = frac - batch.pos
    assert torch.allclose(offset - offset.round(), torch.zeros_like(offset), atol=1.0e-9)

    base_frac, base_cell = interpolate(batch, 0.0, seed=0)
    assert bool(((base_frac >= 0.0) & (base_frac < 1.0)).all())
    assert not torch.allclose(base_cell, batch.cell)
    # The prior's own scale, not the data's: MPTS-52 cells average well over the prior's lengths of about 5 to 8 Angstrom.
    assert float(torch.linalg.det(base_cell).abs().mean()) < float(torch.linalg.det(batch.cell).abs().mean())


def test_the_audited_path_is_reproducible_and_moves_monotonically_towards_the_data() -> None:
    """
    A seed has to fix the base draw, or two audits of the same code disagree and neither can be quoted.

    The cells are checked for monotone approach rather than the positions, because the periodic interpolant's corrector
    wraps positions and a wrapped coordinate is not closer to the target in the naive sense.
    """
    batch = sample_structures(SPLIT, 8, seed=0)
    first = interpolate(batch, 0.5, seed=3)
    again = interpolate(batch, 0.5, seed=3)
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    assert not torch.equal(interpolate(batch, 0.5, seed=4)[1], first[1])

    previous = None
    for time_value in (0.0, 0.25, 0.5, 0.75, 1.0):
        gap = float((interpolate(batch, time_value, seed=3)[1] - batch.cell).abs().sum())
        if previous is not None:
            assert gap <= previous + 1.0e-9
        previous = gap


def test_the_neighbour_summary_counts_the_multiplicity_it_claims_to() -> None:
    """
    Checked on a caesium-chloride-like cell, where the answer is known: one site, eight images, no distinct neighbours.

    Every atom pair is therefore a multiedge, and a summary reporting anything else is reporting the fully connected
    graph's view of the structure rather than the periodic one's.
    """
    lattice = torch.eye(3, dtype=torch.get_default_dtype()) * 4.0
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=lattice.dtype)
    neighbors = periodic_neighbors(*single(frac, lattice), 3.5)
    summary = summarise_neighbors(neighbors, 2)
    # Each atom sees the other at eight body-centre images, at sqrt(3) * 2 Angstrom.
    assert summary["degree_min"] == summary["degree_max"] == 8
    assert summary["multiedge_fraction"] == pytest.approx(1.0)
    assert summary["max_pair_multiplicity"] == 8
    assert summary["isolated_atoms"] == 0

    isolated = summarise_neighbors(periodic_neighbors(*single(frac, lattice * 10.0), 3.5), 2)
    assert isolated["edges"] == 0 and isolated["isolated_atoms"] == 2
    assert isolated["multiedge_fraction"] == 0.0


def test_the_edge_budget_control_is_the_fully_connected_graphs_own_degree() -> None:
    """
    The control's edges per atom is a property of the split, not a constant, and getting it wrong reverses the conclusion.

    ``CSPNetFull.gen_edges`` joins every ordered pair within a structure including the diagonal, so the atom-weighted mean
    is ``sum(n^2) / sum(n)`` -- which is larger than the mean structure size whenever the sizes vary, and it is the number
    the graph's degree is matched to.
    """
    batch = sample_structures(SPLIT, 8, seed=0)
    budget = audit_edge_budget(reference_encoder(), batch, TIMES, (4.0,), seed=0)
    expected = float((batch.n_atoms.double() ** 2).sum()) / int(batch.n_atoms.sum())
    assert budget["fc_edges_per_atom"] == pytest.approx(expected)
    assert expected > float(batch.n_atoms.double().mean())


def test_the_edge_budget_shows_the_bind_that_motivates_the_adaptive_radius() -> None:
    """
    Asserts the measurement the graph's design rests on, so that a change to the path cannot silently invalidate it.

    A fixed radius is a fixed *volume*, so its degree tracks the density, and on this probability path the density changes
    by about a factor of two between the ends. The adaptive radius is a fixed degree by construction. If the fixed radius
    ever stopped varying, the reason for the adaptive one would have gone away and the extra mechanism should go with it.
    """
    batch = sample_structures(SPLIT, 16, seed=0)
    budget = audit_edge_budget(reference_encoder(), batch, TIMES, (5.0,), seed=0)
    fixed = budget["fixed_radius"]["5"]
    adaptive = budget["adaptive_radius"]
    fixed_swing = fixed["worst_edges_per_atom"] / fixed["clean_edges_per_atom"]
    adaptive_swing = adaptive["worst_edges_per_atom"] / adaptive["clean_edges_per_atom"]
    # Stated as a comparison on one batch rather than as an absolute tolerance, because the absolute constancy depends on
    # how many structures are sampled and on how many of them are held at the radius bound.
    assert fixed_swing > 1.8
    assert adaptive_swing < 1.5
    assert adaptive_swing < 0.5 * fixed_swing
    # Matched to the fully connected control it is compared against, which is the reason the degree was set where it was.
    assert adaptive["clean_ratio"] == pytest.approx(1.0, abs=0.3)


def test_the_neighbourhood_audit_reports_a_radius_that_follows_the_density() -> None:
    """
    The graph radius has to grow as the interpolated cell grows towards the data, or the mechanism is not working.

    Also asserts the descriptor's own neighbourhood is *not* held at constant degree: it has a fixed cutoff, because a
    descriptor whose radius moved with the structure would describe a different volume in every structure and its channels
    would no longer be comparable across the batch.
    """
    encoder = reference_encoder()
    batch = sample_structures(SPLIT, 16, seed=0)
    report = audit_neighborhoods(encoder, batch, TIMES, seed=0)
    radii = [report[f"{time_value:g}"]["graph_radius"]["mean"] for time_value in TIMES]
    assert radii[0] < radii[-1]
    assert all(radius <= encoder.graph_max_radius + 1.0e-9 for radius in radii)

    graph_degrees = [report[f"{t:g}"]["graph_neighbors"]["edges_per_atom"] for t in TIMES]
    descriptor_degrees = [report[f"{t:g}"]["descriptor_neighbors"]["edges_per_atom"] for t in TIMES]
    graph_swing = max(graph_degrees) / min(graph_degrees)
    descriptor_swing = max(descriptor_degrees) / min(descriptor_degrees)
    assert graph_swing < 1.5
    assert descriptor_swing > 2.0
    assert graph_swing < 0.5 * descriptor_swing


def test_the_neighbourhood_audit_covers_every_descriptor_channel() -> None:
    """
    The gate reads a variance for every channel name, so a name the statistics do not carry would raise mid-audit.

    Cheap to assert and it couples two files that are otherwise free to drift: the descriptor's channel naming and the
    audit's reading of it.
    """
    encoder = reference_encoder()
    report = audit_neighborhoods(encoder, sample_structures(SPLIT, 4, seed=0), (0.999,), seed=0)
    statistics = report["0.999"]["channels"]
    assert set(statistics) == set(encoder.descriptor_spec.channel_names())
    assert gate_failures(report, {}, encoder.descriptor_spec.channel_names(), COST_CEILING) == []


def test_the_descriptor_the_audit_reports_is_the_one_the_encoder_would_use() -> None:
    """
    The audit summarises ``FeatureMode.BOTH`` so that a mode-masked channel is not reported as dead.

    Without this the ``radial`` arm's audit would list every angular channel as having zero variance, the gate would fail,
    and the fix would look like a descriptor bug rather than a reporting one.
    """
    encoder = DirectGeometryCSPNet(feature_mode="radial", message_graph="fc", hidden_dim=32, latent_dim=16,
                                   num_layers=1, num_freqs=4)
    assert encoder.feature_mode is FeatureMode.RADIAL
    report = audit_neighborhoods(encoder, sample_structures(SPLIT, 4, seed=0), (0.999,), seed=0)
    spreads = [entry["std"] for entry in report["0.999"]["channels"].values()]
    assert all(spread > 0.0 for spread in spreads)


def test_the_audit_batch_is_the_fractional_form_the_encoder_expects() -> None:
    """
    The split stores Cartesian positions and the encoder reads fractional ones.

    Reading the ``StructureDataset`` directly rather than through ``OMGDataset`` skips that conversion, and every number in
    the audit would then describe a crystal a few hundred Angstrom across. It fails loudly here rather than producing a
    report full of isolated atoms.
    """
    batch = sample_structures(SPLIT, 4, seed=0)
    assert bool(batch.pos_is_fractional.all())
    assert float(batch.pos.min()) >= 0.0 and float(batch.pos.max()) <= 1.0


def test_the_expected_degree_relation_the_radius_inverts_holds_on_a_random_structure() -> None:
    """
    Independent check of the algebra the adaptive radius is derived from, on a case with a closed form.

    Atoms scattered uniformly at number density ``rho`` place ``(4 pi / 3) r^3 rho`` neighbours inside radius ``r``. If
    this relation did not hold on this geometry there would be no sense in which ``graph_degree`` is a degree.
    """
    frac, cell, atoms = random_structure(80, CUBIC, seed=211)
    density = int(atoms.sum()) / float(torch.linalg.det(cell).abs())
    for radius in (2.5, 3.5):
        expected = 4.0 / 3.0 * math.pi * radius ** 3 * density
        counted = len(periodic_neighbors(frac, cell, atoms, radius)) / int(atoms.sum())
        assert counted == pytest.approx(expected, rel=0.2)
