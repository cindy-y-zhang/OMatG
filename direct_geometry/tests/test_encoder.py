"""
Tests of the direct-geometry encoder.

The study this package exists to run compares four arms of one model class. For the comparison to say anything, three
properties have to hold, and none of them is self-evident from reading the code:

- the baseline combination has to be the *stock* encoder, not a near copy of it, because the whole point of the atomwise
  baseline is that it is the number the project has already measured;
- every arm has to be the *same function* at initialisation, with the same parameter count and the same weights, so that a
  divergence in a learning curve is something training found rather than a different random draw;
- both additions have to actually be reachable by the optimiser, because a zero-initialised path that receives no gradient
  is indistinguishable from an absent one for the first thousand steps and then forever.

The first of these also guards against a specific hazard: ``_forward`` duplicates the body of ``CSPNetFull._forward``,
because the baseline offers no hook for adding a node term after the atom and time embedding. A drift between the two
bodies must fail a test rather than quietly bias a study, which is what the first test here is for.
"""

from pathlib import Path
import pytest
import torch
from omg.model.encoders.cspnet_full import CSPNetFull

from direct_geometry.encoder import (GEOMETRY_PROJECTION_KEY, DirectGeometryCSPNet, MessageGraph,
                                     baseline_encoder_state)
from direct_geometry.features import FeatureMode
from direct_geometry.neighbors import constant_degree_radius
from .conftest import CUBIC, SHORT, SKEWED


SMALL = dict(hidden_dim=64, latent_dim=32, num_layers=2, num_freqs=8)
"""A small encoder. These tests assert exact identities, which do not depend on width."""

ARMS = (("fc", "none"), ("periodic_distance", "none"), ("fc", "both"), ("periodic_distance", "both"))
"""The four factorial arms, as (message_graph, feature_mode) pairs: A, B, C and D of the plan."""

BASELINE_CHECKPOINT = Path("cgfm/centre_runs/atomwise/checkpoints/best_match_rate.ckpt")
"""
The real 1,100-epoch atomwise run.

Tested against rather than a freshly initialised stand-in, because the claim being checked is that *this* encoder can be
dropped in front of the weights the project's existing baseline number was produced by. Skipped when absent, since the
A100 bundle deliberately ships no checkpoints.
"""


def inputs(sizes: tuple[int, ...] = (5, 4), cells: tuple[torch.Tensor, ...] = (CUBIC, SHORT),
           latent: int = 32, seed: int = 1) -> tuple:
    """
    Build the positional arguments ``_forward`` takes.

    :param sizes:
        Number of atoms of every structure.
        Defaults to (5, 4).
    :type sizes: tuple[int, ...]
    :param cells:
        Lattice of every structure.
        Defaults to (CUBIC, SHORT), so that one structure needs more periodic images than the other and the builder's
        grouping is exercised.
    :type cells: tuple[torch.Tensor, ...]
    :param latent:
        Width of the time embedding.
        Defaults to 32.
    :type latent: int
    :param seed:
        Seed of the random inputs.
        Defaults to 1.
    :type seed: int

    :return:
        Atomic numbers, fractional coordinates, lattices, atom counts, structure indices and embedded times.
    :rtype: tuple
    """
    generator = torch.Generator().manual_seed(seed)
    atoms = torch.tensor(sizes)
    total = int(atoms.sum())
    return (torch.randint(1, 90, (total,), generator=generator),
            torch.rand((total, 3), generator=generator),
            torch.stack(list(cells)),
            atoms,
            torch.repeat_interleave(torch.arange(len(sizes)), atoms),
            torch.rand((len(sizes), latent), generator=generator))


def build(graph: str, mode: str, seed: int = 0, **overrides) -> DirectGeometryCSPNet:
    """
    Build one arm from a fixed random state.

    :param graph:
        Message graph name.
    :type graph: str
    :param mode:
        Feature mode name.
    :type mode: str
    :param seed:
        Seed of the initial weights.
        Defaults to 0.
    :type seed: int
    :param overrides:
        Constructor arguments replacing the small defaults.
    :type overrides: Any

    :return:
        The encoder.
    :rtype: DirectGeometryCSPNet
    """
    torch.manual_seed(seed)
    return DirectGeometryCSPNet(feature_mode=mode, message_graph=graph, **{**SMALL, **overrides})


def same(first, second) -> bool:
    """
    Return whether two encoder outputs agree bit for bit on every field.

    :param first:
        One output.
    :type first: torch_geometric.data.Data
    :param second:
        The other output.
    :type second: torch_geometric.data.Data

    :return:
        True if every field is present in both and bitwise equal.
    :rtype: bool
    """
    keys = set(first.keys()) | set(second.keys())
    return all(torch.equal(first[key], second[key]) for key in keys)


def close(first, second, atol: float = 1.0e-11) -> bool:
    """
    Return whether two encoder outputs agree on every field to a tolerance.

    Needed only where the two encoders differ in the *width* of a matmul, which is the case for a zero-appended edge
    network against the baseline it widens. The sum then runs over more terms in a different order, so adding
    zero-weighted terms is exact in arithmetic and not in floating point. Every comparison between two encoders of equal
    width uses ``same`` instead, because there bit equality is achievable and is the stronger statement.

    :param first:
        One output.
    :type first: torch_geometric.data.Data
    :param second:
        The other output.
    :type second: torch_geometric.data.Data
    :param atol:
        Absolute tolerance. Defaults to 1e-11, some orders of magnitude above the observed 4e-16 double-precision
        disagreement and far below any effect a live column would produce.
    :type atol: float

    :return:
        True if every field is present in both and agrees to the tolerance.
    :rtype: bool
    """
    keys = set(first.keys()) | set(second.keys())
    return all(torch.allclose(first[key], second[key], rtol=0.0, atol=atol) for key in keys)


def test_the_baseline_arm_reproduces_the_stock_encoder_bit_for_bit() -> None:
    """
    ``fc`` and ``none`` is the stock ``CSPNetFull``, on every output field.

    This is the test that makes the duplicated ``_forward`` body safe. If a future edit to this package's forward pass
    drifts from the baseline's -- a reordered concatenation, a missing layer norm, an aggregation changed from mean -- this
    fails, and the study's control does not silently stop being the control.
    """
    torch.manual_seed(0)
    baseline = CSPNetFull(**SMALL)
    arm = build("fc", "none")
    arm.load_baseline_state_dict(baseline.state_dict())
    arguments = inputs()
    assert same(baseline._forward(*arguments), arm._forward(*arguments))


def test_the_baseline_arm_builds_no_neighbour_list_at_all() -> None:
    """
    The control must not pay for geometry it does not use.

    It is the cost reference the thirty per cent ceiling in Gate DG0 is measured against, so if it quietly enumerated
    periodic images the ceiling would be measured against the wrong number and would be easier to pass than it looks.
    """
    assert not build("fc", "none").uses_neighbors
    for graph, mode in ARMS[1:]:
        assert build(graph, mode).uses_neighbors, (graph, mode)


@pytest.mark.parametrize("mode", [mode.value for mode in FeatureMode])
def test_every_node_feature_mode_is_the_same_function_at_initialisation(mode: str) -> None:
    """
    Turning the descriptor on changes nothing until training moves the projection.

    The projection is bias-free and zero-initialised, so at step zero every mode on the fully connected graph produces the
    baseline's output exactly. Without this, an arm's first-epoch loss would differ from the control's for a reason that
    has nothing to do with geometry.
    """
    arguments = inputs()
    reference = build("fc", "none")._forward(*arguments)
    assert same(reference, build("fc", mode)._forward(*arguments))


def test_the_periodic_graph_is_the_same_function_with_or_without_the_descriptor_at_initialisation() -> None:
    """
    Same statement on the other topology: at initialisation B and D are one function.

    Their outputs differ from the fully connected arms', which is the point of the graph factor, but they must not differ
    from each other before training.
    """
    arguments = inputs()
    assert same(build("periodic_distance", "none")._forward(*arguments),
                build("periodic_distance", "both")._forward(*arguments))


def test_the_periodic_graph_changes_the_answer_and_the_fully_connected_one_does_not_see_it() -> None:
    """
    The topology factor is real: a different graph gives a different function even with every addition zeroed.

    Worth pinning because it is easy to write a periodic graph that silently reduces to the fully connected one -- wrap the
    offsets and the images collapse -- and the resulting arm would look like a null result for the topology.
    """
    arguments = inputs()
    assert not same(build("fc", "none")._forward(*arguments), build("periodic_distance", "none")._forward(*arguments))


def test_the_distance_columns_are_the_only_difference_the_edge_feature_makes() -> None:
    """
    Perturbing the appended edge columns changes the prediction, and restoring them to zero restores it exactly.

    That is the precise sense in which the length expansion is an addition to the periodic graph rather than a change to it:
    the topology and every baseline weight are held fixed while only those columns move.
    """
    arguments = inputs()
    arm = build("periodic_distance", "both")
    zeroed = arm._forward(*arguments)
    first = arm.csp_layer_0.edge_mlp[0]
    columns = slice(arm.csp_layer_0.baseline_edge_features, None)
    with torch.no_grad():
        first.weight[:, columns] = 0.1
    assert not same(zeroed, arm._forward(*arguments))
    with torch.no_grad():
        first.weight[:, columns] = 0.0
    assert same(zeroed, arm._forward(*arguments))


def test_the_length_only_arm_keeps_the_baselines_topology_exactly() -> None:
    """
    Arm E changes the edge features and not the graph.

    This is the whole reason E is worth running: it is the baseline's own edges, so E-A cannot be explained by a different
    number of neighbours, and it needs no periodic neighbour list, so it cannot be explained by cost either. A regression
    that routed E through the periodic builder would make it a second copy of B and quietly destroy the decomposition.
    """
    arguments = inputs()
    atom_types, frac, cells, atoms, node2graph, _ = arguments
    control, length_only = build("fc", "none"), build("fc_distance", "none")

    assert not control.uses_neighbors and not length_only.uses_neighbors

    control_edges, control_offsets, control_expansion = control._graph(None, atoms, frac, cells, node2graph)
    edges, offsets, expansion = length_only._graph(None, atoms, frac, cells, node2graph)
    assert torch.equal(control_edges, edges)
    assert torch.equal(control_offsets, offsets)
    assert control_expansion is None and expansion is not None
    assert expansion.shape == (edges.shape[1], length_only.edge_basis)


def test_the_length_only_arm_starts_as_the_baseline_and_moves_when_its_columns_do() -> None:
    """
    E is the baseline function at initialisation, and stops being it exactly when the appended columns are used.

    Same statement already made for the periodic graph, repeated here because E is the arm whose whole claim is "one
    channel added, nothing else touched", and a nonzero initialisation would make its first epoch differ from A's for a
    reason unrelated to distance.

    Compared to a tolerance rather than bit for bit, which the same-width comparisons elsewhere in this file do use. The
    appended columns make the edge network's first matmul wider, so its sum runs over more terms in a different order;
    adding zero-weighted terms is exact in arithmetic but not in floating point. At double precision the arms agree to
    about 4e-16, so the tolerance below is roughly one part in a hundred thousand of any real effect and cannot hide a
    live column.
    """
    arguments = inputs()
    assert close(build("fc", "none")._forward(*arguments), build("fc_distance", "none")._forward(*arguments))

    arm = build("fc_distance", "none")
    zeroed = arm._forward(*arguments)
    columns = slice(arm.csp_layer_0.baseline_edge_features, None)
    with torch.no_grad():
        arm.csp_layer_0.edge_mlp[0].weight[:, columns] = 0.1
    assert not same(zeroed, arm._forward(*arguments))


def test_the_length_only_arm_is_not_the_periodic_arm() -> None:
    """
    E and B are different functions once their distance columns are live.

    They share the distance channel and differ in topology, so this is the assertion that B-E measures something. If the
    periodic builder ever collapsed to one edge per pair the two would coincide and B-E would read as a null topology
    effect rather than as a broken graph.
    """
    arguments = inputs()
    length_only, periodic = build("fc_distance", "none"), build("periodic_distance", "none")
    columns = slice(length_only.csp_layer_0.baseline_edge_features, None)
    for arm in (length_only, periodic):
        with torch.no_grad():
            arm.csp_layer_0.edge_mlp[0].weight[:, columns] = 0.1
    assert not same(length_only._forward(*arguments), periodic._forward(*arguments))


def test_the_length_only_arm_reads_a_true_nearest_image_distance() -> None:
    """
    The length E supplies is the true nearest-image separation, not the length of the wrapped fractional offset.

    The baseline hands the edge network ``(x_j - x_i) % 1``, whose naive Cartesian length reports the long way round for
    any pair more than half a cell apart, and whose *folded* length is still wrong on a skewed cell because rounding picks
    the nearest image in fractional space rather than in Cartesian space. Since supplying a correct length is the entire
    mechanism this arm tests, either error would make E a test of nothing.

    Checked on a skewed cell against a brute-force search over a wider range of images than the helper searches, which
    also confirms the helper's own range is enough here. The wrong answers are asserted to be wrong, so this fails rather
    than passes if the encoder ever reverts to either shortcut.
    """
    atoms = torch.tensor([4])
    frac = torch.tensor([[0.05, 0.05, 0.05], [0.95, 0.5, 0.5], [0.5, 0.9, 0.1], [0.4, 0.4, 0.6]],
                        dtype=torch.float64)
    cells = SKEWED.unsqueeze(0)
    node2graph = torch.zeros(4, dtype=torch.long)
    arm = build("fc_distance", "none")

    edges, offsets, expansion = arm._graph(None, atoms, frac, cells, node2graph)
    span = torch.arange(-3, 4, dtype=torch.float64)
    shifts = torch.stack(torch.meshgrid(span, span, span, indexing="ij"), dim=-1).reshape(-1, 3)
    reference = ((offsets.unsqueeze(1) + shifts.unsqueeze(0)) @ cells[0]).norm(dim=-1).amin(dim=1)

    # Read the distance back out of the basis rather than trusting an internal: the arm's contract is what the edge
    # network sees, and the profile peaks at the shell nearest the true length.
    centres = torch.linspace(0.0, arm.graph_max_radius, arm.edge_basis, dtype=expansion.dtype)
    recovered = centres[expansion.argmax(dim=-1)]
    assert torch.allclose(recovered, reference, atol=float(centres[1] - centres[0]))

    unfolded = (offsets @ cells[0]).norm(dim=-1)
    folded = ((offsets - offsets.round()) @ cells[0]).norm(dim=-1)
    assert not torch.allclose(unfolded, reference, atol=1.0e-6)
    assert not torch.allclose(folded, reference, atol=1.0e-6)


def test_the_message_graph_uses_each_structures_own_radius() -> None:
    """
    The graph is specified by degree, so a batch of differently sized cells must get differently sized neighbourhoods.

    Checked by comparing against the radius computed independently: a bug that used one radius for the whole batch -- the
    global maximum, say -- would still produce a plausible graph, and on the near-uniform cells of a real batch it would
    be invisible in every summary statistic.
    """
    arm = build("periodic_distance", "none", graph_degree=8.0)
    _, frac, cell, atoms, node2graph, _ = inputs(sizes=(6, 6), cells=(CUBIC, CUBIC * 0.6))
    radius = constant_degree_radius(cell, atoms, arm.graph_degree, arm.graph_max_radius)
    assert float(radius[0]) > float(radius[1])

    neighbors = arm.build_neighbors(frac, cell, atoms)
    edges, offsets, expansion = arm._graph(neighbors, atoms, frac, cell, node2graph)
    edge2graph = node2graph[edges[0]]
    lengths = torch.einsum("ek,ekj->ej", offsets, cell[edge2graph]).norm(dim=-1)
    assert bool((lengths <= radius[edge2graph] + 1.0e-9).all())
    for structure in (0, 1):
        selected = edge2graph == structure
        expected = int((neighbors.within(radius).edge2graph == structure).sum())
        assert int(selected.sum()) == expected
    assert expansion.shape == (edges.shape[1], arm.edge_basis)


def test_the_edge_basis_reads_absolute_distance_not_a_ratio() -> None:
    """
    Two structures of different density must map the same physical bond length to the same expansion.

    This is the property that makes the length channel worth adding at all. The baseline's edge feature is a sinusoidal
    embedding of a *fractional* offset, which says nothing about Angstrom, so the trunk has no absolute length anywhere.
    Had the expansion been placed on a grid that scaled with each structure's radius -- the obvious way to pair a basis
    with an adaptive cutoff -- it would encode a ratio and the arm would restate what the baseline already has.
    """
    arm = build("periodic_distance", "none")
    separation = 2.0
    features = []
    for size in (5.0, 8.0):
        # One pair at a fixed 2 Angstrom separation, in cells that differ in volume by a factor of four.
        frac = torch.tensor([[0.0, 0.0, 0.0], [separation / size, 0.0, 0.0]], dtype=torch.get_default_dtype())
        cell = (torch.eye(3, dtype=frac.dtype) * size).unsqueeze(0)
        atoms = torch.tensor([2])
        node2graph = torch.zeros(2, dtype=torch.long)
        neighbors = arm.build_neighbors(frac, cell, atoms)
        edges, offsets, expansion = arm._graph(neighbors, atoms, frac, cell, node2graph)
        lengths = torch.einsum("ek,ekj->ej", offsets, cell[node2graph[edges[0]]]).norm(dim=-1)
        nearest = int(lengths.argmin())
        assert float(lengths[nearest]) == pytest.approx(separation)
        features.append(expansion[nearest])
    # The Gaussian coefficients are identical; only the smooth envelope differs, since the two radii differ.
    first, second = features
    assert torch.allclose(first / first.max(), second / second.max(), atol=1.0e-12)
    assert int(first.argmax()) == int(second.argmax())


def test_every_arm_starts_from_identical_baseline_weights() -> None:
    """
    The arms differ in what they are shown, not in how they were initialised.

    The replacement layers consume extra randomness when they are built, so they copy the layers they replace rather than
    keeping their own draw. Without that, arm D's trunk would start from different weights than arm A's and every contrast
    would carry the difference.
    """
    reference = build("fc", "none").state_dict()
    for graph, mode in ARMS[1:]:
        other = build(graph, mode).state_dict()
        for name, value in reference.items():
            assert name in other, name
            if value.shape == other[name].shape:
                assert torch.equal(value, other[name]), f"{graph}/{mode}: {name}"
            else:
                # Only the widened edge weight may differ in shape, and only by zeroed columns on the right.
                assert torch.equal(value, other[name][:, :value.shape[1]]), f"{graph}/{mode}: {name}"
                assert bool((other[name][:, value.shape[1]:] == 0.0).all()), f"{graph}/{mode}: {name}"


@pytest.mark.parametrize("graph,mode", ARMS)
def test_every_arm_has_the_same_trainable_parameter_count_up_to_the_two_additions(graph: str, mode: str) -> None:
    """
    The additions are the only parameters that vary, and their sizes are named here.

    A study whose arms differ in capacity cannot attribute a win to information. The projection is present in every arm,
    including the ones that never evaluate it, so only the periodic graph's edge columns actually change the count.
    """
    arm = build(graph, mode)
    total = sum(parameter.numel() for parameter in arm.parameters())
    baseline = sum(parameter.numel() for parameter in CSPNetFull(**SMALL).parameters())
    expected = baseline + arm.descriptor_spec.dim * arm.hidden_dim
    if graph == MessageGraph.PERIODIC_DISTANCE.value:
        expected += arm.num_layers * arm.edge_basis * arm.hidden_dim
    assert total == expected


@pytest.mark.parametrize("graph,mode", ARMS)
def test_one_optimizer_step_reaches_both_additions(graph: str, mode: str) -> None:
    """
    A zero-initialised path must receive gradient, or it is an absent path with a parameter count.

    Checked as a gradient *and* as a change in the prediction after a step, because a nonzero gradient that the optimiser
    cannot act on -- a detached tensor, a projection applied to a constant -- would pass the first check alone.
    """
    arguments = inputs()
    arm = build(graph, mode)
    before = arm._forward(*arguments)["pos_b"].detach().clone()
    optimizer = torch.optim.SGD(arm.parameters(), lr=1.0e-2)
    output = arm._forward(*arguments)
    (output["pos_b"].pow(2).sum() + output["cell_b"].pow(2).sum()).backward()

    projection = arm.geometry_projection.weight.grad
    if FeatureMode(mode).uses_geometry:
        assert projection is not None and float(projection.abs().max()) > 0.0
    else:
        # The descriptor is never evaluated in this mode, so the projection is genuinely outside the graph and has no
        # gradient at all. It exists anyway, to hold the parameter count equal across arms, and stays at zero.
        assert projection is None
        assert float(arm.geometry_projection.weight.detach().abs().max()) == 0.0
    if graph == MessageGraph.PERIODIC_DISTANCE.value:
        columns = arm.csp_layer_0.edge_mlp[0].weight.grad[:, arm.csp_layer_0.baseline_edge_features:]
        assert float(columns.abs().max()) > 0.0

    optimizer.step()
    after = arm._forward(*arguments)["pos_b"]
    assert not torch.allclose(before, after)


@pytest.mark.parametrize("graph,mode", ARMS)
def test_a_stock_state_dict_loads_into_every_arm(graph: str, mode: str) -> None:
    """
    Every arm can be initialised from a baseline run, which is what makes a warm-started ablation possible at all.

    On the fully connected graph the loaded arm must reproduce the baseline exactly; on the periodic graph it reproduces
    the baseline weights on a different topology, which is the arm's whole content.
    """
    torch.manual_seed(0)
    baseline = CSPNetFull(**SMALL)
    arm = build(graph, mode, seed=99)
    arm.load_baseline_state_dict(baseline.state_dict())
    for name, value in baseline.state_dict().items():
        loaded = arm.state_dict()[name]
        assert torch.equal(value, loaded[:, :value.shape[1]] if value.shape != loaded.shape else loaded), name
    assert bool((arm.geometry_projection.weight == 0.0).all())


def test_loading_refuses_a_state_dict_that_does_not_describe_this_encoder() -> None:
    """
    A silent partial load is the worst outcome here: it produces a model that runs and whose weights are half random.

    Only the geometry projection may be absent. Anything else missing, anything unexpected, or a shape that is not an
    accountable widening stops the load.
    """
    torch.manual_seed(0)
    baseline = CSPNetFull(**SMALL).state_dict()
    arm = build("fc", "none")

    truncated = {name: value for name, value in baseline.items() if "csp_layer_1" not in name}
    with pytest.raises(ValueError, match="does not describe this encoder"):
        arm.load_baseline_state_dict(truncated)

    extra = dict(baseline)
    extra["something_else"] = torch.zeros(3)
    with pytest.raises(ValueError, match="does not describe this encoder"):
        arm.load_baseline_state_dict(extra)

    wrong = dict(baseline)
    wrong["coord_out.weight"] = torch.zeros((7, 7))
    with pytest.raises(ValueError, match="not a widening"):
        arm.load_baseline_state_dict(wrong)

    # And the one permitted absence is permitted.
    assert GEOMETRY_PROJECTION_KEY not in baseline
    arm.load_baseline_state_dict(baseline)


def test_an_unknown_arm_name_is_refused() -> None:
    """A misspelled mode must not fall back to the baseline, which would silently run the control as an experimental arm."""
    with pytest.raises(ValueError):
        DirectGeometryCSPNet(feature_mode="radial_and_angular", **SMALL)
    with pytest.raises(ValueError):
        DirectGeometryCSPNet(message_graph="knn", **SMALL)
    with pytest.raises(ValueError, match="graph degree must be positive"):
        DirectGeometryCSPNet(graph_degree=0.0, **SMALL)
    with pytest.raises(ValueError, match="graph radius bound must be positive"):
        DirectGeometryCSPNet(graph_max_radius=-1.0, **SMALL)


def test_the_output_fields_are_the_baseline_ones() -> None:
    """
    The stochastic interpolants read specific field names off the encoder's output, and they are not changed here.

    The additions are inputs to the trunk, not new outputs, so the sampler, the losses and the flow weights stay the
    baseline's and a direct-geometry run is solving the same generative problem.
    """
    arguments = inputs()
    arm = build("periodic_distance", "both")
    output = arm._forward(*arguments)
    assert set(output.keys()) == {"species_b", "species_eta", "pos_b", "pos_eta", "cell_b", "cell_eta"}
    atoms = int(arguments[3].sum())
    assert output["pos_b"].shape == (atoms, 3) and output["cell_b"].shape == (len(arguments[3]), 3, 3)


def test_a_skewed_cell_and_a_short_cell_in_one_batch_both_run() -> None:
    """
    Both factors have to survive a realistic batch: cells needing different image counts, and structures of different size.

    The neighbour builder groups by repetition count, and the shapes of everything downstream depend on that grouping
    producing one flat edge list in a consistent order.
    """
    arguments = inputs(sizes=(3, 6, 4), cells=(SKEWED, SHORT, CUBIC))
    output = build("periodic_distance", "both")._forward(*arguments)
    assert bool(torch.isfinite(output["pos_b"]).all())
    assert bool(torch.isfinite(output["cell_b"]).all())


def test_masked_species_still_works() -> None:
    """
    The discrete-species paths widen the node embedding after construction, and this encoder must not break that.

    Not used by the atomwise configuration, but silently breaking a baseline capability is how a package stops being a
    drop-in replacement.
    """
    arm = build("periodic_distance", "both")
    arm.enable_masked_species()
    assert arm.species_shift == 0
    arguments = list(inputs())
    arguments[0] = torch.zeros_like(arguments[0])
    assert bool(torch.isfinite(arm._forward(*arguments)["pos_b"]).all())


@pytest.mark.skipif(not BASELINE_CHECKPOINT.exists(), reason="no local atomwise checkpoint")
@pytest.mark.parametrize("mode", [mode.value for mode in FeatureMode])
def test_the_real_atomwise_checkpoint_loads_and_predicts_identically_on_the_fully_connected_graph(mode: str) -> None:
    """
    The production-shaped check the plan asks for, against the real 1,100-epoch atomwise weights.

    Every node-feature mode on the fully connected graph must reproduce the stock encoder's prediction on every output
    field, because the projection is zero. This is the strongest available statement that the baseline number this project
    already owns is still the baseline number after the encoder is swapped.
    """
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    state = baseline_encoder_state(checkpoint)
    torch.manual_seed(0)
    baseline = CSPNetFull()
    baseline.load_state_dict(state)
    torch.manual_seed(0)
    arm = DirectGeometryCSPNet(feature_mode=mode, message_graph="fc")
    arm.load_baseline_state_dict(state)

    # The checkpoint is single precision and ``load_state_dict`` casts into the existing double-precision parameters, so
    # both encoders run in double here. That makes "bitwise equal" a stronger statement than it would be at the production
    # dtype, where two orderings of the same sum can differ in the last bit for reasons that are not a code difference.
    arguments = inputs(sizes=(6, 5), cells=(CUBIC, SKEWED), latent=256)
    with torch.no_grad():
        assert same(baseline._forward(*arguments), arm._forward(*arguments))


@pytest.mark.skipif(not BASELINE_CHECKPOINT.exists(), reason="no local atomwise checkpoint")
def test_the_real_atomwise_checkpoint_loads_into_the_periodic_arm_with_zeroed_columns() -> None:
    """
    The widening is accounted for: the loaded periodic arm carries the baseline's weights and nothing else.

    Its prediction differs from the baseline's because the topology differs, which is the factor. What must not differ is
    any weight that both encoders have.
    """
    state = baseline_encoder_state(torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False))
    torch.manual_seed(0)
    arm = DirectGeometryCSPNet(feature_mode="both", message_graph="periodic_distance")
    arm.load_baseline_state_dict(state)
    own = arm.state_dict()
    for name, value in state.items():
        loaded = own[name]
        if value.shape == loaded.shape:
            assert torch.equal(value, loaded), name
        else:
            assert torch.equal(value, loaded[:, :value.shape[1]]), name
            assert bool((loaded[:, value.shape[1]:] == 0.0).all()), name
    assert bool((arm.geometry_projection.weight == 0.0).all())
