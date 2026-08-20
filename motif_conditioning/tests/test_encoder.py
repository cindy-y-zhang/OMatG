"""Tests for census conditioning, conditioning dropout, and classifier-free guidance."""

from __future__ import annotations

import pytest
import torch

from direct_geometry.encoder import DirectGeometryCSPNet
from motif_conditioning.data import CENSUS_FIELD
from motif_conditioning.encoder import ConditioningMask, MotifConditionedCSPNet

from .conftest import census_batch


DIMENSION = 8


def build(guidance_scale: float = 1.0, p_uncond: float = 0.2, **kwargs) -> MotifConditionedCSPNet:
    torch.manual_seed(0)
    return MotifConditionedCSPNet(
        census_dimension=DIMENSION,
        guidance_scale=guidance_scale,
        p_uncond=p_uncond,
        hidden_dim=32,
        num_layers=2,
        **kwargs,
    )


def time_for(batch) -> torch.Tensor:
    return torch.rand((len(batch.n_atoms), 256))


def test_construction_refuses_incoherent_settings() -> None:
    with pytest.raises(ValueError, match="census_dimension must be positive"):
        MotifConditionedCSPNet(census_dimension=0)
    with pytest.raises(ValueError, match="p_uncond must lie"):
        MotifConditionedCSPNet(census_dimension=4, p_uncond=1.0)
    with pytest.raises(ValueError, match="must not also recompute a per-site descriptor"):
        MotifConditionedCSPNet(census_dimension=4, feature_mode="radial")


def test_a_state_without_a_census_is_refused() -> None:
    model = build().eval()
    batch = census_batch(dimension=DIMENSION, with_census=False)
    with pytest.raises(ValueError, match="without a motif census"):
        model(batch, time_for(batch))


def test_a_mis_sized_census_is_refused() -> None:
    model = build().eval()
    batch = census_batch(dimension=DIMENSION + 1)
    with pytest.raises(ValueError, match="Expected a motif census of shape"):
        model(batch, time_for(batch))


def test_the_untrained_census_pathway_is_inert() -> None:
    """The zero-initialised mixin must leave the baseline's prediction untouched."""
    model = build().eval()
    batch = census_batch(dimension=DIMENSION)
    time = time_for(batch)

    torch.manual_seed(0)
    baseline = DirectGeometryCSPNet(feature_mode="none", message_graph="fc", hidden_dim=32,
                                    num_layers=2).eval()
    reference = baseline(batch, time)
    result = model(batch, time)

    assert torch.allclose(result.pos_b, reference.pos_b)
    assert torch.allclose(result.cell_b, reference.cell_b)


def test_a_warm_start_reproduces_the_baseline_exactly() -> None:
    torch.manual_seed(3)
    baseline = DirectGeometryCSPNet(feature_mode="none", message_graph="fc", hidden_dim=32,
                                    num_layers=2).eval()
    model = build().eval()
    model.load_baseline_state_dict(baseline.state_dict())

    batch = census_batch(dimension=DIMENSION)
    time = time_for(batch)
    assert torch.allclose(model(batch, time).pos_b, baseline(batch, time).pos_b)


def test_a_warm_start_still_refuses_an_unrelated_checkpoint() -> None:
    model = build().eval()
    state = model.state_dict()
    state["not_a_real_parameter"] = torch.zeros(1)
    with pytest.raises(ValueError, match="does not describe this encoder"):
        model.load_baseline_state_dict(state)


def activate(model: MotifConditionedCSPNet) -> None:
    """Give the adapters a non-zero mixin so conditioning can change the prediction."""
    with torch.no_grad():
        for adapter in model.adapters:
            adapter.mixin.weight.normal_(std=0.3)


def test_the_census_changes_the_prediction_once_the_pathway_is_active() -> None:
    model = build().eval()
    activate(model)
    batch = census_batch(dimension=DIMENSION)
    time = time_for(batch)

    first = model(batch, time).pos_b
    setattr(batch, CENSUS_FIELD, torch.randn_like(getattr(batch, CENSUS_FIELD)))
    second = model(batch, time).pos_b

    assert not torch.allclose(first, second)


def test_evaluation_is_deterministic() -> None:
    """No coin flip may reach inference, or paired comparisons stop being paired."""
    model = build().eval()
    activate(model)
    batch = census_batch(dimension=DIMENSION)
    time = time_for(batch)

    assert torch.equal(model(batch, time).pos_b, model(batch, time).pos_b)


def test_guidance_interpolates_between_the_two_branches() -> None:
    batch = census_batch(dimension=DIMENSION)
    time = time_for(batch)

    conditional = build(guidance_scale=1.0).eval()
    activate(conditional)
    reference_conditional = conditional(batch, time).pos_b

    unconditional = build(guidance_scale=0.0).eval()
    activate(unconditional)
    reference_unconditional = unconditional(batch, time).pos_b

    assert not torch.allclose(reference_conditional, reference_unconditional)

    amplified = build(guidance_scale=2.5).eval()
    activate(amplified)
    expected = reference_unconditional + 2.5 * (reference_conditional - reference_unconditional)
    assert torch.allclose(amplified(batch, time).pos_b, expected, atol=1e-10)


def test_the_unconditional_branch_matches_a_model_with_no_census_pathway() -> None:
    """Guidance scale zero must be exactly the baseline, not merely close to it."""
    model = build(guidance_scale=0.0).eval()
    activate(model)
    batch = census_batch(dimension=DIMENSION)
    time = time_for(batch)

    torch.manual_seed(0)
    baseline = DirectGeometryCSPNet(feature_mode="none", message_graph="fc", hidden_dim=32,
                                    num_layers=2).eval()
    assert torch.allclose(model(batch, time).pos_b, baseline(batch, time).pos_b)


def test_training_drops_conditioning_at_the_configured_rate() -> None:
    model = build(p_uncond=0.25).train()
    observed = []

    original = model._conditioning

    def record(mask):
        observed.append(mask.clone())
        return original(mask)

    model._conditioning = record
    batch = census_batch(sizes=tuple([3] * 64), dimension=DIMENSION)
    for _ in range(40):
        model(batch, time_for(batch))

    gates = torch.cat(observed)
    assert set(gates.unique().tolist()) <= {0.0, 1.0}
    assert 0.20 < float((gates == 0.0).double().mean()) < 0.30


def test_training_uses_one_gate_per_structure_not_per_atom() -> None:
    model = build().train()
    observed = []
    original = model._conditioning
    model._conditioning = lambda mask: (observed.append(mask.clone()), original(mask))[1]

    batch = census_batch(sizes=(4, 6, 5), dimension=DIMENSION)
    model(batch, time_for(batch))

    assert observed[0].shape == (3,)


def test_the_conditioning_context_restores_the_previous_gate() -> None:
    mask = ConditioningMask()
    with pytest.raises(RuntimeError, match="outside a conditioning context"):
        mask.require()

    model = build()
    outer = torch.ones(2)
    with model._conditioning(outer):
        with model._conditioning(torch.zeros(2)):
            assert torch.equal(model._mask.require(), torch.zeros(2))
        assert torch.equal(model._mask.require(), outer)
    assert model._mask.value is None


def test_the_conditioning_residual_is_scaled_to_the_features_it_modifies() -> None:
    """The fix for the measured scale mismatch: the residual is in units of feature RMS."""
    model = build().eval()
    adapter = model.adapters[0]
    with torch.no_grad():
        adapter.mixin.weight.normal_(std=0.3)

    features = torch.randn((7, model.hidden_dim)) * 50.0
    embedding = torch.randn((2, 32))
    counts = torch.tensor([3, 4])
    with model._conditioning(torch.ones(2)):
        updated = adapter(features, embedding, None, counts)

    units = features.square().mean(dim=-1, keepdim=True).sqrt()
    expected = adapter.residual(embedding).repeat_interleave(counts, dim=0)
    assert torch.allclose(updated - features, units * expected)


@pytest.mark.parametrize("feature_rms", [0.5, 10.0, 97.0])
def test_relative_scaling_amplifies_by_exactly_the_feature_scale(feature_rms: float) -> None:
    """The measured trunk sits near an RMS of 97, so this is the factor the fix recovers."""
    relative = build().eval()
    with torch.no_grad():
        for adapter in relative.adapters:
            adapter.mixin.weight.normal_(std=0.3)
    absolute = build(relative_scaling=False).eval()
    absolute.load_state_dict(relative.state_dict())

    features = torch.full((5, relative.hidden_dim), feature_rms)
    embedding = torch.randn((2, 32))
    counts = torch.tensor([2, 3])
    with torch.no_grad(), relative._conditioning(torch.ones(2)):
        scaled = relative.adapters[0](features, embedding, None, counts) - features
    with torch.no_grad(), absolute._conditioning(torch.ones(2)):
        plain = absolute.adapters[0](features, embedding, None, counts) - features

    assert torch.allclose(scaled, feature_rms * plain)


def test_the_absolute_form_is_still_available_for_reproduction() -> None:
    model = build(relative_scaling=False).eval()
    assert not model.relative_scaling
    assert all(not adapter.relative_scaling for adapter in model.adapters)


def test_scaling_leaves_the_untrained_pathway_inert() -> None:
    """Zero-initialised mixins must still reproduce the baseline under either form."""
    batch = census_batch(dimension=DIMENSION)
    time = time_for(batch)
    torch.manual_seed(0)
    baseline = DirectGeometryCSPNet(feature_mode="none", message_graph="fc", hidden_dim=32,
                                    num_layers=2).eval()
    for scaling in (True, False):
        model = build(relative_scaling=scaling).eval()
        assert torch.allclose(model(batch, time).pos_b, baseline(batch, time).pos_b)
