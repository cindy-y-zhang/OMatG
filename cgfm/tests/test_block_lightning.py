"""Tests for readout-aware block training, endpoint integration and oracle/joint identity."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from cgfm.block_encoder import BlockCSPNet
from cgfm.block_lightning import _read_out_structure
from cgfm.block_sampler import BlockSampler
from cgfm.block_si import CoupledBlockInterpolants
from cgfm.tests.test_blockdata import tiny_library
from cgfm.tests.test_block_si import _batch_from_chain
from omg.model.heads.pass_through import PassThrough
from omg.model.model import Model
from omg.model.model_utils import SinusoidalTimeEmbeddings
from omg.sampler.cell_distributions import MirrorCell
from omg.sampler.position_distributions import UniformPositionDistribution


def _tiny_model() -> Model:
    """Build a tiny BlockCSPNet wrapped as an OMatG Model."""
    encoder = BlockCSPNet(hidden_dim=32, latent_dim=16, num_layers=1, num_freqs=8, am_hidden_dim=8)
    return Model(encoder=encoder, head=PassThrough(), time_embedder=SinusoidalTimeEmbeddings(16))


def test_training_loss_is_finite_for_oracle_and_joint():
    x_1 = _batch_from_chain()
    model = _tiny_model()
    for mode in ("oracle", "joint"):
        sampler = BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode=mode)
        si = CoupledBlockInterpolants(cn_mode=mode, integration_time_steps=3)
        x_0 = sampler.sample_p_0(x_1)
        t = torch.full((len(x_1.n_atoms),), 0.4)
        losses = si.losses(model, t, x_0, x_1)
        assert set(losses) == set(si.loss_keys())
        for key, value in losses.items():
            assert torch.isfinite(value)


def test_endpoint_integration_preserves_shapes():
    x_1 = _batch_from_chain()
    model = _tiny_model()
    sampler = BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode="oracle")
    si = CoupledBlockInterpolants(cn_mode="oracle", integration_time_steps=4, enable_progress_bar=False)
    generated = si.integrate(sampler.sample_p_0(x_1), model)
    assert generated.rot.shape == x_1.rot.shape
    assert generated.pos.shape == x_1.pos.shape
    assert generated.block_type.shape == x_1.block_type.shape
    assert generated.cell.shape == x_1.cell.shape


def test_readout_from_oracle_poses_recovers_the_chain_composition():
    x_1 = _batch_from_chain()
    library = tiny_library()
    reference, generated, diagnostics = _read_out_structure(x_1, x_1, 0, library)
    assert len(generated) == len(reference)
    assert sorted(generated.get_atomic_numbers().tolist()) == sorted(reference.get_atomic_numbers().tolist())
    assert diagnostics["shortfall"] >= 0


def test_readout_can_skip_matching_when_the_caller_does_not_use_it():
    """Prediction must not spend one StructureMatcher call per generated structure."""
    x_1 = _batch_from_chain()
    _, _, diagnostics = _read_out_structure(x_1, x_1, 0, tiny_library(), compute_match=False)
    assert "matched" not in diagnostics


def test_readout_rejects_a_coordination_token_left_masked_by_integration():
    generated = _batch_from_chain()
    generated.block_type[0] = 0
    with pytest.raises(RuntimeError, match="masked coordination token"):
        _read_out_structure(generated, _batch_from_chain(), 0, tiny_library(), compute_match=False)


def test_reshape_t_handles_rot_and_block_type():
    from omg.utils import DataField, reshape_t
    t = torch.tensor([0.2, 0.8])
    n_atoms = torch.tensor([2, 3])
    assert reshape_t(t, n_atoms, DataField.block_type).shape == (5,)
    assert reshape_t(t, n_atoms, DataField.rot).shape == (5, 3, 3)
    assert reshape_t(t, n_atoms, DataField.species).shape == (5,)


def _protocol_costs() -> dict[str, float]:
    """Load the relative SI costs locked in the shared block protocol yaml."""
    from pathlib import Path
    import yaml
    config = yaml.safe_load(Path("cgfm/configs/block_mpts52.yaml").read_text())
    return dict(config["model"]["relative_si_costs"])


def test_block_lightning_constructs_with_protocol_costs():
    from cgfm.block_lightning import BlockLightning
    costs = _protocol_costs()
    assert abs(sum(costs.values()) - 1.0) < 1e-10
    for mode in ("oracle", "joint"):
        module = BlockLightning(
            si=CoupledBlockInterpolants(cn_mode=mode, integration_time_steps=3, enable_progress_bar=False),
            sampler=BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode=mode),
            model=_tiny_model(),
            relative_si_costs=costs,
            validation_mode="match_rate",
            dataset_name="mpts_52",
            number_cpus=1,
        )
        assert module.consensus_weight == 0.0
        assert set(module.si.loss_keys()) == set(costs)


def test_consensus_endpoint_loss_is_finite_and_differentiable():
    from cgfm.block_lightning import BlockLightning
    x_1 = _batch_from_chain()
    sampler = BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode="oracle")
    module = BlockLightning(
        si=CoupledBlockInterpolants(cn_mode="oracle", integration_time_steps=3, enable_progress_bar=False),
        sampler=sampler, model=_tiny_model(), relative_si_costs=_protocol_costs(), consensus_weight=1.0,
        validation_mode="match_rate", dataset_name="mpts_52", number_cpus=1)
    x_0 = sampler.sample_p_0(x_1)
    loss = module._consensus_loss(x_0, x_1, torch.tensor([0.4]))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in module.model.parameters())


def test_consensus_training_reuses_the_field_loss_network_evaluation():
    """The endpoint objective must not double the expensive encoder work per training batch."""
    from cgfm.block_lightning import BlockLightning
    module = BlockLightning(
        si=CoupledBlockInterpolants(cn_mode="oracle", integration_time_steps=3, enable_progress_bar=False),
        sampler=BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode="oracle"),
        model=_tiny_model(), relative_si_costs=_protocol_costs(), consensus_weight=0.1,
        validation_mode="match_rate", dataset_name="mpts_52", number_cpus=1)
    calls = 0
    original_forward = module.model.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    module.model.forward = counted_forward
    module.log_dict = lambda *args, **kwargs: None
    loss = module.training_step(_batch_from_chain())
    assert torch.isfinite(loss)
    assert calls == 1


def test_block_protocol_yaml_matches_the_locked_schedule():
    from pathlib import Path
    import yaml
    shared = yaml.safe_load(Path("cgfm/configs/block_mpts52.yaml").read_text())
    atomwise = yaml.safe_load(Path("cgfm/configs/atomwise_mpts52.yaml").read_text())
    oracle = yaml.safe_load(Path("cgfm/configs/block_oracle_mpts52.yaml").read_text())
    joint = yaml.safe_load(Path("cgfm/configs/block_joint_mpts52.yaml").read_text())
    assert shared["trainer"]["max_epochs"] == atomwise["trainer"]["max_epochs"] == 400
    assert shared["trainer"]["max_time"]["hours"] == 28
    assert shared["data"]["batch_size"] == atomwise["data"]["batch_size"] == 256
    for option, expected in (("pin_memory", True), ("persistent_workers", True), ("prefetch_factor", 4)):
        assert shared["data"][option] == atomwise["data"][option] == expected
    assert shared["trainer"]["accumulate_grad_batches"] == atomwise["trainer"]["accumulate_grad_batches"] == 4
    assert shared["model"]["si"]["init_args"]["integration_time_steps"] == 210
    assert shared["trainer"]["check_val_every_n_epoch"] == atomwise["trainer"]["check_val_every_n_epoch"] == 25
    assert oracle["model"]["si"]["init_args"]["cn_mode"] == "oracle"
    assert joint["model"]["si"]["init_args"]["cn_mode"] == "joint"
