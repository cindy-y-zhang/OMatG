"""Tests for the coupled CN / SE(3) / lattice interpolant."""

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from cgfm.block_sampler import BlockSampler
from cgfm.block_si import CoupledBlockInterpolants, DiscreteFlowMatchingMaskN, consensus_positions
from cgfm.blocks import CN_CLASSES
from cgfm.so3 import geodesic, sample_uniform, vertex_distance
from cgfm.tests.test_blockdata import tiny_library, record_structure
from cgfm.tests.test_readout import CHAIN, CHAIN_NUMBERS, chain_blocks, hand_built
from cgfm.readout import orphan_free
from cgfm.blockdata import _to_block_data
from omg.sampler.cell_distributions import InformedLatticeDistribution, MirrorCell
from omg.sampler.position_distributions import UniformPositionDistribution
from omg.globals import SMALL_TIME


def _batch_from_chain() -> Batch:
    """Build a one-structure block batch from the hand-built chain."""
    repaired = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, chain_blocks()))
    record = record_structure(repaired, tiny_library())
    return Batch.from_data_list([_to_block_data(record)])


def test_discrete_flow_has_thirteen_classes():
    interpolant = DiscreteFlowMatchingMaskN()
    x_0 = torch.zeros(5, dtype=torch.long)
    x_1 = torch.arange(1, 6)
    t = torch.full((5,), 0.5)
    x_t, _ = interpolant.interpolate(t, x_0, x_1, torch.zeros(5, dtype=torch.long))
    logits = torch.randn(5, CN_CLASSES)
    loss = interpolant.loss(lambda _x: (logits, logits), t, x_0, x_1, x_t, torch.zeros(5), torch.zeros(5, dtype=torch.long))
    assert loss["loss"].ndim == 0


def test_discrete_flow_unmasks_every_token_on_the_final_step():
    interpolant = DiscreteFlowMatchingMaskN()
    logits = torch.full((6, CN_CLASSES), -100.0)
    logits[:, 4] = 100.0
    integrated = interpolant.integrate(
        lambda _time, _x: (logits, logits), torch.zeros(6, dtype=torch.long),
        torch.tensor(0.9), torch.tensor(0.1), torch.zeros(6, dtype=torch.long), final_step=True)
    assert torch.equal(integrated, torch.full((6,), 5, dtype=torch.long))


def test_oracle_sampler_copies_cn_and_joint_masks_it():
    x_1 = _batch_from_chain()
    torch.manual_seed(0)
    np.random.seed(0)
    oracle = BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode="oracle").sample_p_0(x_1)
    torch.manual_seed(0)
    np.random.seed(0)
    joint = BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode="joint").sample_p_0(x_1)
    assert torch.equal(oracle.block_type, x_1.block_type)
    assert torch.equal(joint.block_type, torch.zeros_like(x_1.block_type))
    assert torch.allclose(oracle.pos, joint.pos)
    assert torch.allclose(oracle.rot, joint.rot)
    assert torch.allclose(oracle.cell, joint.cell)


def test_block_sampler_uses_batched_cell_sampling_when_available():
    one = _batch_from_chain().to_data_list()[0]
    x_1 = Batch.from_data_list([one, one.clone()])

    class BatchedCells:
        def __init__(self):
            self.calls = 0

        def __call__(self, _cell):
            raise AssertionError("The scalar cell sampler should not be used.")

        def sample_batch(self, cells):
            self.calls += 1
            return 2.0 * cells

    cells = BatchedCells()
    sampled = BlockSampler(UniformPositionDistribution(), cells, cn_mode="oracle").sample_p_0(x_1)
    assert cells.calls == 1
    assert torch.allclose(sampled.cell, 2.0 * x_1.cell)


def test_informed_lattice_batch_sampler_returns_valid_cells():
    cells = torch.eye(3).repeat(32, 1, 1)
    sampled = InformedLatticeDistribution("mpts_52").sample_batch(cells)
    assert sampled.shape == cells.shape
    assert torch.all(torch.linalg.det(sampled) > 0.0)


def test_online_canonicalisation_picks_the_nearest_stabiliser_element():
    si = CoupledBlockInterpolants(cn_mode="oracle", integration_time_steps=4)
    x_1 = _batch_from_chain()
    x_0 = x_1.clone()
    x_0.rot = sample_uniform(int(x_1.rot.shape[0]), dtype=x_1.rot.dtype)
    x_t = si._interpolate(torch.zeros(len(x_1.n_atoms)), x_0, x_1)
    assert x_t.rot_target.shape == x_1.rot.shape
    distance_raw = vertex_distance(x_0.rot, x_1.rot, x_1.template_offsets, x_1.template_mask)
    distance_canon = vertex_distance(x_0.rot, x_t.rot_target, x_1.template_offsets, x_1.template_mask)
    assert torch.all(distance_canon <= distance_raw + 1e-8)


def test_geodesic_interpolation_starts_at_the_base_orientation():
    si = CoupledBlockInterpolants(cn_mode="oracle", integration_time_steps=4)
    x_1 = _batch_from_chain()
    x_0 = x_1.clone()
    x_0.rot = sample_uniform(int(x_1.rot.shape[0]), dtype=x_1.rot.dtype)
    x_t = si._interpolate(torch.zeros(len(x_1.n_atoms)) + SMALL_TIME, x_0, x_1)
    assert torch.allclose(geodesic(x_0.rot, x_t.rot_target, torch.full((x_0.rot.shape[0],), SMALL_TIME)),
                          x_t.rot, atol=1e-7)


def test_loss_keys_match_the_cost_dictionary():
    from omg.si.stochastic_interpolants import StochasticInterpolants
    si = CoupledBlockInterpolants(cn_mode="joint")
    assert isinstance(si, StochasticInterpolants)
    assert set(si.loss_keys()) == {"species_loss", "block_type_loss", "pos_loss_b", "rot_loss_b", "cell_loss_b"}
    oracle = CoupledBlockInterpolants(cn_mode="oracle")
    assert set(oracle.loss_keys()) == set(si.loss_keys())


def test_velocity_losses_are_minimised_by_the_full_target_velocity():
    """A missing factor of two would train half-speed fields and stop integration halfway."""
    si = CoupledBlockInterpolants(cn_mode="oracle")
    x_1 = _batch_from_chain()
    x_0 = x_1.clone()
    delta = torch.linspace(-0.1, 0.1, len(x_1.pos), dtype=x_1.pos.dtype)[:, None]
    delta = delta * torch.tensor([[1.0, -0.5, 0.25]], dtype=x_1.pos.dtype)
    delta = delta - delta.mean(dim=0, keepdim=True)
    x_0.pos = x_1.pos - delta
    x_0.cell = x_1.cell - 0.2 * torch.eye(3, dtype=x_1.cell.dtype).unsqueeze(0)
    t = torch.full((len(x_1.n_atoms),), 0.5, dtype=x_1.pos.dtype)
    expected_pos = x_1.pos - x_0.pos
    expected_pos = expected_pos - expected_pos.mean(dim=0, keepdim=True)
    expected_cell = x_1.cell - x_0.cell

    def model(scale):
        return lambda x_t, _time: Data(
            pos_b=scale * expected_pos,
            rot_b=torch.zeros((len(x_t.pos), 3), dtype=x_t.pos.dtype),
            cell_b=scale * expected_cell,
            block_type_b=torch.zeros((len(x_t.pos), CN_CLASSES), dtype=x_t.pos.dtype))

    full = si.losses(model(1.0), t, x_0, x_1)
    half = si.losses(model(0.5), t, x_0, x_1)
    assert full["pos_loss_b"] < half["pos_loss_b"]
    assert full["cell_loss_b"] < half["cell_loss_b"]


def test_consensus_positions_average_two_votes_for_one_atom():
    frac = torch.tensor([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    rotations = torch.eye(3).expand(2, 3, 3).clone()
    cell = torch.eye(3).unsqueeze(0) * 10.0
    offsets = torch.zeros(2, 12, 3)
    offsets[0, 0] = torch.tensor([3.0, 0.0, 0.0])
    offsets[1, 0] = torch.tensor([-3.0, 0.0, 0.0])
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, 0] = True
    vote = torch.full((2, 12), -1, dtype=torch.long)
    vote[:, 0] = 2
    target_frac = torch.zeros(1, 52, 3)
    target_frac[0, :3] = torch.tensor([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5], [0.5, 0.5, 0.5]])
    target_mask = torch.zeros(1, 52, dtype=torch.bool)
    target_mask[0, :3] = True
    predicted, true = consensus_positions(frac, rotations, cell, offsets, mask, vote, torch.tensor([0, 1]),
                                          target_frac, target_mask, torch.tensor([3]),
                                          torch.zeros(2, dtype=torch.long))
    assert predicted.shape == true.shape == (3, 3)
    assert torch.allclose(predicted, true)
    assert torch.allclose(predicted[2], torch.tensor([5.0, 5.0, 5.0]))


def test_consensus_positions_uses_minimum_image_residuals():
    """A vote crossing a cell face must agree with its target rather than average across the cell."""
    frac = torch.tensor([[0.99, 0.5, 0.5]], dtype=torch.float64)
    rotations = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0
    offsets = torch.zeros(1, 12, 3, dtype=torch.float64)
    offsets[0, 0] = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, 0] = True
    vote = torch.full((1, 12), -1, dtype=torch.long)
    vote[0, 0] = 1
    target_frac = torch.zeros(1, 52, 3, dtype=torch.float64)
    target_frac[0, 0] = frac[0]
    target_frac[0, 1] = torch.tensor([0.01, 0.5, 0.5], dtype=torch.float64)
    target_mask = torch.zeros(1, 52, dtype=torch.bool)
    target_mask[0, :2] = True

    predicted, true = consensus_positions(
        frac, rotations, cell, offsets, mask, vote, torch.tensor([0]), target_frac, target_mask,
        torch.tensor([2]), torch.tensor([0]))

    assert torch.allclose(predicted, true, atol=1.0e-10)


def test_integrator_uses_the_configured_number_of_network_evaluations():
    """integration_time_steps denotes Euler steps, not time-grid points."""
    x_0 = _batch_from_chain()
    calls = 0

    def zero_model(x_t, _time):
        nonlocal calls
        calls += 1
        return Data(
            pos_b=torch.zeros_like(x_t.pos),
            rot_b=torch.zeros((len(x_t.pos), 3), dtype=x_t.pos.dtype),
            cell_b=torch.zeros_like(x_t.cell),
            block_type_b=torch.zeros((len(x_t.pos), CN_CLASSES), dtype=x_t.pos.dtype))

    steps = 4
    CoupledBlockInterpolants(
        cn_mode="oracle", integration_time_steps=steps, enable_progress_bar=False).integrate(x_0, zero_model)
    assert calls == steps
