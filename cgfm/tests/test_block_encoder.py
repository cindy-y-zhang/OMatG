"""Tests for BlockCSPNet input embeddings and CN masking."""

import torch
from torch_geometric.data import Batch
from cgfm.block_encoder import BlockCSPNet
from cgfm.blocks import CN_CLASSES
from cgfm.tests.test_block_si import _batch_from_chain
from omg.model.model_utils import SinusoidalTimeEmbeddings


def test_block_encoder_emits_translation_rotation_cell_and_cn():
    batch = _batch_from_chain()
    encoder = BlockCSPNet(hidden_dim=32, latent_dim=16, num_layers=1, num_freqs=8, am_hidden_dim=8)
    times = SinusoidalTimeEmbeddings(16)(torch.rand(len(batch.n_atoms)))
    prediction = encoder(batch, times)
    n_blocks = int(batch.n_atoms.sum())
    assert prediction.pos_b.shape == (n_blocks, 3)
    assert prediction.rot_b.shape == (n_blocks, 3)
    assert prediction.cell_b.shape == (len(batch.n_atoms), 3, 3)
    assert prediction.block_type_b.shape == (n_blocks, CN_CLASSES)


def test_block_encoder_masks_impossible_cn_logits():
    batch = _batch_from_chain()
    encoder = BlockCSPNet(hidden_dim=32, latent_dim=16, num_layers=1, num_freqs=8, am_hidden_dim=8)
    mask = torch.zeros_like(encoder.cn_valid)
    mask[:, 2] = True
    encoder.set_cn_mask(mask)
    times = SinusoidalTimeEmbeddings(16)(torch.rand(len(batch.n_atoms)))
    logits = encoder(batch, times).block_type_b
    assert (logits[:, 0] <= -1.0e8).all()
    assert torch.isfinite(logits[:, 2]).all()


def test_block_encoder_reuses_fully_connected_edges_during_integration():
    """Euler steps change edge displacements but not fully connected topology."""
    batch = _batch_from_chain()
    encoder = BlockCSPNet(hidden_dim=32, latent_dim=16, num_layers=1, num_freqs=8, am_hidden_dim=8)
    times = SinusoidalTimeEmbeddings(16)(torch.rand(len(batch.n_atoms)))
    original = encoder.gen_edges
    calls = 0

    def counted_gen_edges(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    encoder.gen_edges = counted_gen_edges
    encoder(batch, times)
    batch.pos = torch.remainder(batch.pos + 0.1, 1.0)
    encoder(batch, times)
    assert calls == 1
