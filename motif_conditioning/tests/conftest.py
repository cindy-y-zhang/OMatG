"""Shared fixtures for the motif-conditioning tests."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from motif_conditioning.data import CENSUS_FIELD


@pytest.fixture(autouse=True)
def deterministic_environment():
    """Run every test in double precision from a fixed global random state."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


def census_batch(
    sizes: tuple[int, ...] = (4, 6, 5),
    dimension: int = 8,
    seed: int = 0,
    with_census: bool = True,
) -> Batch:
    """A small batched crystal state carrying one census per structure."""
    generator = torch.Generator().manual_seed(seed)
    graphs = []
    for index, count in enumerate(sizes):
        fields = {
            "n_atoms": torch.tensor(count),
            "species": torch.randint(1, 20, (count,), generator=generator),
            "pos": torch.rand((count, 3), generator=generator),
            "cell": (torch.eye(3) * (4.0 + 0.6 * index)).unsqueeze(0),
            "pos_is_fractional": torch.tensor(True),
        }
        if with_census:
            fields[CENSUS_FIELD] = torch.randn((1, dimension), generator=generator)
        graphs.append(Data(**fields))
    return Batch.from_data_list(graphs)
