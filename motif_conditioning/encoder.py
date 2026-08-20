"""Encoder that conditions on a global motif census instead of diffusing one."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from direct_geometry.encoder import DirectGeometryCSPNet
from omg.model.model_utils import AdapterModule

from .data import CENSUS_FIELD


CENSUS_PARAMETER_PREFIXES = ("census_embedding.", "adapters.")
"""Parameters that a baseline checkpoint cannot describe, because it has no census."""


class ConditioningMask:
    """Per-structure gate on the adapter residual, shared by every layer's adapter.

    ``AdapterModule`` draws its own Bernoulli gate, which is right for training and wrong
    for inference: it would randomise every generated structure and make the conditional
    and unconditional passes of classifier-free guidance indistinguishable. The mask lives
    outside the adapters so the encoder can set one gate for a whole forward pass.
    """

    def __init__(self) -> None:
        self.value: Optional[torch.Tensor] = None

    def require(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("A census adapter ran outside a conditioning context.")
        return self.value


class CensusAdapter(AdapterModule):
    """Adapter gated by the owning encoder and scaled to the features it modifies.

    The stock adapter adds an absolute residual. That fails here for a measured reason: a
    warm-started trunk carries node features of norm ~2200, a zero-initialised mixin starts
    at 0, and the gradient reaching the mixin is proportional to how much its output already
    changes the prediction. After 2000 steps the residual had reached 0.03 per cent of the
    feature scale and the prediction moved by 0.1 per cent when the conditioning was removed
    outright -- weak enough that the pathway could never bootstrap itself out.
    See ``motif_conditioning/reports/CONDITIONING-PROBE.json``.

    Expressing the residual in units of the features' own root-mean-square element removes
    the mismatch: the conditioning starts at zero exactly as before, so a warm start is
    still reproduced bit for bit, but a given change in the mixin now moves the prediction
    by about two orders of magnitude more, and the gradient that returns is larger by the
    same factor. The scale is detached, so it sets the units of the residual without letting
    the conditioning reshape the trunk's magnitudes.
    """

    def __init__(self, mask: ConditioningMask, relative_scaling: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mask = mask
        self.relative_scaling = bool(relative_scaling)

    def residual(self, property_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.adapter_relu(self.adapter_fc1(property_embedding))
        hidden = self.adapter_relu(self.adapter_fc2(hidden))
        return self.mixin(hidden)

    def forward(self, node_features, property_embedding, property_indicator, num_atoms):
        gated = self._mask.require().view(-1, 1) * self.residual(property_embedding)
        per_atom = gated.repeat_interleave(num_atoms, dim=0)
        if not self.relative_scaling:
            return node_features + per_atom
        units = node_features.detach().square().mean(dim=-1, keepdim=True).sqrt()
        return node_features + units * per_atom


class MotifConditionedCSPNet(DirectGeometryCSPNet):
    """The D-baseline backbone plus static per-structure motif conditioning.

    The census enters through the existing per-layer adapters as a residual on node
    features. Unlike a jointly diffused geometry state it is constant along the whole
    path, so the model never has to read it through a prior.

    :param census_dimension:
        Width of the per-structure census vector.
    :type census_dimension: int
    :param census_embed_dim:
        Width of the embedding the adapters consume.
        Defaults to 32, matching the baseline adapter's property width.
    :type census_embed_dim: int
    :param p_uncond:
        Fraction of training structures whose conditioning is dropped, which is what makes
        the unconditional branch of classifier-free guidance a trained path rather than an
        extrapolation.
        Defaults to 0.2.
    :type p_uncond: float
    :param guidance_scale:
        Inference-time weight on the conditional direction. One reproduces the plain
        conditional model, zero the unconditional one, and larger values amplify the
        census. Ignored while training.
        Defaults to 1.0.
    :type guidance_scale: float
    :param relative_scaling:
        Whether the conditioning residual is expressed in units of the node features' own
        scale rather than in absolute units. See :class:`CensusAdapter` for the measurement
        that motivates it.
        Defaults to True. False reproduces the first, diagnosed-null arm.
    :type relative_scaling: bool
    """

    def __init__(
        self,
        census_dimension: int,
        census_embed_dim: int = 32,
        am_hidden_dim: int = 128,
        p_uncond: float = 0.2,
        guidance_scale: float = 1.0,
        relative_scaling: bool = True,
        feature_mode: str = "none",
        **kwargs: Any,
    ) -> None:
        if census_dimension <= 0:
            raise ValueError("census_dimension must be positive.")
        if not 0.0 <= p_uncond < 1.0:
            raise ValueError(f"p_uncond must lie in [0, 1), got {p_uncond}.")
        if feature_mode != "none":
            raise ValueError(
                "The census arm must not also recompute a per-site descriptor; "
                "that factor belongs to the separate direct-geometry study."
            )
        # ``prop=False`` so the parent builds no adapters of its own; the ones below differ
        # only in taking their gate from this encoder rather than from a fresh coin flip.
        super().__init__(feature_mode=feature_mode, prop=False, **kwargs)
        self.census_dimension = int(census_dimension)
        self.p_uncond = float(p_uncond)
        self.guidance_scale = float(guidance_scale)

        self.census_embedding = nn.Sequential(
            nn.Linear(self.census_dimension, census_embed_dim),
            nn.SiLU(),
            nn.Linear(census_embed_dim, census_embed_dim),
        )
        self._mask = ConditioningMask()
        self.relative_scaling = bool(relative_scaling)
        self.adapters = nn.ModuleList(
            CensusAdapter(
                self._mask,
                relative_scaling=self.relative_scaling,
                input_dim=self.hidden_dim,
                am_hidden_dim=am_hidden_dim,
                property_dim=census_embed_dim,
            )
            for _ in range(self.num_layers)
        )

    @contextmanager
    def _conditioning(self, mask: torch.Tensor) -> Iterator[None]:
        previous = self._mask.value
        self._mask.value = mask
        try:
            yield
        finally:
            self._mask.value = previous

    def _census(self, x: Data) -> torch.Tensor:
        if not hasattr(x, CENSUS_FIELD):
            raise ValueError("MotifConditionedCSPNet received a state without a motif census.")
        census = getattr(x, CENSUS_FIELD)
        expected = (len(x.n_atoms), self.census_dimension)
        if tuple(census.shape) != expected:
            raise ValueError(
                f"Expected a motif census of shape {expected}, got {tuple(census.shape)}."
            )
        return census

    def load_baseline_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Warm-start from a baseline encoder, leaving the census pathway at its initialisation.

        The adapters' mixin layers are zero-initialised, so a warm-started census arm
        reproduces the baseline exactly and has to learn that the census is worth reading.
        Every parameter the baseline does have is still matched strictly by the parent.
        """
        prepared = dict(state_dict)
        for name, value in self.state_dict().items():
            if name.startswith(CENSUS_PARAMETER_PREFIXES) and name not in prepared:
                prepared[name] = value.clone()
        super().load_baseline_state_dict(prepared)

    def forward(self, x: Data, t: torch.Tensor, prop: Optional[torch.Tensor] = None, **kwargs):
        census = self._census(x)
        embedded = self.census_embedding(census.to(self.census_embedding[0].weight.dtype))
        structures = census.shape[0]

        if self.training:
            gate = torch.bernoulli(
                torch.full((structures,), 1.0 - self.p_uncond, device=census.device)
            )
            with self._conditioning(gate):
                return super().forward(x, t, embedded, **kwargs)

        ones = torch.ones(structures, device=census.device)
        with self._conditioning(ones):
            conditional = super().forward(x, t, embedded, **kwargs)
        if self.guidance_scale == 1.0:
            return conditional
        with self._conditioning(torch.zeros(structures, device=census.device)):
            unconditional = super().forward(x, t, embedded, **kwargs)
        return _guide(conditional, unconditional, self.guidance_scale)


def _guide(conditional: Data, unconditional: Data, scale: float) -> Data:
    """Combine two predictions along the direction the conditioning adds."""
    guided = {}
    for key, value in conditional.to_dict().items():
        other = unconditional[key]
        guided[key] = (
            other + scale * (value - other)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value
        )
    return Data(**guided)
