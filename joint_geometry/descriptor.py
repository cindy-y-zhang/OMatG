"""
Compact, continuous endpoint descriptors for the joint geometry state.

The raw measurement is the species-blind radial block already audited in
``direct_geometry``: sixteen smooth, normalised shell occupancies and one
``log1p`` soft-coordination channel, evaluated over a complete periodic
neighbour list.  The train-only transform keeps the coordination channel
explicit and whitens a small PCA of the shell profile.  If compression loses
too much information, the same class can standardise all seventeen channels
without changing the rest of the joint-state implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import pickle

import numpy as np
import torch
from sklearn.decomposition import PCA

from direct_geometry.features import DescriptorSpec, radial_block
from direct_geometry.neighbors import periodic_neighbors


DEFAULT_REPRESENTATION = "cn-rdf4"
"""Smallest representation tested by the compression gate."""

REPRESENTATION_DIMENSIONS = {
    "cn-rdf4": 4,
    "cn-rdf8": 8,
    "radial17": 17,
}
"""Supported nested representations and their per-site widths."""

MINIMUM_SCALE = 1.0e-8
"""Smallest standard deviation accepted when normalising a channel."""


@dataclass(frozen=True)
class JointGeometryDescriptorSettings:
    """Settings that completely determine a raw endpoint descriptor."""

    version: int = 1
    cutoff: float = 6.0
    shells: int = 16
    representation: str = DEFAULT_REPRESENTATION
    species_blind: bool = True
    periodic_images: str = "complete"

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATION_DIMENSIONS:
            raise ValueError(
                f"Unknown representation {self.representation!r}; expected one of "
                f"{tuple(REPRESENTATION_DIMENSIONS)}."
            )
        DescriptorSpec(cutoff=self.cutoff, num_shells=self.shells)
        if self.shells != 16:
            raise ValueError(
                "CN-RDF compression is defined for the audited 16-shell radial block; "
                f"got {self.shells} shells."
            )
        if not self.species_blind:
            raise ValueError("The joint geometry descriptor must remain species-blind.")
        if self.periodic_images != "complete":
            raise ValueError("The descriptor requires complete periodic-image enumeration.")

    @property
    def raw_dimension(self) -> int:
        """Width of the uncompressed radial block."""
        return self.shells + 1

    @property
    def dimension(self) -> int:
        """Width of the transformed joint state."""
        return REPRESENTATION_DIMENSIONS[self.representation]

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serialisable settings dictionary."""
        return asdict(self)

    def feature_spec(self) -> DescriptorSpec:
        """Return the existing descriptor implementation's matching spec."""
        return DescriptorSpec(cutoff=self.cutoff, num_shells=self.shells)


@dataclass
class DescriptorTransform:
    """
    Train-only standardisation and optional PCA for the radial descriptor.

    For CN-RDF4/8 the first output is always the standardised soft
    coordination and the remaining outputs are whitened components of the
    normalised shell profile.  Keeping those parts separate prevents PCA from
    spending all of a compact representation on the high-variance count.
    """

    settings: JointGeometryDescriptorSettings
    coordination_mean: float
    coordination_scale: float
    radial_mean: np.ndarray
    radial_scale: np.ndarray
    pca: PCA | None = None

    @classmethod
    def fit(
        cls,
        raw: np.ndarray,
        settings: JointGeometryDescriptorSettings,
        seed: int = 0,
    ) -> "DescriptorTransform":
        """Fit a transform using training atoms only."""
        values = _validated_raw(raw, settings)
        if settings.representation == "radial17":
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
            scale = np.maximum(scale, MINIMUM_SCALE)
            return cls(
                settings=settings,
                coordination_mean=float(mean[-1]),
                coordination_scale=float(scale[-1]),
                radial_mean=mean.astype(np.float64),
                radial_scale=scale.astype(np.float64),
                pca=None,
            )

        coordination = values[:, -1]
        coordination_mean = float(coordination.mean())
        coordination_scale = max(float(coordination.std()), MINIMUM_SCALE)
        components = settings.dimension - 1
        pca = PCA(n_components=components, whiten=True, random_state=seed)
        pca.fit(values[:, :-1])
        return cls(
            settings=settings,
            coordination_mean=coordination_mean,
            coordination_scale=coordination_scale,
            radial_mean=np.asarray(pca.mean_, dtype=np.float64),
            radial_scale=np.sqrt(
                np.maximum(np.asarray(pca.explained_variance_, dtype=np.float64), MINIMUM_SCALE**2)
            ),
            pca=pca,
        )

    @property
    def dimension(self) -> int:
        """Width of the transformed descriptor."""
        return self.settings.dimension

    @property
    def explained_variance_ratio(self) -> float:
        """Fraction of shell-profile variance kept, or one without PCA."""
        if self.pca is None:
            return 1.0
        return float(np.asarray(self.pca.explained_variance_ratio_).sum())

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Transform raw radial blocks to the joint endpoint coordinates."""
        values = _validated_raw(raw, self.settings)
        if self.pca is None:
            return ((values - self.radial_mean) / self.radial_scale).astype(np.float32)
        coordination = ((values[:, -1] - self.coordination_mean) / self.coordination_scale)[:, None]
        radial = self.pca.transform(values[:, :-1])
        return np.concatenate([coordination, radial], axis=1).astype(np.float32)

    def transform_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Apply the fitted transform without leaving torch.

        This path is used for generated-endpoint consistency diagnostics; data
        preprocessing uses ``transform`` and stores the result.
        """
        if raw.dim() != 2 or raw.shape[1] != self.settings.raw_dimension:
            raise ValueError(
                f"Expected raw descriptors of shape (atoms, {self.settings.raw_dimension}), "
                f"got {tuple(raw.shape)}."
            )
        dtype, device = raw.dtype, raw.device
        if self.pca is None:
            mean = torch.as_tensor(self.radial_mean, dtype=dtype, device=device)
            scale = torch.as_tensor(self.radial_scale, dtype=dtype, device=device)
            return (raw - mean) / scale

        coordination = (raw[:, -1:] - self.coordination_mean) / self.coordination_scale
        mean = torch.as_tensor(self.radial_mean, dtype=dtype, device=device)
        components = torch.as_tensor(self.pca.components_, dtype=dtype, device=device)
        scale = torch.as_tensor(self.radial_scale, dtype=dtype, device=device)
        radial = ((raw[:, :-1] - mean) @ components.T) / scale
        return torch.cat([coordination, radial], dim=-1)

    def save(self, path: Path) -> None:
        """Serialise the fitted transform."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "DescriptorTransform":
        """Load and type-check a fitted transform."""
        value = pickle.loads(Path(path).read_bytes())
        if not isinstance(value, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}.")
        return value


def clean_radial_descriptor(
    fractional_coordinates: torch.Tensor,
    cells: torch.Tensor,
    atom_counts: torch.Tensor,
    settings: JointGeometryDescriptorSettings | None = None,
) -> torch.Tensor:
    """Compute the raw 17-channel descriptor for a clean crystal batch."""
    settings = settings or JointGeometryDescriptorSettings()
    spec = settings.feature_spec()
    neighbors = periodic_neighbors(
        fractional_coordinates,
        cells,
        atom_counts,
        spec.cutoff,
    )
    return radial_block(neighbors.within(spec.cutoff), fractional_coordinates.shape[0], spec)


def transformed_clean_descriptor(
    fractional_coordinates: torch.Tensor,
    cells: torch.Tensor,
    atom_counts: torch.Tensor,
    transform: DescriptorTransform,
) -> torch.Tensor:
    """Compute and transform a clean descriptor in torch."""
    raw = clean_radial_descriptor(
        fractional_coordinates,
        cells,
        atom_counts,
        transform.settings,
    )
    return transform.transform_tensor(raw)


def _validated_raw(
    raw: np.ndarray,
    settings: JointGeometryDescriptorSettings,
) -> np.ndarray:
    """Return raw descriptors as finite float64 after checking their schema."""
    values = np.asarray(raw, dtype=np.float64)
    expected = settings.raw_dimension
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"Expected raw descriptors of shape (atoms, {expected}), got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Raw radial descriptors contain non-finite values.")
    return values
