"""Global motif census: a per-structure histogram over coordination-environment prototypes.

The joint-geometry arms deliver per-site environments as a diffused state that starts
from Gaussian noise. The paired shuffled-content control showed that the network reads
that state's content, but also that carrying the channel at all costs more match rate
than the content is worth, because the channel is noise for most of the trajectory.

This module builds the alternative payload: pool the same per-site environments into one
permutation-invariant vector per structure, so it can be supplied as static conditioning
that never passes through a prior.

Prototypes are fitted by k-means on training sites only. Each site is softly assigned to
the prototypes and the assignments are averaged over the supercell, giving a distribution
over motifs -- "what fraction of the sites in this crystal look like each environment".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans

from direct_geometry.features import DescriptorSpec
from joint_geometry.descriptor import (
    JointGeometryDescriptorSettings,
    clean_radial_descriptor,
)


MINIMUM_SCALE = 1.0e-6
"""Floor on per-channel standard deviations, guarding constant census channels."""

DEFAULT_PROTOTYPES = 16
"""Motif vocabulary size used unless a sweep overrides it."""


@dataclass(frozen=True)
class MotifCensusSettings:
    """Everything that determines the census, hashed into the artifact manifest."""

    version: int = 1
    cutoff: float = 6.0
    shells: int = 16
    prototypes: int = DEFAULT_PROTOTYPES
    temperature_scale: float = 1.0
    fit_sites: int = 100_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.prototypes < 2:
            raise ValueError(f"A census needs at least two prototypes, got {self.prototypes}.")
        if self.temperature_scale <= 0.0:
            raise ValueError(f"The temperature scale must be positive, got {self.temperature_scale}.")

    @property
    def raw_dimension(self) -> int:
        """Width of the per-site radial descriptor the prototypes are fitted on."""
        return self.shells + 1

    @property
    def dimension(self) -> int:
        """Width of the per-structure census vector."""
        return self.prototypes

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "cutoff": self.cutoff,
            "shells": self.shells,
            "prototypes": self.prototypes,
            "temperature_scale": self.temperature_scale,
            "fit_sites": self.fit_sites,
            "seed": self.seed,
        }

    def descriptor_settings(self) -> JointGeometryDescriptorSettings:
        """Per-site settings shared with the joint-geometry descriptor for comparability."""
        return JointGeometryDescriptorSettings(
            cutoff=self.cutoff, shells=self.shells, representation="radial17"
        )

    def feature_spec(self) -> DescriptorSpec:
        return self.descriptor_settings().feature_spec()


@dataclass
class MotifCensusTransform:
    """Train-only prototypes plus the standardisations applied either side of them."""

    settings: MotifCensusSettings
    site_mean: np.ndarray
    site_scale: np.ndarray
    centroids: np.ndarray
    temperature: float
    census_mean: np.ndarray
    census_scale: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.centroids.shape[0])

    @classmethod
    def fit(cls, raw: np.ndarray, settings: MotifCensusSettings) -> "MotifCensusTransform":
        """Fit prototypes on a sample of training sites.

        ``raw`` holds untransformed per-site radial descriptors. Census standardisation is
        deferred to :meth:`finalise` because it needs pooled structures, not sites.
        """
        values = _validated_sites(raw, settings)
        site_mean = values.mean(axis=0)
        site_scale = np.maximum(values.std(axis=0), MINIMUM_SCALE)
        standardised = (values - site_mean) / site_scale

        kmeans = KMeans(
            n_clusters=settings.prototypes,
            random_state=settings.seed,
            n_init=10,
        ).fit(standardised)
        centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float64)

        # Set the softmax temperature from the spread of the data around its own
        # prototypes, so assignment sharpness does not silently change with K.
        nearest = _squared_distances(standardised, centroids).min(axis=1)
        temperature = max(float(nearest.mean()) * settings.temperature_scale, MINIMUM_SCALE)

        identity = np.zeros(settings.prototypes, dtype=np.float64)
        return cls(
            settings=settings,
            site_mean=site_mean,
            site_scale=site_scale,
            centroids=centroids,
            temperature=temperature,
            census_mean=identity,
            census_scale=np.ones(settings.prototypes, dtype=np.float64),
        )

    def finalise(self, fractions: np.ndarray) -> None:
        """Record the training census statistics used to standardise model inputs."""
        fractions = np.asarray(fractions, dtype=np.float64)
        if fractions.ndim != 2 or fractions.shape[1] != self.dimension:
            raise ValueError(
                f"Census fractions must have shape (n_structures, {self.dimension}), "
                f"got {fractions.shape}."
            )
        self.census_mean = fractions.mean(axis=0)
        self.census_scale = np.maximum(fractions.std(axis=0), MINIMUM_SCALE)

    def assign(self, raw: np.ndarray) -> np.ndarray:
        """Soft-assign per-site descriptors to prototypes, returning rows on the simplex."""
        values = _validated_sites(raw, self.settings)
        standardised = (values - self.site_mean) / self.site_scale
        distances = _squared_distances(standardised, self.centroids)
        logits = -distances / (2.0 * self.temperature)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)

    def fractions(self, raw: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Pool site assignments into one motif distribution per structure."""
        weights = self.assign(raw)
        offsets = np.asarray(offsets, dtype=np.int64)
        counts = np.diff(offsets)
        if counts.min() < 1:
            raise ValueError("Every structure must contain at least one site.")
        totals = np.add.reduceat(weights, offsets[:-1], axis=0)
        return totals / counts[:, None]

    def standardise(self, fractions: np.ndarray) -> np.ndarray:
        """Rescale census fractions to the well-conditioned inputs the adapter consumes."""
        fractions = np.asarray(fractions, dtype=np.float64)
        return ((fractions - self.census_mean) / self.census_scale).astype(np.float32)

    def transform(self, raw: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return self.standardise(self.fractions(raw, offsets))

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "MotifCensusTransform":
        with Path(path).open("rb") as handle:
            transform = pickle.load(handle)
        if not isinstance(transform, cls):
            raise TypeError(f"{path} does not hold a {cls.__name__}.")
        return transform


def raw_site_descriptors(
    fractional_coordinates: torch.Tensor,
    cells: torch.Tensor,
    atom_counts: torch.Tensor,
    settings: Optional[MotifCensusSettings] = None,
) -> torch.Tensor:
    """Per-site radial descriptors, identical to the joint-geometry per-site features."""
    settings = settings or MotifCensusSettings()
    return clean_radial_descriptor(
        fractional_coordinates, cells, atom_counts, settings.descriptor_settings()
    )


def _validated_sites(raw: np.ndarray, settings: MotifCensusSettings) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != settings.raw_dimension:
        raise ValueError(
            f"Per-site descriptors must have shape (n_sites, {settings.raw_dimension}), "
            f"got {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Per-site descriptors contain non-finite values.")
    return values


def _squared_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Expanded rather than broadcast: half a million sites times K prototypes would
    # otherwise materialise a multi-gigabyte difference tensor.
    cross = points @ centroids.T
    squared = (
        np.einsum("nd,nd->n", points, points)[:, None]
        - 2.0 * cross
        + np.einsum("kd,kd->k", centroids, centroids)[None, :]
    )
    return np.maximum(squared, 0.0)
