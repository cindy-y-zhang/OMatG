"""Tests for the motif census itself: prototypes, pooling, and standardisation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motif_conditioning.census import (
    MotifCensusSettings,
    MotifCensusTransform,
    raw_site_descriptors,
)


def sites(count: int = 600, seed: int = 0) -> np.ndarray:
    """Per-site descriptors drawn from three well-separated environments."""
    generator = np.random.default_rng(seed)
    centres = generator.normal(size=(3, 17)) * 4.0
    labels = generator.integers(0, 3, size=count)
    return centres[labels] + generator.normal(scale=0.05, size=(count, 17))


def test_settings_reject_a_degenerate_vocabulary() -> None:
    with pytest.raises(ValueError, match="at least two prototypes"):
        MotifCensusSettings(prototypes=1)
    with pytest.raises(ValueError, match="temperature scale must be positive"):
        MotifCensusSettings(temperature_scale=0.0)


def test_the_census_is_a_distribution_over_prototypes() -> None:
    settings = MotifCensusSettings(prototypes=4)
    raw = sites()
    transform = MotifCensusTransform.fit(raw, settings)
    offsets = np.array([0, 100, 350, 600])
    fractions = transform.fractions(raw, offsets)

    assert fractions.shape == (3, 4)
    assert np.allclose(fractions.sum(axis=1), 1.0)
    assert (fractions >= 0.0).all()


def test_separated_environments_produce_separated_censuses() -> None:
    """A structure built from one environment must not look like one built from another."""
    settings = MotifCensusSettings(prototypes=3)
    generator = np.random.default_rng(1)
    centres = generator.normal(size=(3, 17)) * 6.0
    pure = [centres[index] + generator.normal(scale=0.02, size=(50, 17)) for index in range(3)]
    raw = np.concatenate(pure)
    transform = MotifCensusTransform.fit(raw, settings)
    fractions = transform.fractions(raw, np.array([0, 50, 100, 150]))

    # Each structure is dominated by a different prototype.
    assert len(set(fractions.argmax(axis=1))) == 3
    assert (fractions.max(axis=1) > 0.9).all()


def test_pooling_ignores_the_order_of_sites() -> None:
    settings = MotifCensusSettings(prototypes=4)
    raw = sites()
    transform = MotifCensusTransform.fit(raw, settings)
    offsets = np.array([0, len(raw)])
    shuffled = raw[np.random.default_rng(2).permutation(len(raw))]

    assert np.allclose(
        transform.fractions(raw, offsets), transform.fractions(shuffled, offsets), atol=1e-12
    )


def test_standardisation_uses_the_recorded_training_statistics() -> None:
    settings = MotifCensusSettings(prototypes=4)
    raw = sites()
    transform = MotifCensusTransform.fit(raw, settings)
    offsets = np.arange(0, len(raw) + 1, 60)
    fractions = transform.fractions(raw, offsets)
    transform.finalise(fractions)
    standardised = transform.standardise(fractions)

    assert np.allclose(standardised.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(standardised.std(axis=0), 1.0, atol=1e-5)


def test_a_wrongly_shaped_or_non_finite_input_is_refused() -> None:
    settings = MotifCensusSettings(prototypes=4)
    transform = MotifCensusTransform.fit(sites(), settings)
    with pytest.raises(ValueError, match="must have shape"):
        transform.assign(np.zeros((10, 5)))
    broken = sites(50)
    broken[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        transform.assign(broken)


def test_finalise_refuses_a_mismatched_width() -> None:
    settings = MotifCensusSettings(prototypes=4)
    transform = MotifCensusTransform.fit(sites(), settings)
    with pytest.raises(ValueError, match="Census fractions must have shape"):
        transform.finalise(np.zeros((5, 3)))


def test_site_descriptors_match_the_joint_geometry_per_site_features() -> None:
    """The census is built from the same environments the joint arm diffuses."""
    from joint_geometry.descriptor import clean_radial_descriptor

    settings = MotifCensusSettings()
    generator = torch.Generator().manual_seed(0)
    positions = torch.rand((8, 3), generator=generator, dtype=torch.float64)
    cells = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1) * 5.0
    counts = torch.tensor([4, 4])

    ours = raw_site_descriptors(positions, cells, counts, settings)
    theirs = clean_radial_descriptor(positions, cells, counts, settings.descriptor_settings())

    assert torch.allclose(ours, theirs)
    assert ours.shape == (8, settings.raw_dimension)


def test_the_census_recovers_the_mixture_of_environments_present() -> None:
    """The defining property: each channel is the fraction of sites in that environment."""
    settings = MotifCensusSettings(prototypes=3)
    generator = np.random.default_rng(7)
    centres = generator.normal(size=(3, 17)) * 8.0

    def build(mixture: tuple[int, int, int]) -> np.ndarray:
        return np.concatenate(
            [
                centres[index] + generator.normal(scale=0.01, size=(count, 17))
                for index, count in enumerate(mixture)
            ]
        )

    mixtures = [(20, 0, 0), (10, 10, 0), (5, 5, 10)]
    raw = np.concatenate([build(mixture) for mixture in mixtures])
    offsets = np.concatenate([[0], np.cumsum([sum(mixture) for mixture in mixtures])])
    transform = MotifCensusTransform.fit(raw, settings)
    fractions = transform.fractions(raw, offsets)

    # k-means labels prototypes arbitrarily, so compare against the mixture reordered by
    # which prototype the first structure's single environment landed on.
    expected = np.array([np.asarray(m, dtype=float) / sum(m) for m in mixtures])
    order = [transform.assign(centres[index : index + 1])[0].argmax() for index in range(3)]
    assert len(set(order)) == 3
    assert np.allclose(fractions[:, order], expected, atol=1e-6)
