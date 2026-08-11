"""
Coarse-to-fine flow matching for crystal structure prediction.

This package adds one probability path to OMatG. Instead of moving every atom along its own periodic geodesic at the
same rate, it moves the centroids of a partition of the structure ahead of the atoms' positions within their groups, so
that generation resolves coordination-scale arrangement before internal geometry.

Four arms share everything except that path: an atomwise baseline, two fixed partitions (periodic k-medoids and
CrystalNN coordination shells) and one partition learned end to end through the flow-matching objective.
"""

from .blur import apply_group_mean, coarse_fine_delta, fine_energy_fraction, one_hot_assignment
from .data import CGDataModule, CGOMGDataset
from .diagnostics import CoarseToFineDiagnostics, GroupingTemperatureSchedule, adjusted_rand_index, group_statistics
from .groupfile import GroupTable
from .grouper import AnchorMembershipGrouper, build_grouper
from .grouping import Grouping, PrecomputedGrouping
from .interpolant import CoarseToFineStochasticInterpolant
from .lightning import CGLightning

__all__ = [
    "AnchorMembershipGrouper",
    "CGDataModule",
    "CGLightning",
    "CGOMGDataset",
    "CoarseToFineDiagnostics",
    "CoarseToFineStochasticInterpolant",
    "Grouping",
    "GroupTable",
    "GroupingTemperatureSchedule",
    "PrecomputedGrouping",
    "adjusted_rand_index",
    "apply_group_mean",
    "build_grouper",
    "coarse_fine_delta",
    "fine_energy_fraction",
    "group_statistics",
    "one_hot_assignment",
]
