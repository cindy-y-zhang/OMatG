"""
Inverse design of lithium superionic conductors by coordination motif.

The target is a structure family named by how many GeS4, PS4 and SiS4 tetrahedra it holds
and by how those tetrahedra connect, following Li21Ge8P3S34, whose eight-to-three ratio had
no precedent. :mod:`inverse_design.polyhedra` measures that description in a structure.
"""

from .polyhedra import (
    IDEAL_TETRAHEDRAL_ANGLE,
    Polyhedron,
    PolyhedraCensus,
    PolyhedronSettings,
    census,
    summarise,
)

__all__ = [
    "IDEAL_TETRAHEDRAL_ANGLE",
    "Polyhedron",
    "PolyhedraCensus",
    "PolyhedronSettings",
    "census",
    "summarise",
]
