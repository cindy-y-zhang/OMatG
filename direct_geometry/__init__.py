"""
Direct geometric features for the OMatG denoiser.

The project this package belongs to has twice tried to give the denoiser a *semantic* variable -- a rigid block pose, then
a coordination-geometry token -- and twice found that the variable was either unreadable by the encoder or worth less than
the optimisation capacity it consumed. The diagnosis that survived both attempts is narrower and more mechanical than
either hypothesis: ``CSPNet`` never computes an interatomic distance, and on a fully connected graph it cannot represent
periodic image multiplicity at all. A classifier asked to name an atom's coordination number from a clean crystal beat a
geometry-blind control by 1.57 accuracy points; adding the Cartesian distance took that to 7.29, and adding a periodic
multiedge graph on top took it to about 10.2.

So this package skips the semantic variable. It adds continuous, rotationally and translationally invariant summaries of
each atom's local environment straight into the node stream, and separately offers a corrected periodic message graph.
Nothing in ``omg`` is modified: the interpolants, the losses, the sampler and the output heads are the baseline's, so a
direct-geometry run and the atomwise baseline solve the same generative problem and their match rates are comparable.

The two additions are deliberately independent switches, because they are different claims and the earlier work cannot
tell them apart. ``feature_mode`` controls the node descriptor; ``message_graph`` controls the topology and the edge
features. A win that comes only from the graph is a graph result, and reporting it as evidence for the descriptor is the
mistake this package is arranged to make impossible.

Both additions enter through zero-initialised weights, so every arm is *functionally identical at initialisation* and any
difference in a learning curve is something training found rather than a different starting point.
"""

from .features import DescriptorSpec, FeatureMode, local_environment_descriptor
from .neighbors import Neighbors, bounded_cutoff, constant_degree_radius, periodic_neighbors

__all__ = [
    "DescriptorSpec",
    "FeatureMode",
    "Neighbors",
    "bounded_cutoff",
    "constant_degree_radius",
    "local_environment_descriptor",
    "periodic_neighbors",
]
