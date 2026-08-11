"""
Lightning module for the coarse-to-fine arms.

The only thing this adds to OMGLightning is registering the grouping network as a child module. StochasticInterpolants
is a plain object rather than an nn.Module, so a grouping nested inside it would be invisible to Lightning: its
parameters would never be moved to the accelerator, never handed to the optimizer, and never saved in a checkpoint.
Assigning the same object to an attribute of the Lightning module fixes all four, and because it is the same object the
interpolant keeps using it directly.

The grouping network is trained by the flow-matching loss alone, with the same optimizer and learning rate as the
denoiser. That is the experiment as specified: no auxiliary objective and no separate schedule, so that any difference
between the arms is attributable to the probability path rather than to extra supervision.
"""

from typing import Any
from torch import nn
from omg.omg_lightning import OMGLightning


class CGLightning(OMGLightning):
    """
    OMGLightning that registers a learned grouping so that it is trained, moved and checkpointed with the model.

    :param args:
        Positional arguments forwarded to OMGLightning.
    :type args: Any
    :param kwargs:
        Keyword arguments forwarded to OMGLightning.
    :type kwargs: Any
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct the Lightning module."""
        super().__init__(*args, **kwargs)
        interpolant = self.si.get_stochastic_interpolant("pos")
        grouping = interpolant.get_grouping() if hasattr(interpolant, "get_grouping") else None
        if isinstance(grouping, nn.Module):
            self.cg_grouper = grouping

    def has_learned_grouping(self) -> bool:
        """
        Whether this module holds a grouping with trainable parameters.

        :return:
            Whether a learned grouping is registered.
        :rtype: bool
        """
        return isinstance(getattr(self, "cg_grouper", None), nn.Module)
