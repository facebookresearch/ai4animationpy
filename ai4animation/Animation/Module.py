# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Abstract base class for animation modules."""

import logging
from abc import ABC

from ai4animation.AI4Animation import AI4Animation

logger: logging.Logger = logging.getLogger(__name__)


class Module(ABC):
    """Base class for pluggable animation data extraction modules."""

    Visualize = {}

    def __init__(self, motion) -> None:
        """Initialize module with a motion asset and register visualization state."""
        self.Motion = motion
        if not type(self) in Module.Visualize:
            Module.Visualize[type(self)] = False
        try:
            if AI4Animation.Standalone is not None:
                self.Standalone()
        except Exception:
            logger.debug(
                "Standalone initialization skipped for %s", type(self).__name__
            )

    def Initialize(self):  # noqa: B027
        pass

    def GetName(self) -> str:
        """Return the name of this module subclass."""
        return type(self).__name__

    def Standalone(self) -> None:
        """Called when the application is running in standalone (windowed) mode."""

    def Callback(self, editor) -> None:
        """Called by the editor to process module-specific callbacks."""

    def GUI(self, editor) -> None:
        """Called every frame to render module GUI elements."""

    def Draw(self, editor) -> None:
        """Called every frame to render module visuals."""

    def ToggleVisualize(self) -> None:
        """Toggle the visualization state for this module type."""
        Module.Visualize[type(self)] = not Module.Visualize[type(self)]

    @staticmethod
    def GetVisualizeStates(types) -> list:
        """Return visualization states for a list of module instances."""
        return [Module.Visualize[type(t)] for t in types]
