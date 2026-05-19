# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Abstract base class for entity components in the ECS architecture."""

from abc import ABC

from ai4animation.AI4Animation import AI4Animation


class Component(ABC):
    """Base component attached to an Entity in the scene graph."""

    def __init__(self, entity, params) -> None:
        """Initialize component, run Start, and optionally enter standalone mode."""
        self.Entity = entity
        self.Start(params)
        if AI4Animation.Standalone is not None:
            self.Standalone()

    def Start(self, params) -> None:
        """Called once when the component is first attached to an entity."""

    def Standalone(self) -> None:
        """Called when the application is running in standalone (windowed) mode."""

    def Update(self) -> None:
        """Called every frame to update component state."""

    def Draw(self) -> None:
        """Called every frame to render component visuals."""

    def GUI(self) -> None:
        """Called every frame to render component GUI elements."""
