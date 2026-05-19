# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Module for computing bone-ground contact labels based on height and velocity thresholds."""

from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Tensor


class ContactModule(Module):
    def __init__(
        self, motion: Motion, configs, proportional=False
    ) -> (
        None
    ):  # Each config is a tuple of (boneName, heightThreshold, velocityThreshold)
        super().__init__(motion)

        self.Configs = configs
        self.BoneNames = [config[0] for config in configs]
        self.BoneIndices = self.Motion.GetBoneIndices(self.BoneNames)
        self.HeightThresholds = [config[1] for config in configs]
        self.VelocityThresholds = [config[2] for config in configs]
        self.Proportional = proportional

    def Initialize(self):
        pass

    def GetName(self):
        return "Contact"

    def GUI(self, editor):
        pass

    def Draw(self, editor):
        if Module.Visualize[ContactModule]:
            timestamps = editor.TimeSeries.SimulateTimestamps(editor.Timestamp)
            positions = self.Motion.GetBonePositions(
                timestamps, self.BoneIndices, editor.Mirror
            ).reshape(-1, 3)
            contacts = self.GetContacts(timestamps, editor.Mirror).reshape(-1, 1)
            grounded = self.GetGrounded(timestamps, editor.Mirror).reshape(-1, 1)
            for i in range(contacts.shape[0]):
                if contacts[i]:
                    color = (
                        AI4Animation.Color.RED
                        if grounded[i]
                        else AI4Animation.Color.GREEN
                    )
                    AI4Animation.Draw.Sphere(
                        positions[i], size=0.04, color=Utility.Opacity(color, 0.5)
                    )
                else:
                    AI4Animation.Draw.Sphere(
                        positions[i],
                        size=0.02,
                        color=Utility.Opacity(AI4Animation.Color.BLACK, 0.25),
                    )

    def GetContacts(self, timestamps, mirrored):
        velocities = self.Motion.GetBoneVelocities(
            timestamps, self.BoneIndices, mirrored
        )
        velocities = Tensor.Norm(velocities, keepDim=False)
        if self.Proportional:
            lengths = self.Motion.GetBoneLengths(
                timestamps=timestamps, mirrored=mirrored
            )
            scales = Tensor.Sum(lengths, axis=-2, keepDim=False)
        else:
            scales = 1
        vavlues = velocities < (self.VelocityThresholds * scales)
        return vavlues

    def GetGrounded(self, timestamps, mirrored):
        positions = self.Motion.GetBonePositions(timestamps, self.BoneIndices, mirrored)
        heights = positions[..., 1]
        if self.Proportional:
            lengths = self.Motion.GetBoneLengths(
                timestamps=timestamps, mirrored=mirrored
            )
            scales = Tensor.Sum(lengths, axis=-2, keepDim=False)
        else:
            scales = 1
        values = heights < (self.HeightThresholds * scales)
        return values

    class Series(TimeSeries):
        def __init__(self, timeSeries, names, values=None):
            super().__init__(timeSeries.Start, timeSeries.End, timeSeries.SampleCount)
            self.Names = names
            self.Values = (
                Tensor.Zeros((self.SampleCount, len(self.Names)))
                if values is None
                else values
            )

        def GUI(self, x, y, width, height):
            AI4Animation.GUI.BarPlot(
                x,
                y,
                width,
                height,
                Tensor.SwapAxes(self.Values, 0, 1),
                label="Contacts",
                colors=AI4Animation.Color.GetRainbowColors(len(self.Names)),
            )

        def Draw(self):
            pass
