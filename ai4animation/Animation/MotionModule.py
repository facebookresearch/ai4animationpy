# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Module for extracting per-bone motion features (positions, rotations, velocities)."""

from ai4animation import AssetManager, Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.RootModule import RootModule
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Tensor, Transform, Vector3


class MotionModule(Module):
    def __init__(self, motion: Motion) -> None:
        super().__init__(motion)

    def Initialize(self):
        pass

    def GetName(self):
        return "Motion"

    def ComputeSeries(
        self,
        timestamp: float,
        mirrored: bool,
        names: list[str],
        timeseries: TimeSeries,
        smoothing: TimeSeries = None,
        power=1.0,
    ):
        timestamps = timeseries.SimulateTimestamps(timestamp)
        transforms = self.GetTransforms(timestamps, mirrored, names, smoothing, power)
        velocities = self.GetVelocities(timestamps, mirrored, names, smoothing, power)
        instance = self.Series(timeseries, names, transforms, velocities)
        return instance

    def GetTransforms(
        self,
        timestamps,
        mirrored,
        names,
        smoothing: TimeSeries = None,
        power=1.0,
    ):
        if smoothing is not None and smoothing.Window > 0.0:
            return Transform.Normalize(
                self.SmoothCurves(
                    self.Motion.GetBoneTransformations,
                    timestamps,
                    mirrored,
                    names,
                    smoothing,
                    power,
                )
            )
        else:
            return self.Motion.GetBoneTransformations(timestamps, names, mirrored)

    def GetPositions(
        self,
        timestamps,
        mirrored,
        names,
        smoothing: TimeSeries = None,
        power=1.0,
    ):
        if smoothing is not None and smoothing.Window > 0.0:
            return self.SmoothCurves(
                self.Motion.GetBonePositions,
                timestamps,
                mirrored,
                names,
                smoothing,
                power,
            )
        else:
            return self.Motion.GetBonePositions(timestamps, names, mirrored)

    def GetVelocities(
        self,
        timestamps,
        mirrored,
        names,
        smoothing: TimeSeries = None,
        power=1.0,
    ):
        if smoothing is not None and smoothing.Window > 0.0:
            return self.SmoothCurves(
                self.Motion.GetBoneVelocities,
                timestamps,
                mirrored,
                names,
                smoothing,
                power,
            )
        else:
            return self.Motion.GetBoneVelocities(timestamps, names, mirrored)

    def SmoothCurves(self, fn, timestamps, mirrored, names, smoothing, power):
        axis = len(timestamps.shape)
        timestamps = Tensor.Unsqueeze(timestamps, -1)
        timestamps = timestamps + smoothing.Timestamps
        values = fn(timestamps, names, mirrored)
        values = Tensor.Gaussian(
            values,
            power=power,
            axis=axis,
            keepDim=False,
        )
        return values

    def Standalone(self):
        self.Button_Smooth = AI4Animation.GUI.Button(
            "Smooth", 0.4, 0.2, 0.2, 0.05, False, True
        )
        self.Slider_Window = AI4Animation.GUI.Slider(
            0.4, 0.25, 0.2, 0.05, 1.0, 0.0, 2.0, label="Window"
        )
        self.Slider_Power = AI4Animation.GUI.Slider(
            0.4, 0.3, 0.2, 0.05, 1.0, 0.0, 1.0, label="Power"
        )

    def GUI(self, editor):
        if Module.Visualize[MotionModule]:
            self.Button_Smooth.GUI()
            self.Slider_Window.GUI()
            self.Slider_Power.GUI()

    def Draw(self, editor):
        if Module.Visualize[MotionModule]:
            window = self.Slider_Window.GetValue()
            power = self.Slider_Power.GetValue()
            smoothing = (
                TimeSeries(
                    -window / 2,
                    window / 2,
                    editor.TimeSeries.SampleCount,
                )
                if self.Button_Smooth.Active
                else None
            )
            self.ComputeSeries(
                editor.Timestamp,
                editor.Mirror,
                editor.Actor.GetBoneNames(),
                editor.TimeSeries,
                smoothing,
                power,
            ).Draw()

    class Series(TimeSeries):
        def __init__(self, timeSeries, names, transforms=None, velocities=None):
            super().__init__(timeSeries.Start, timeSeries.End, timeSeries.SampleCount)
            self.Names = names
            self.NameToIndexMap = {}
            for i in range(len(names)):
                self.NameToIndexMap[names[i]] = i

            self.Transforms = (
                Transform.Identity((self.SampleCount, self.TrajectoryCount))
                if transforms is None
                else transforms
            )
            self.Velocities = (
                Vector3.Zero((self.SampleCount, self.TrajectoryCount))
                if velocities is None
                else velocities
            )

        @property
        def TrajectoryCount(self) -> int:
            return len(self.Names)

        def GetTransforms(self, bone_names=None, start=None, end=None):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            if bone_names == None:
                return self.Transforms[start:end]
            else:
                bone_indices = [self.NameToIndexMap[name] for name in bone_names]
                return self.Transforms[start:end, bone_indices, :, :]

        def GetPositions(self, bone_names=None, start=None, end=None):
            return Transform.GetPosition(self.GetTransforms(bone_names, start, end))

        def GetRotations(self, bone_names=None, start=None, end=None):
            return Transform.GetRotation(self.GetTransforms(bone_names, start, end))

        def GetVelocities(self, bone_names=None, start=None, end=None):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            if bone_names == None:
                return self.Velocities[start:end]
            else:
                bone_indices = [self.NameToIndexMap[name] for name in bone_names]
                return self.Velocities[start:end, bone_indices, :]

        def ClampDistance(self, pivot, distance):
            for index in range(self.SampleCount):
                for bone in range(self.TrajectoryCount):
                    offset = Transform.GetPosition(self.Transforms[index, bone]) - pivot
                    horizontal = Vector3.Create(offset[0], 0, offset[2])
                    if Vector3.Length(horizontal) > distance:
                        horizontal = distance * Vector3.Normalize(horizontal)
                        self.Transforms[index, bone, :3, 3] = pivot + Vector3.Create(
                            horizontal[0], offset[1], horizontal[2]
                        )

        def Draw(
            self,
            start=None,
            end=None,
            thickness=1.0,
            drawConnections=True,
            drawPositions=True,
            drawVelocities=True,
            positionColor=None,
            velocityColor=None,
            actor=None,
        ):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            if actor is None:
                for i, _ in enumerate(self.Names):
                    positions = Transform.GetPosition(self.Transforms[start:end, i])
                    velocities = self.Velocities[start:end, i]
                    pColor = (
                        AI4Animation.Color.BLACK
                        if positionColor is None
                        else positionColor
                    )
                    vColor = Utility.Opacity(
                        (
                            AI4Animation.Color.GREEN
                            if velocityColor is None
                            else velocityColor
                        ),
                        0.5,
                    )
                    if drawConnections:
                        AI4Animation.Draw.LineStrip(positions, color=pColor)
                    if drawPositions:
                        AI4Animation.Draw.Sphere(
                            positions, 0.02 * thickness, color=pColor
                        )
                    if drawVelocities:
                        AI4Animation.Draw.Vector(
                            positions, velocities, 0.005 * thickness, color=vColor
                        )
            else:
                for i in range(start, end, 1):
                    AI4Animation.Draw.Skeleton(
                        None,
                        Transform.GetPosition(self.Transforms[i]),
                        actor,
                        bones=self.Names,
                    )
