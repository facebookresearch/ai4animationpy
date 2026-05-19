# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Guidance module for loading and applying external motion guidance data."""

import os

import numpy as np
from ai4animation.AI4Animation import AI4Animation, Utility
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.MotionModule import MotionModule
from ai4animation.Animation.RootModule import RootModule
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Tensor, Transform, Vector3


class GuidanceModule(Module):
    def __init__(self, motion: Motion) -> None:
        super().__init__(motion)
        self.RootModule = None

    def Initialize(self):
        self.RootModule = self.Motion.GetModule(RootModule)
        self.MotionModule = self.Motion.GetModule(MotionModule)

    def GetName(self):
        return "Guidance"

    def CreateLegacyGuidance(self, id, timestamp, mirrored, smoothing, names):
        return self.Guidance(
            id,
            names,
            self.GetLegacyGuidance(timestamp, mirrored, smoothing, names),
        )

    def GetLegacyGuidance(self, timestamps, mirrored, smoothing, names):
        timestamps = Tensor.Unsqueeze(timestamps, -1) + smoothing.Timestamps
        roots = self.RootModule.GetTransforms(timestamps, mirrored)
        positions = self.Motion.GetBonePositions(timestamps, names, mirrored)
        positions = Vector3.PositionTo(positions, Tensor.Unsqueeze(roots, -3))
        positions = Tensor.Squeeze(Tensor.Mean(positions, axis=-3), -3)
        return positions

    def CreateAgnosticGuidance(
        self, id, timestamp, mirrored, smoothing, names, parents
    ):
        return self.Guidance(
            id,
            names,
            self.GetAgnosticGuidance(timestamp, mirrored, smoothing, names, parents),
        )

    def GetAgnosticGuidance(self, timestamps, mirrored, smoothing, names, parents):
        timestamps = Tensor.Unsqueeze(timestamps, -1) + smoothing.Timestamps
        roots = self.RootModule.GetTransforms(timestamps, mirrored)
        parentPositions = self.Motion.GetBonePositions(timestamps, parents, mirrored)
        positions = self.Motion.GetBonePositions(timestamps, names, mirrored)
        deltas = positions - parentPositions
        deltas = Vector3.DirectionTo(deltas, Tensor.Unsqueeze(roots, -3))
        deltas = Tensor.Normalize(deltas)
        deltas = Tensor.Squeeze(Tensor.Mean(deltas, axis=-3), -3)
        return deltas

    def GetCenterOfBones(
        self,
        timestamps=None,
        mirrored=False,
        bone_names_or_indices=None,
        smoothing: TimeSeries = None,
        power=1.0,
    ):
        roots = self.RootModule.GetTransforms(timestamps, mirrored, smoothing)
        roots = Tensor.Unsqueeze(roots, -3)
        positions = self.MotionModule.GetPositions(
            timestamps, mirrored, bone_names_or_indices, smoothing, power
        )
        positions = Vector3.PositionTo(positions, roots)
        positions = Tensor.Mean(positions, axis=-2, keepDim=False)
        return positions

    def GetGradientOfBones(
        self,
        timestamps=None,
        mirrored=False,
        source_name_or_index=None,
        target_names_or_indices=None,
        smoothing: TimeSeries = None,
        power=1.0,
    ):
        roots = self.RootModule.GetTransforms(timestamps, mirrored, smoothing)
        roots = Tensor.Unsqueeze(roots, -3)
        source = self.MotionModule.GetPositions(
            timestamps,
            mirrored,
            [source_name_or_index],
            smoothing,
            power,
        )
        target = self.MotionModule.GetPositions(
            timestamps,
            mirrored,
            target_names_or_indices,
            smoothing,
            power,
        )
        gradients = target - source
        gradients = Vector3.DirectionTo(gradients, roots)
        gradients = Tensor.Mean(gradients, axis=-2, keepDim=False)
        gradients = Vector3.Normalize(gradients)
        return gradients

    def Standalone(self):
        self.Button_Legacy = AI4Animation.GUI.Button(
            "Legacy", 0.4, 0.2, 0.1, 0.05, True, True
        )
        self.Button_Agnostic = AI4Animation.GUI.Button(
            "Agnostic", 0.5, 0.2, 0.1, 0.05, True, True
        )
        self.Slider_Smooth = AI4Animation.GUI.Slider(
            0.4, 0.25, 0.2, 0.05, 1.0, 0.0, 2.0, label="Smoothing"
        )
        self.Button_Save = AI4Animation.GUI.Button(
            "Save", 0.45, 0.3, 0.1, 0.05, False, False
        )

    def GUI(self, editor):
        if Module.Visualize[GuidanceModule]:
            self.Slider_Smooth.GUI()
            self.Button_Save.GUI()
            self.Button_Legacy.GUI()
            self.Button_Agnostic.GUI()

    def Draw(self, editor):
        if Module.Visualize[GuidanceModule]:
            if self.Button_Legacy.Active:
                guidance = self.CreateLegacyGuidance(
                    self.Motion.Name
                    + "_Legacy_"
                    + str(self.Motion.GetFrameIndices(editor.Timestamp)[0] + 1),
                    editor.Timestamp,
                    editor.Mirror,
                    TimeSeries(
                        0.0,
                        self.Slider_Smooth.GetValue(),
                        editor.TimeSeries.SampleCount,
                    ),
                    self.Motion.Hierarchy.BoneNames,
                )
                guidance.DrawLegacy(editor.Actor)
                if self.Button_Save.IsPressed():
                    guidance.Save()

            if self.Button_Agnostic.Active:
                guidance = self.CreateAgnosticGuidance(
                    self.Motion.Name
                    + "_Agnostic_"
                    + str(self.Motion.GetFrameIndices(editor.Timestamp)[0] + 1),
                    editor.Timestamp,
                    editor.Mirror,
                    TimeSeries(
                        0.0,
                        self.Slider_Smooth.GetValue(),
                        editor.TimeSeries.SampleCount,
                    ),
                    self.Motion.Hierarchy.BoneNames,
                    self.Motion.Hierarchy.ParentNames,
                )
                guidance.DrawAgnostic(editor.Actor.Root)
                if self.Button_Save.IsPressed():
                    guidance.Save()

    class Guidance:
        def __init__(self, id, names, positions):
            self.ID = id
            self.Names = names
            self.Positions = positions

        def DrawLegacy(self, actor) -> None:
            AI4Animation.Draw.Skeleton(
                None,
                Vector3.PositionFrom(self.Positions, actor.Root),
                actor,
                size=2.0,
                color=AI4Animation.Color.MAGENTA,
            )

        def DrawAgnostic(self, root=None):
            root = root if root is not None else Transform.Identity()
            AI4Animation.Draw.Vector(
                Vector3.PositionFrom(Vector3.Zero(len(self.Names)), root),
                Vector3.DirectionFrom(self.Positions, root),
                size=0.05,
                color=Utility.Opacity(AI4Animation.Color.MAGENTA, 0.25),
            )

        def Draw(self, root=None):
            self.DrawAgnostic(root)

        def Save(self):
            directory = "Guidances/"
            os.makedirs(directory, exist_ok=True)
            filename = f"{self.ID}.npz"
            path = os.path.join(directory, f"{filename}")
            np.savez_compressed(
                path, ID=self.ID, Names=self.Names, Positions=self.Positions
            )
            print(f"Saved {self.ID} to {path}")

        @classmethod
        def Load(self):
            pass

    class Descriptor:
        def __init__(self, centers, gradients):
            self.Centers = centers
            self.Gradients = gradients

        def Draw(self, root=None):
            root = root if root is not None else Transform.Identity()
            AI4Animation.Draw.Vector(
                Vector3.PositionFrom(self.Centers, root),
                Vector3.DirectionFrom(self.Gradients, root),
                size=0.025,
                color=Utility.Opacity(AI4Animation.Color.MAGENTA, 0.25),
            )

        @classmethod
        def Load(self):
            pass
