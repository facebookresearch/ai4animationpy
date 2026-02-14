# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import numpy as np
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.RootModule import RootModule
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Tensor, Vector3


class GuidanceModule(Module):
    def __init__(self, motion: Motion) -> None:
        super().__init__(motion)
        self.RootModule = None

    def GetName(self):
        return "Guidance"

    def GetRootModule(self):
        if self.RootModule is None:
            self.RootModule = self.Motion.GetModule(RootModule)
        return self.RootModule

    def CreateGuidance(self, id, timestamp, mirrored, names, smoothing):
        return self.Guidance(
            id, names, self.GetGuidancePositions(timestamp, mirrored, names, smoothing)
        )

    def GetGuidancePositions(self, timestamps, mirrored, names, smoothing):
        timestamps = Tensor.Unsqueeze(timestamps, -1) + smoothing.Timestamps
        roots = self.GetRootModule().GetTransforms(timestamps, mirrored)
        positions = self.Motion.GetBonePositions(timestamps, names, mirrored)
        positions = Vector3.PositionTo(positions, Tensor.Unsqueeze(roots, -3))
        positions = Tensor.Squeeze(Tensor.Mean(positions, axis=-3), -3)
        return positions

    def Standalone(self):
        self.Slider_Smooth = AI4Animation.GUI.Slider(
            0.45, 0.3, 0.1, 0.05, 1.0, 0.0, 2.0
        )
        self.Button_Save = AI4Animation.GUI.Button(
            "Save", 0.45, 0.45, 0.1, 0.05, False, False
        )

    def GUI(self, editor):
        if Module.Visualize[GuidanceModule]:
            self.Slider_Smooth.GUI()
            self.Button_Save.GUI()

    def Draw(self, editor):
        if Module.Visualize[GuidanceModule]:
            guidance = self.CreateGuidance(
                self.Motion.Name
                + "_"
                + str(self.Motion.GetFrameIndices(editor.Timestamp)[0] + 1),
                editor.Timestamp,
                editor.Mirror,
                editor.Actor.GetBoneNames(),
                TimeSeries(
                    0.0, self.Slider_Smooth.GetValue(), editor.TimeSeries.SampleCount
                ),
            )
            guidance.Draw(editor.Actor)
            if self.Button_Save.IsPressed():
                guidance.Save()

    class Guidance:
        def __init__(self, id, names, positions):
            self.ID = id
            self.Names = names
            self.Positions = positions

        def Draw(self, actor):
            AI4Animation.Draw.Skeleton(
                None,
                Vector3.PositionFrom(self.Positions, actor.Root),
                actor,
                size=2.0,
                color=AI4Animation.Standalone.Color.MAGENTA,
            )

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
