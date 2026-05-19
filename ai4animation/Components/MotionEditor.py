# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Editor component for interactive motion playback, timeline control, and module visualization."""

from ai4animation import Time
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Components.Actor import Actor
from ai4animation.Components.Component import Component


class MotionEditor(Component):
    def Start(self, params):
        self.Dataset = params[0]
        self.ModelPath = params[1]
        self.BoneNames = params[2]

        self.Actor = self.Entity.AddComponent(Actor, self.ModelPath, self.BoneNames)

        self.Timestamp = 0.0

        self.TimeSeries = TimeSeries(start=-0.5, end=0.5, samples=31)

        self.Timescale = 1.0
        self.Mirror = False

        self.Motion = None
        self.LoadMotion(0)

        self.Files = self.Dataset.Files

    def LoadMotion(self, index):
        if len(self.Dataset) > 0:
            if 0 <= index < len(self.Dataset):
                self.Motion = self.Dataset.LoadMotion(index)
                self.Timestamp = 0.0
        else:
            print("Dataset has no files")

    def Update(self):
        # if AI4Animation.Standalone is None:
        #     if not self.IsPlaying():
        #         return
        # else:
        #     if (
        #         not self.IsPlaying()
        #         and not self.Slider_Assets.Modified
        #         and not self.Slider_Timeline.Modified
        #         and not self.Slider_Scale.Modified
        #         and not self.Button_Mirror.IsPressed()
        #     ):
        #         return
        self.LoadFrame(self.Timestamp)

    def IsSetup(self):
        return self.Motion is not None

    def IsPlaying(self):
        return self.Button_Play_Motion.Active

    def LoadFrame(self, timestamp):
        if self.IsSetup():
            self.Timestamp = timestamp
            if self.Actor:
                self.Actor.SetTransforms(
                    self.Motion.GetBoneTransformations(
                        timestamp, self.Actor.GetBoneNames(), mirrored=self.Mirror
                    )
                )
                self.Actor.SetVelocities(
                    self.Motion.GetBoneVelocities(
                        timestamp, self.Actor.GetBoneNames(), mirrored=self.Mirror
                    )
                )
                self.Actor.Entity.SetScale(self.Motion.Scale)

            for module in self.Motion.Modules:
                module.Callback(self)

            if self.Actor:
                self.Actor.SyncToScene()

    def LoadPreviousMotion(self):
        index = self.Dataset.GetMotionIndex(self.Motion)
        self.LoadMotion(0 if index is None else (index - 1))

    def LoadNextMotion(self):
        index = self.Dataset.GetMotionIndex(self.Motion)
        self.LoadMotion(0 if index is None else (index + 1))

    def Standalone(self):
        self.Canvas = AI4Animation.GUI.Canvas(
            "Motion Editor", 0.25, 0.01, 0.5, 0.15, scale_width=True, scale_height=False
        )
        self.Slider_Assets = AI4Animation.GUI.Slider(
            0.15, 0.035, 0.7, 0.025, 0.0, 0.0, 1.0, canvas=self.Canvas
        )
        self.Button_Prev_Motion = AI4Animation.GUI.Button(
            "<<", 0.01, 0.035, 0.06, 0.025, False, False, self.Canvas
        )
        self.Button_Next_Motion = AI4Animation.GUI.Button(
            ">>", 0.08, 0.035, 0.06, 0.025, False, False, self.Canvas
        )
        self.Button_Play_Motion = AI4Animation.GUI.Button(
            "|>",
            0.01,
            0.065,
            0.13,
            0.04,
            False,
            True,
            self.Canvas,
            color_default=AI4Animation.Color.GREEN,
            color_hovered=AI4Animation.Color.GREEN,
            color_active=AI4Animation.Color.RED,
        )
        self.Slider_Timeline = AI4Animation.GUI.Slider(
            0.15, 0.065, 0.7, 0.04, 0.0, 0.0, 1.0, canvas=self.Canvas
        )
        self.Slider_Scale = AI4Animation.GUI.Slider(
            0.85, 0.155, 0.14, 0.03, 1.0, 0.5, 1.5, canvas=self.Canvas, label="Scale"
        )
        self.Dropdown_Modules = AI4Animation.GUI.Dropdown(
            "Modules",
            0.01,
            0.1125,
            0.13,
            0.03,
            options=[
                (
                    self.Motion.Modules[i].GetName(),
                    lambda x: self.Motion.Modules[x].ToggleVisualize(),
                )
                for i in range(len(self.Motion.Modules))
            ],
            canvas=self.Canvas,
        )
        self.Button_Mirror = AI4Animation.GUI.Button(
            "Mirror", 0.855, 0.1125, 0.13, 0.03, False, False, canvas=self.Canvas
        )
        self.Search_Field = AI4Animation.GUI.TextField(
            0.15, 0.11, 0.7, 0.035, canvas=self.Canvas, default="Search..."
        )

    def GUI(self):
        self.Canvas.GUI()

        # Asset Selection
        self.Slider_Assets.GUI()
        count = max(len(self.Dataset) - 1, 1)
        if self.Slider_Assets.Modified:
            index = int(round(self.Slider_Assets.GetValue() * count))
            value = index / count
            self.Slider_Assets.SetValue(value)
            self.LoadMotion(index)
        else:
            index = self.Dataset.GetMotionIndex(self.Motion)
            self.Slider_Assets.SetValue(0 if index is None else (index / count))

        self.Button_Prev_Motion.GUI()
        self.Button_Next_Motion.GUI()

        self.Search_Field.GUI()
        if self.Search_Field.Changed:
            self.Dataset.Filter(self.Search_Field.Text)

        if self.Button_Prev_Motion.IsPressed():
            self.LoadPreviousMotion()
        if self.Button_Next_Motion.IsPressed():
            self.LoadNextMotion()

        AI4Animation.Draw.Text(
            self.Motion.Name,
            0.5,
            0.05,
            0.015,
            AI4Animation.Color.BLACK,
            pivot=0.5,
            canvas=self.Canvas,
        )

        index = self.Dataset.GetMotionIndex(self.Motion)
        AI4Animation.Draw.Text(
            f"Asset {0 if index is None else (index + 1)}/{len(self.Dataset)}",
            0.86,
            0.05,
            0.0175,
            AI4Animation.Color.BLACK,
            canvas=self.Canvas,
        )

        # Motion Timeline
        self.Button_Play_Motion.GUI()
        if self.Button_Play_Motion.Active:
            self.Button_Play_Motion.Label = "||"
            self.Timestamp = (self.Timestamp + Time.DeltaTime) % self.Motion.TotalTime
        else:
            self.Button_Play_Motion.Label = "|>"

        self.Slider_Timeline.GUI()
        if self.Slider_Timeline.Modified:
            self.Timestamp = self.Slider_Timeline.GetValue() * self.Motion.TotalTime
        else:
            self.Slider_Timeline.SetValue(self.Timestamp / self.Motion.TotalTime)

        self.Motion.Scale = self.Slider_Scale.GUI(self.Motion.Scale)
        # if self.Slider_Scale.Modified:
        #     self.Motion.Scale = self.Slider_Scale.GetValue()
        # else:
        #     self.Slider_Scale.SetValue(self.Motion.Scale)
        # self.Slider_Scale.Label = str(round(self.Motion.Scale, 3))

        AI4Animation.Draw.Text(
            f"{self.Motion.GetFrameIndices(self.Timestamp)[0] + 1}/{self.Motion.NumFrames}",
            0.5,
            0.0825,
            0.025,
            AI4Animation.Color.BLACK,
            pivot=0.5,
            canvas=self.Canvas,
        )

        AI4Animation.Draw.Text(
            str(round(self.Timestamp, 2)) + "s",
            0.86,
            0.075,
            0.02,
            AI4Animation.Color.BLACK,
            pivot=0,
            canvas=self.Canvas,
        )

        AI4Animation.Draw.Text(
            str(round(self.Motion.Framerate, 2)) + "Hz",
            0.86,
            0.095,
            0.02,
            AI4Animation.Color.BLACK,
            pivot=0,
            canvas=self.Canvas,
        )

        self.Dropdown_Modules.GUI(Module.GetVisualizeStates(self.Motion.Modules))

        self.Button_Mirror.GUI()

        if self.Button_Mirror.IsPressed():
            self.Mirror = not self.Mirror
        self.Button_Mirror.Active = self.Mirror

        for module in self.Motion.Modules:
            module.GUI(self)

        # if not self.IsPlaying():
        #     self.Actor.DrawHandle()

    def Draw(self):
        for module in self.Motion.Modules:
            module.Draw(self)
