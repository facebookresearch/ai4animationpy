# Copyright (c) Meta Platforms, Inc. and affiliates.
import AnimRig as Definitions
from ai4animation import (
    AI4Animation,
    AssetManager,
    ContactModule,
    Dataset,
    MotionEditor,
    MotionModule,
    RootModule,
    TrackingModule,
)


class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")

        editor.AddComponent(
            MotionEditor,
            Dataset(
                AssetManager.GetPath("AnimRig/Cranberry"),
                [
                    lambda x: RootModule(
                        x,
                        Definitions.HipName,
                        Definitions.LeftHipName,
                        Definitions.RightHipName,
                        Definitions.LeftShoulderName,
                        Definitions.RightShoulderName,
                    ),
                    lambda x: MotionModule(x),
                    lambda x: TrackingModule(
                        x,
                        [
                            Definitions.HeadName,
                            Definitions.LeftWristName,
                            Definitions.RightWristName,
                        ],
                    ),
                    lambda x: ContactModule(
                        x,
                        [
                            (Definitions.LeftAnkleName, 0.15, 0.25),
                            (Definitions.LeftBallName, 0.1, 0.25),
                            (Definitions.RightAnkleName, 0.15, 0.25),
                            (Definitions.RightBallName, 0.1, 0.25),
                        ],
                    ),
                ],
            ),
            AssetManager.GetPath("Assets/AnimRig/Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        pass


AI4Animation(Program(), headless=False)
