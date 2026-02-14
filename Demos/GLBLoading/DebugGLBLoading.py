# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

from ai4animation import (
    AI4Animation,
    ContactModule,
    Dataset,
    Motion,
    MotionEditor,
    MotionModule,
    RootModule,
    Transform,
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Trinity3")

sys.path.append(ASSETS_PATH)
import Definitions


class Program:
    def __init__(self, filename):
        self.Filename = filename

    def Start(self):
        glb_motion = Motion.LoadFromGLB(
            self.Filename + ".glb", names=None, floor="body_world"
        )
        glb_motion.SaveToNPZ(self.Filename)
        npz_motion = Motion.LoadFromNPZ(self.Filename)
        self.Motion = npz_motion
        self.Mirror = False
        self.Pose = None

        editor = AI4Animation.Scene.AddEntity("MotionEditor")
        editor.AddComponent(
            MotionEditor,
            Dataset(
                "./",
                [
                    lambda x: RootModule(
                        x,
                        Definitions.HIPS_NAME,
                        Definitions.LEFT_HIP_NAME,
                        Definitions.RIGHT_HIP_NAME,
                        Definitions.LEFT_SHOULDER_NAME,
                        Definitions.RIGHT_SHOULDER_NAME,
                    ),
                    lambda x: MotionModule(x),
                    lambda x: ContactModule(
                        x,
                        [
                            (Definitions.LEFT_ANKLE_NAME, 0.15, 0.25),
                            (Definitions.LEFT_BALL_NAME, 0.1, 0.25),
                            (Definitions.RIGHT_ANKLE_NAME, 0.15, 0.25),
                            (Definitions.RIGHT_BALL_NAME, 0.1, 0.25),
                        ],
                    ),
                ],
            ),
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES_WITH_HANDS,
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        timestamp = [0]  # Time.TotalTime % self.Motion.TotalTime
        self.Pose = self.Motion.GetBoneTransformations(
            timestamps=timestamp, mirrored=self.Mirror
        )

    def GUI(self):
        positions = Transform.GetPosition(self.Pose).reshape(-1, 3)
        AI4Animation.Draw.Text3D(
            self.Motion.Hierarchy.BoneNames,
            positions,
            size=0.0125,
            color=AI4Animation.Color.BLACK,
        )

    def Draw(self):
        AI4Animation.Draw.Transform(self.Pose, size=0.25, axisSize=0)


AI4Animation(Program("Motion"))
