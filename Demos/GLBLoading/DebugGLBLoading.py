# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation import (
    AI4Animation,
    AssetManager,
    ContactModule,
    Dataset,
    Motion,
    MotionEditor,
    MotionModule,
    RootModule,
    Transform,
)
from Trinity import v3 as Definitions


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
            AssetManager.GetPath("Trinity/v3.glb"),
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


AI4Animation(Program("MD1_X0089.Tiktok_electro_19_A3_A"))
