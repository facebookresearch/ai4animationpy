# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path
from ai4animation import (
    AI4Animation,
    Dataset,
    Motion,
    MotionEditor,
    Actor,
    Vector3,
    Time,
    Transform,
    RootModule,
    FBX,
    Motion
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Geno")

sys.path.append(ASSETS_PATH)
import Definitions

sys.path.append(Path(__file__).parent)

class Program:
    def __init__(self, filename):
        self.Filename = filename

    def Start(self):
        fbx = FBX(self.Filename)
        self.Motion = fbx.LoadMotion()
        self.Mirror = False
        self.Pose = None
        entity = AI4Animation.Scene.AddEntity("Actor")
        model_path = os.path.join(ASSETS_PATH, "Model.fbx")
        self.Actor = entity.AddComponent(
            Actor, model_path, Definitions.FULL_BODY_NAMES
        )
        self.Actor.Entity.SetPosition(Vector3.Create(0, 0, 0))
        self.RootModule = RootModule(self.Motion, Definitions.HipName, Definitions.LeftHipName, Definitions.RightHipName, Definitions.LeftShoulderName, Definitions.RightShoulderName)

    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self):
        timestamp = Time.TotalTime % self.Motion.TotalTime # [0]
        self.Pose = self.Motion.GetBoneTransformations(
            timestamps=timestamp, mirrored=self.Mirror
        )
        self.Actor.SetRoot(self.RootModule.GetTransforms(timestamps=timestamp, mirrored=self.Mirror))
        self.Actor.SetTransforms(
            self.Motion.GetBoneTransformations(
                timestamps=timestamp,
                bone_names_or_indices=self.Actor.GetBoneNames(),
                mirrored=self.Mirror,
            )
        )
        self.Actor.SyncToScene()

    # def GUI(self):
    #     positions = Transform.GetPosition(self.Pose).reshape(-1, 3)
    #     AI4Animation.Draw.Text3D(
    #         self.Motion.Hierarchy.BoneNames,
    #         positions,
    #         size=0.0125,
    #         color=AI4Animation.Color.BLACK,
    #     )

    # def Draw(self):
    #     AI4Animation.Draw.Transform(self.Pose, size=0.25, axisSize=0)

AI4Animation(Program("lafan.fbx"))
