# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

from ai4animation import Actor, AI4Animation, Motion, RootModule, Time, Vector3

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Geno")

sys.path.append(ASSETS_PATH)
import Definitions

sys.path.append(Path(__file__).parent)


class Program:
    def __init__(self, filename):
        self.Filename = filename

    def Start(self):
        self.Motion = Motion.LoadFromFBX(self.Filename)
        self.RootModule = RootModule(
            self.Motion,
            Definitions.HipName,
            Definitions.LeftHipName,
            Definitions.RightHipName,
            Definitions.LeftShoulderName,
            Definitions.RightShoulderName,
            Definitions.NeckName,
        )

        self.Mirror = False
        self.Pose = None
        entity = AI4Animation.Scene.AddEntity("Actor")
        model_path = os.path.join(ASSETS_PATH, "Model.glb")
        self.Actor = entity.AddComponent(Actor, model_path, None)
        self.Actor.Entity.SetPosition(Vector3.Create(0, 0, 0))

    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self):
        timestamp = Time.TotalTime % self.Motion.TotalTime  # [0]
        self.Pose = self.Motion.GetBoneTransformations(
            timestamps=timestamp, mirrored=self.Mirror
        )
        self.Actor.SetRoot(
            self.RootModule.GetTransforms(timestamps=timestamp, mirrored=self.Mirror)
        )
        self.Actor.SetTransforms(
            self.Motion.GetBoneTransformations(
                timestamps=timestamp,
                bone_names_or_indices=self.Actor.GetBoneNames(),
                mirrored=self.Mirror,
            )
        )
        self.Actor.SyncToScene()


AI4Animation(Program("geno.fbx"))
