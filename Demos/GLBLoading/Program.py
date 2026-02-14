# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

from ai4animation import Actor, AI4Animation, Motion, Time

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Trinity3")

sys.path.append(ASSETS_PATH)
import Definitions


class Program:
    def __init__(self, path):
        self.Path = path

    def Start(self):
        glb_motion = Motion.LoadFromGLB(
            self.Path, names=Definitions.FULL_BODY_NAMES, floor=None
        )
        glb_motion.SaveToNPZ(glb_motion.Name)
        npz_motion = Motion.LoadFromNPZ(glb_motion.Name)
        self.Motion = npz_motion
        self.Mirror = False

        self.Pose = None

        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )

    def Update(self):
        timestamp = Time.TotalTime % self.Motion.TotalTime
        self.Pose = self.Motion.GetBoneTransformations(
            timestamps=timestamp, mirrored=self.Mirror
        )
        self.Actor.SetTransforms(
            self.Motion.GetBoneTransformations(
                timestamps=timestamp,
                bone_names_or_indices=self.Actor.GetBoneNames(),
                mirrored=self.Mirror,
            )
        )
        self.Actor.SyncToScene()

    # def GUI(self, standalone):
    #     standalone.Draw.Text3D(self.Motion.Hierarchy.BoneNames, Tensor.GetPosition(self.Pose), size=0.0125, color=standalone.Color.BLACK)

    # def Draw(self, standalone):
    #     standalone.Draw.Matrix(self.Pose, size=0.5, axisSize=0.25)


def main():
    AI4Animation(Program("Motion.glb"))


if __name__ == "__main__":
    main()
