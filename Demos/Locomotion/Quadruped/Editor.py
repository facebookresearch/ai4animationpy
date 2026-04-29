# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

from ai4animation import (
    AI4Animation,
    ContactModule,
    Dataset,
    GuidanceModule,
    MirrorModule,
    MotionEditor,
    MotionModule,
    RootModule,
    Vector3,
    AssetManager
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Quadruped")

sys.path.append(ASSETS_PATH)
import Definitions


class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")
        self.dataset = Dataset(
            os.path.join(ASSETS_PATH, "Motions"),
            [
                lambda x: RootModule(
                    x,
                    Definitions.HipName,
                    Definitions.LeftHipName,
                    Definitions.RightHipName,
                    Definitions.LeftShoulderName,
                    Definitions.RightShoulderName,
                    Definitions.NeckName,
                    topology=RootModule.Topology.QUADRUPED,
                ),
                lambda x: MotionModule(x),
                lambda x: ContactModule(
                    x,
                    [
                        (Definitions.LeftHandSiteName, 0.05, 1.0),
                        (Definitions.RightHandSiteName, 0.05, 1.0),
                        (Definitions.LeftFootSiteName, 0.05, 1.0),
                        (Definitions.RightFootSiteName, 0.05, 1.0),
                    ],
                ),
                lambda x: GuidanceModule(x),
                lambda x: MirrorModule(
                    x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 0)
                ),
            ],
        )

        editor.AddComponent(
            MotionEditor,
            self.dataset,
            os.path.join(ASSETS_PATH, "Dog.glb"),
            Definitions.FULL_BODY_NAMES,
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        pass


def main():
    AI4Animation(Program())


if __name__ == "__main__":
    main()
