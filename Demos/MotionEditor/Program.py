import sys

from ai4animation import (
    AI4Animation,
    ContactModule,
    Dataset,
    MotionEditor,
    MotionModule,
    RootModule,
)

ASSETS_PATH = "../_ASSETS_/"
sys.path.append(ASSETS_PATH)
import Definitions


class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")

        editor.AddComponent(
            MotionEditor,
            Dataset(
                ASSETS_PATH + "Motions",
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
            ASSETS_PATH + "Model.glb",
            Definitions.FULL_BODY_NAMES,
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        pass


def main():
    AI4Animation(Program())


if __name__ == "__main__":
    main()
