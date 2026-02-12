import sys

from ai4animation import Actor, AI4Animation, FABRIK, Vector3

ASSETS_PATH = "../_ASSETS_/"
sys.path.append(ASSETS_PATH)
import Definitions


class Program:
    def Start(self):
        actor = AI4Animation.Scene.AddEntity("Actor")
        self.Actor = actor.AddComponent(
            Actor, ASSETS_PATH + "Model.glb", Definitions.FULL_BODY_NAMES
        )

        self.IK = FABRIK(
            self.Actor.GetBone(Definitions.LeftShoulderName),
            self.Actor.GetBone(Definitions.LeftWristName),
        )

        self.Target = AI4Animation.Scene.AddEntity("Target")
        self.Target.SetPosition(
            self.Actor.GetBone(Definitions.LeftWristName).GetPosition()
        )

        self.Pose = self.Actor.GetTransforms()

    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self):
        self.Actor.SetTransforms(self.Pose)
        self.IK.Solve(
            self.Target.GetPosition(),
            self.Target.GetRotation(),
            max_iterations=10,
            threshold=0.001,
        )
        self.Actor.SyncToScene(self.IK.Bones)

    def Draw(self):
        pass

    def GUI(self):
        self.Target.DrawHandle()
        pass


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "___main__":
    main()
