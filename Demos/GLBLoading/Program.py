from ai4animation import Actor, AI4Animation, AssetManager, Motion, Time
from Trinity import v3 as Trinity


class Program:
    def __init__(self, path):
        self.Path = path

    def Start(self):
        glb_motion = Motion.LoadFromGLB(
            self.Path, names=Trinity.FULL_BODY_NAMES, floor=None
        )
        glb_motion.SaveToNPZ(glb_motion.Name)
        npz_motion = Motion.LoadFromNPZ(glb_motion.Name)
        self.Motion = npz_motion
        self.Mirror = False

        self.Pose = None

        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            AssetManager.GetPath("Trinity/v3.glb"),
            Trinity.FULL_BODY_NAMES,
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
    AI4Animation(Program("MD1_X0110_Breakdance_10_A3_C3D_A.glb"))


if __name__ == "__main__":
    main()
