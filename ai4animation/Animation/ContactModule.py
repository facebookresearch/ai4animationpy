from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Tensor


class ContactModule(Module):
    def __init__(
        self, motion: Motion, configs
    ) -> (
        None
    ):  # Each config is a tuple of (boneName, heightThreshold, velocityThreshold)
        super().__init__(motion)

        self.Configs = configs
        self.BoneNames = [config[0] for config in configs]
        self.BoneIndices = self.Motion.GetBoneIndices(self.BoneNames)
        self.HeightThresholds = [config[1] for config in configs]
        self.VelocityThresholds = [config[2] for config in configs]

    def GetName(self):
        return "Contact"

    def GUI(self, editor):
        pass

    def Draw(self, editor):
        if Module.Visualize[ContactModule]:
            timestamps = editor.TimeSeries.SimulateTimestamps(editor.Timestamp)
            positions = self.Motion.GetBonePositions(
                timestamps, self.BoneIndices, editor.Mirror
            ).reshape(-1, 3)
            contacts = self.GetContacts(timestamps, editor.Mirror).reshape(-1, 1)
            for i in range(positions.shape[0]):
                if contacts[i]:
                    AI4Animation.Draw.Sphere(
                        positions[i], size=0.05, color=AI4Animation.Color.GREEN
                    )
                else:
                    AI4Animation.Draw.Sphere(
                        positions[i],
                        size=0.05,
                        color=Utility.Opacity(AI4Animation.Color.BLACK, 0.25),
                    )

    def GetContacts(self, timestamps, mirrored):
        positions = self.Motion.GetBonePositions(timestamps, self.BoneIndices, mirrored)
        velocities = self.Motion.GetBoneVelocities(
            timestamps, self.BoneIndices, mirrored
        )
        heightCriterion = positions[..., 1] < self.HeightThresholds
        velocityCriterion = (
            Tensor.Norm(velocities, keepDim=False) < self.VelocityThresholds
        )
        contacts = heightCriterion & velocityCriterion
        return contacts
