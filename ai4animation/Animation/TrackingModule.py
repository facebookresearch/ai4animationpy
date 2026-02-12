from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.MotionModule import MotionModule
from ai4animation.Animation.RootModule import RootModule
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Tensor, Transform, Vector3


class TrackingModule(Module):
    def __init__(
        self, motion: Motion, head_name, left_wrist_name, right_wrist_name
    ) -> None:
        super().__init__(motion)
        self.TrackerNames = [head_name, left_wrist_name, right_wrist_name]
        self.TrackerIndices = motion.GetBoneIndices(self.TrackerNames)

        self.RootModule = None
        self.MotionModule = None

    @property
    def HeadName(self):
        return self.TrackerNames[0]

    @property
    def HeadIndex(self):
        return self.TrackerIndices[0]

    @property
    def LeftWristName(self):
        return self.TrackerNames[1]

    @property
    def LeftWristIndex(self):
        return self.TrackerIndices[1]

    @property
    def RightWristName(self):
        return self.TrackerNames[2]

    @property
    def RightWristIndex(self):
        return self.TrackerIndices[2]

    def GetName(self):
        return "Tracking"

    def GetRootModule(self):
        if self.RootModule is None:
            self.RootModule = self.Motion.GetModule(RootModule)
        return self.RootModule

    def GetMotionModule(self):
        if self.MotionModule is None:
            self.MotionModule = self.Motion.GetModule(MotionModule)
        return self.MotionModule

    def ComputeSeries(
        self,
        timestamp: float,
        mirrored: bool,
        timeseries: TimeSeries,
        smoothing: TimeSeries = None,
    ):
        timestamps = timeseries.SimulateTimestamps(timestamp)
        instance = self.Series(
            timeseries,
            self.TrackerNames,
            self.GetTransforms(timestamps, mirrored, smoothing),
            self.GetVelocities(timestamps, mirrored, smoothing),
        )
        return instance

    def GetReference(self, timestamps, mirrored, smoothing: TimeSeries = None):
        position = self.GetMotionModule().GetPositions(
            timestamps, mirrored, self.HeadIndex
        )
        position[..., 1] = 0.0
        rotation = self.GetRootModule().GetRotations(timestamps, mirrored, smoothing)
        # The most hacky thing ever
        position = Tensor.Squeeze(position, -2)
        while position.shape[0] == 1:
            position = Tensor.Squeeze(position, 0)
        return Transform.TR(position, rotation)

    def GetTransforms(self, timestamps, mirrored, smoothing: TimeSeries = None):
        if smoothing is not None and smoothing.Window > 0.0:
            timestamps = Tensor.Unsqueeze(timestamps, -1) + smoothing.Timestamps
            axis = len(timestamps.shape) - 1
            transforms = self.GetTransforms(timestamps, mirrored)
            transforms = Tensor.Squeeze(
                Tensor.Gaussian(transforms, power=1.0, axis=axis), axis
            )  # This might be buggy, need to renormalize matrix
            return transforms
        else:
            return self.Motion.GetBoneTransformations(
                timestamps, self.TrackerIndices, mirrored
            )

    def GetVelocities(self, timestamps, mirrored, smoothing: TimeSeries = None):
        if smoothing is not None and smoothing.Window > 0.0:
            timestamps = Tensor.Unsqueeze(timestamps, -1) + smoothing.Timestamps
            axis = len(timestamps.shape) - 1
            velocities = self.GetVelocities(timestamps, mirrored)
            velocities = Tensor.Squeeze(
                Tensor.Gaussian(velocities, power=1.0, axis=axis), axis
            )
            return velocities
        else:
            return self.Motion.GetBoneVelocities(
                timestamps, self.TrackerIndices, mirrored
            )

    def Standalone(self):
        self.Button_Smooth = AI4Animation.GUI.Button(
            "Smoothed", 0.45, 0.25, 0.1, 0.05, False, True
        )
        self.Slider_Smooth = AI4Animation.GUI.Slider(
            0.45, 0.3, 0.1, 0.05, 1.0, 0.0, 2.0
        )

    def GUI(self, editor):
        if Module.Visualize[TrackingModule]:
            self.Button_Smooth.GUI()
            self.Slider_Smooth.GUI()

    def Draw(self, editor):
        if Module.Visualize[TrackingModule]:
            window = self.Slider_Smooth.GetValue()
            self.ComputeSeries(
                editor.Timestamp,
                editor.Mirror,
                editor.TimeSeries,
                TimeSeries(-window / 2, window / 2, editor.TimeSeries.SampleCount)
                if self.Button_Smooth.Active
                else None,
            ).Draw()

    class Series(TimeSeries):
        def __init__(self, timeSeries, names, transforms=None, velocities=None):
            super().__init__(timeSeries.Start, timeSeries.End, timeSeries.SampleCount)
            self.Names = names
            self.NameToIndexMap = {}
            for i in range(len(names)):
                self.NameToIndexMap[names[i]] = i

            self.Transforms = (
                Transform.Identity((self.SampleCount, self.TrajectoryCount))
                if transforms is None
                else transforms
            )
            self.Velocities = (
                Vector3.Zero((self.SampleCount, self.TrajectoryCount))
                if velocities is None
                else velocities
            )

        @property
        def TrajectoryCount(self) -> int:
            return len(self.Names)

        def GetTransforms(self, bone_names=None, start=None, end=None):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            if bone_names == None:
                return self.Transforms[start:end]
            else:
                bone_indices = [self.NameToIndexMap[name] for name in bone_names]
                return self.Transforms[start:end, bone_indices, :, :]

        def GetPositions(self, bone_names=None, start=None, end=None):
            return Transform.GetPosition(self.GetTransforms(bone_names, start, end))

        def GetRotations(self, bone_names=None, start=None, end=None):
            return Transform.GetRotation(self.GetTransforms(bone_names, start, end))

        def GetVelocities(self, bone_names=None, start=None, end=None):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            if bone_names == None:
                return self.Velocities[start:end]
            else:
                bone_indices = [self.NameToIndexMap[name] for name in bone_names]
                return self.Velocities[start:end, bone_indices, :]

        def Draw(
            self,
            start=None,
            end=None,
            thickness=1.0,
            drawConnections=True,
            drawPositions=True,
            drawVelocities=True,
            positionColor=None,
            velocityColor=None,
            actor=None,
        ):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            if actor is None:
                for i in range(len(self.Names)):
                    positions = Transform.GetPosition(
                        self.Transforms[start:end, i, :, :]
                    )
                    velocities = self.Velocities[start:end, i, :]
                    positionColor = (
                        AI4Animation.Color.BLACK
                        if positionColor is None
                        else positionColor
                    )
                    velocityColor = Utility.Opacity(
                        AI4Animation.Color.GREEN
                        if velocityColor is None
                        else velocityColor,
                        0.5,
                    )
                    if drawConnections:
                        AI4Animation.Draw.LineStrip(positions, color=positionColor)
                    if drawPositions:
                        AI4Animation.Draw.Sphere(
                            positions, 0.02 * thickness, color=positionColor
                        )
                    if drawVelocities:
                        AI4Animation.Draw.Vector(
                            positions,
                            velocities,
                            0.005 * thickness,
                            color=velocityColor,
                        )
            else:
                for i in range(start, end, 1):
                    AI4Animation.Draw.Skeleton(
                        None,
                        Transform.GetPosition(self.Transforms[i]),
                        actor,
                        bones=self.Names,
                    )
