# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Animation.TimeSeries import TimeSeries
from ai4animation.Math import Rotation, Tensor, Transform, Vector3


class RootModule(Module):
    def __init__(
        self,
        motion: Motion,
        hip,
        left_hip,
        right_hip,
        left_shoulder,
        right_shoulder,
    ) -> None:
        super().__init__(motion)
        self.BoneIndices = motion.GetBoneIndices(
            [hip, left_hip, right_hip, left_shoulder, right_shoulder]
        )
        self.Hip, self.LeftHip, self.RightHip, self.LeftShoulder, self.RightShoulder = (
            self.BoneIndices
        )

        self.StandardMatrices = self.Compute(False)
        self.MirroredMatrices = self.Compute(True)

    def GetName(self):
        return "Root"

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
            self.GetTransforms(timestamps, mirrored, smoothing),
            self.GetVelocities(timestamps, mirrored, smoothing),
        )
        return instance

    def GetTransforms(self, timestamps, mirrored: bool, smoothing: TimeSeries = None):
        if smoothing is not None and smoothing.Window > 0.0:
            timestamps = Tensor.Unsqueeze(timestamps, -1) + smoothing.Timestamps
            axis = len(timestamps.shape) - 1

            matrices = self.GetTransforms(timestamps, mirrored)

            positions = Transform.GetPosition(matrices)
            directions = Transform.GetAxisZ(matrices)

            center = positions[..., int((smoothing.SampleCount - 1) / 2), :]
            positions = (
                Tensor.Squeeze(
                    Tensor.Gaussian(
                        positions - Tensor.Unsqueeze(center, axis), power=2.0, axis=axis
                    ),
                    axis,
                )
                + center
            )

            angles = Vector3.SignedAngle(
                directions[..., 0:-1, :],
                directions[..., 1:, :],
                Vector3.Create(0, 1, 0),
            ) / (timestamps[..., 1:] - timestamps[..., 0:-1])
            powers = Tensor.Deg2Rad(Tensor.Abs(Tensor.Sum(angles, -1)))
            directions = Tensor.Squeeze(
                Tensor.Gaussian(directions, power=powers, axis=axis), axis
            )

            matrices = Transform.TR(positions, Rotation.LookPlanar(directions))
            return matrices
        else:
            frame_indices = self.Motion.GetFrameIndices(timestamps)
            return (
                self.StandardMatrices[frame_indices]
                if not mirrored
                else self.MirroredMatrices[frame_indices]
            )

    def GetPositions(self, timestamps, mirrored: bool, smoothing: TimeSeries = None):
        return Transform.GetPosition(
            self.GetTransforms(timestamps, mirrored, smoothing)
        )

    def GetRotations(self, timestamps, mirrored: bool, smoothing: TimeSeries = None):
        return Transform.GetRotation(
            self.GetTransforms(timestamps, mirrored, smoothing)
        )

    def GetVelocities(self, timestamps, mirrored: bool, smoothing: TimeSeries = None):
        t_previous = Tensor.Clamp(
            timestamps - self.Motion.DeltaTime,
            0.0,
            self.Motion.TotalTime - self.Motion.DeltaTime,
        )
        t_current = Tensor.Clamp(
            timestamps, self.Motion.DeltaTime, self.Motion.TotalTime
        )
        pos_previous = self.GetPositions(t_previous, mirrored, smoothing)
        pos_current = self.GetPositions(t_current, mirrored, smoothing)
        return (pos_current - pos_previous) / self.Motion.DeltaTime

        # timestamps = Tensor.Clamp(Tensor.Concat((timestamps[...,:1] - self.Motion.DeltaTime, timestamps), -1), 0, self.Motion.TotalTime)
        # positions = self.GetPositions(timestamps, mirrored, smoothing)
        # print(timestamps)
        # prev = positions[..., :-1, :]
        # next = positions[..., 1:, :]
        # velocities = (next - prev) / self.Motion.DeltaTime
        # return velocities

    def GetDeltaTransforms(
        self,
        timestamps=None,
        mirrored=False,
        deltaTime=None,
        smoothing: TimeSeries = None,
    ):
        dt = deltaTime if deltaTime is None else self.Motion.DeltaTime
        timestamps = Tensor.Clamp(
            Tensor.Concat((timestamps[..., :1] - dt, timestamps), -1),
            0,
            self.Motion.TotalTime,
        )
        transforms = self.GetTransforms(timestamps, mirrored, smoothing)
        prev = transforms[..., :-1, :, :]
        next = transforms[..., 1:, :, :]
        delta = Transform.TransformationTo(next, prev)
        return delta

    def GetDeltaVectors(
        self,
        timestamps=None,
        mirrored=False,
        deltaTime=None,
        smoothing: TimeSeries = None,
    ):
        delta = self.GetDeltaTransforms(timestamps, mirrored, deltaTime, smoothing)
        pos = Transform.GetPosition(delta)
        rot = Transform.GetRotation(delta)
        x = pos[..., 0]
        z = pos[..., 2]
        y = Vector3.SignedAngle(Vector3.Z, Rotation.GetAxisZ(rot), Vector3.Y)
        vec = Tensor.Stack((x, y, z), -1)
        return vec

    def Compute(self, mirrored):
        bone_transformations = self.Motion.GetBoneTransformations(
            timestamps=None, bone_names_or_indices=self.BoneIndices, mirrored=mirrored
        )
        bone_positions = Transform.GetPosition(bone_transformations)
        hip_pos = bone_positions[..., 0, :]
        left_hip_pos = bone_positions[..., 1, :]
        right_hip_pos = bone_positions[..., 2, :]
        left_shoulder_pos = bone_positions[..., 3, :]
        right_shoulder_pos = bone_positions[..., 4, :]

        # root positions
        root_positions = Tensor.ZerosLike(hip_pos)
        root_positions[..., 0] = hip_pos[..., 0]
        root_positions[..., 1] = 0.0
        root_positions[..., 2] = hip_pos[..., 2]

        # root rotations
        up_batch = Tensor.ZerosLike(hip_pos)
        up_batch[..., 1] = 1.0
        hip_batch = left_hip_pos - right_hip_pos
        shoulder_batch = left_shoulder_pos - right_shoulder_pos

        # Project on horizontal plane: v - (v·up)up
        hip_dot_up = Tensor.Unsqueeze(Tensor.Dot(hip_batch, up_batch), -1)
        hip_batch_projected = hip_batch - hip_dot_up * up_batch
        hip_batch_normalized = Tensor.Normalize(hip_batch_projected)
        shoulder_dot_up = Tensor.Unsqueeze(Tensor.Dot(shoulder_batch, up_batch), -1)
        shoulder_batch_projected = shoulder_batch - shoulder_dot_up * up_batch
        shoulder_batch_normalized = Tensor.Normalize(shoulder_batch_projected)

        averaged_batch = (hip_batch_normalized + shoulder_batch_normalized) * 0.5
        averaged_batch_normalized = Tensor.Normalize(averaged_batch)

        forward_batch = Tensor.Cross(averaged_batch_normalized, up_batch)

        # Project forward on horizontal plane and normalize
        forward_dot_up = Tensor.Unsqueeze(Tensor.Dot(forward_batch, up_batch), -1)
        forward_batch = forward_batch - forward_dot_up * up_batch
        forward_batch = Tensor.Normalize(forward_batch)

        root_rotations = Rotation.Look(forward_batch, up_batch)

        return Transform.TR(root_positions, root_rotations)

    def Callback(self, editor):
        if editor.Actor:
            editor.Actor.Root = self.GetTransforms(editor.Timestamp, editor.Mirror)

    def Standalone(self):
        self.Button_Smooth = AI4Animation.GUI.Button(
            "Smoothed", 0.45, 0.25, 0.1, 0.05, False, True
        )

    def GUI(self, editor):
        if Module.Visualize[RootModule]:
            self.Button_Smooth.GUI()

    def Draw(self, editor):
        if Module.Visualize[RootModule]:
            self.ComputeSeries(
                editor.Timestamp,
                editor.Mirror,
                editor.TimeSeries,
                TimeSeries(-1.0, 1.0, editor.TimeSeries.SampleCount)
                if self.Button_Smooth.Active
                else None,
            ).Draw()

    class Series(TimeSeries):
        def __init__(self, timeSeries, transforms=None, velocities=None):
            super().__init__(timeSeries.Start, timeSeries.End, timeSeries.SampleCount)

            self.Transforms = (
                Transform.Identity(self.SampleCount)
                if transforms is None
                else transforms
            )
            self.Velocities = (
                Vector3.Zero(self.SampleCount) if velocities is None else velocities
            )

        def Draw(
            self,
            start=None,
            end=None,
            drawPositions=True,
            drawDirections=True,
            drawVelocities=True,
            positionColor=None,
            directionColor=None,
            velocityColor=None,
        ):
            start = 0 if start is None else start
            end = self.SampleCount if end is None else end

            positionColor = (
                AI4Animation.Color.BLACK if positionColor is None else positionColor
            )
            directionColor = (
                AI4Animation.Color.ORANGE if directionColor is None else directionColor
            )
            velocityColor = Utility.Opacity(
                AI4Animation.Color.GREEN if velocityColor is None else velocityColor,
                0.5,
            )

            positions = Transform.GetPosition(self.Transforms[start:end])
            directions = Transform.GetAxisZ(self.Transforms[start:end])
            velocities = self.Velocities[start:end]
            AI4Animation.Draw.LineStrip(positions, color=positionColor)
            if drawPositions:
                AI4Animation.Draw.Sphere(positions, size=0.025, color=positionColor)
            if drawDirections:
                AI4Animation.Draw.Cylinder(
                    positions,
                    positions + 0.5 * directions,
                    0.025,
                    0,
                    1,
                    color=directionColor,
                )
            if drawVelocities:
                AI4Animation.Draw.Vector(
                    positions, velocities, 0.025, color=velocityColor
                )

        def SetPosition(self, value, index):
            Transform.SetPosition(self.Transforms, value, index)

        def GetPosition(self, index):
            return Transform.GetPosition(self.Transforms, index)

        def SetDirection(self, value, index):
            Transform.SetRotation(self.Transforms, Rotation.LookPlanar(value), index)

        def GetDirection(self, index):
            return Transform.GetAxisZ(self.Transforms, index)

        def SetVelocity(self, value, index):
            Vector3.SetVector(self.Velocities, value, index)

        def GetVelocity(self, index):
            return Vector3.GetVector(self.Velocities, index)

        def GetLength(self):
            prev = Transform.GetPosition(self.Transforms)[..., :-1, :]
            next = Transform.GetPosition(self.Transforms)[..., 1:, :]
            length = Tensor.Sum(Tensor.Norm(next - prev, keepDim=False))
            return length

        def Control(
            self,
            position,
            direction,
            velocity,
            deltaTime,
            moveSensitivty=10.0,
            turnSensitivity=10.0,
        ):
            pivot = 0
            direction = Vector3.Normalize(direction)
            if Vector3.Length(direction) == 0.0:
                if Vector3.Length(velocity) != 0.0:
                    direction = Vector3.Normalize(velocity)
                else:
                    direction = self.GetDirection(pivot)
            self.SetVelocity(
                Vector3.LerpDt(
                    self.GetVelocity(pivot), velocity, deltaTime, moveSensitivty
                ),
                pivot,
            )
            self.SetPosition(position + self.GetVelocity(pivot) * deltaTime, pivot)
            self.SetDirection(
                Vector3.SlerpDt(
                    self.GetDirection(pivot), direction, deltaTime, turnSensitivity
                ),
                pivot,
            )

            for index in range(pivot + 1, self.SampleCount):
                ratio = Utility.Ratio(index, pivot, self.SampleCount - 1)
                self.SetVelocity(
                    Vector3.LerpDt(
                        self.GetVelocity(index - 1),
                        velocity,
                        self.DeltaTime,
                        ratio * moveSensitivty,
                    ),
                    index,
                )
                self.SetPosition(
                    self.GetPosition(index - 1)
                    + self.GetVelocity(index) * self.DeltaTime,
                    index,
                )
                self.SetDirection(
                    Vector3.Slerp(self.GetDirection(pivot), direction, ratio), index
                )
