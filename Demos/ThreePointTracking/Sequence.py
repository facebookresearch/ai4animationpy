# Copyright (c) Meta Platforms, Inc. and affiliates.
import math

from ai4animation import AI4Animation, Utility
from ai4animation.Math import Rotation, Tensor, Transform, Vector3


class Sequence:
    def __init__(self):
        self.Timestamps = None
        self.Trajectory = None
        self.Motion = None
        self.Contacts = None
        self.RootLock = 0.0

    def SampleRoot(self, timestamp: float):
        a, b, w = self.GetIndexPair(timestamp)
        return Transform.Interpolate(
            self.Trajectory.Transforms[a], self.Trajectory.Transforms[b], w
        )

    def SamplePositions(self, timestamp: float):
        a, b, w = self.GetIndexPair(timestamp)
        return Vector3.Lerp(
            Transform.GetPosition(self.Motion.Transforms[a]),
            Transform.GetPosition(self.Motion.Transforms[b]),
            w,
        )

    def SampleRotations(self, timestamp: float):
        a, b, w = self.GetIndexPair(timestamp)
        return Rotation.Interpolate(
            Transform.GetRotation(self.Motion.Transforms[a]),
            Transform.GetRotation(self.Motion.Transforms[b]),
            w,
        )

    def SampleVelocities(self, timestamp: float):
        a, b, w = self.GetIndexPair(timestamp)
        return Vector3.Lerp(
            self.Motion.Velocities[a],
            self.Motion.Velocities[b],
            w,
        )

    def SampleContacts(self, timestamp: float):
        a, b, w = self.GetIndexPair(timestamp)
        return Tensor.Interpolate(
            self.Contacts[a],
            self.Contacts[b],
            w,
        )

    def GetRootLock(self):
        return 1.0 if Tensor.Mean(Tensor.Flatten(self.Contacts)) > 0.75 else 0.0

    def GetIndexPair(self, timestamp: float):
        ratio = Utility.Normalize(
            timestamp,
            self.Timestamps[0],
            self.Timestamps[-1],
            0.0,
            self.Timestamps.size - 1,
        )
        ratio = Tensor.Clamp(ratio, 0.0, self.Timestamps.size - 1)
        a = int(math.floor(ratio))
        b = int(math.ceil(ratio))
        w = Utility.Ratio(timestamp, self.Timestamps[a], self.Timestamps[b])
        w = Tensor.Clamp(w, 0.0, 1.0)
        return a, b, w

    def GetLength(self):
        return self.Trajectory.GetLength()

    def Draw(self, actor, color=None):
        for i in range(self.Timestamps.size):
            AI4Animation.Draw.Skeleton(
                self.Trajectory.Transforms[i],
                Transform.GetPosition(self.Motion.Transforms[i]),
                actor,
                color=color,
            )
