# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation import FABRIK, Rotation, Vector3


class LegIK:
    def __init__(self, ik: FABRIK):
        self.IK = ik
        ee_pos = ik.LastBone().GetPosition().copy()
        self.EEBaseline: float = ee_pos[..., 1]

        self.TargetPosition = Vector3.Create(0, 0, 0)
        self.TargetRotation = Rotation.Euler(0, 0, 0)

        self.TargetPosition = self.IK.LastBone().GetPosition()
        self.TargetRotation = self.IK.LastBone().GetRotation()

    def Solve(
        self,
        contact: float,
        maxIterations: int,
        maxAccuracy: float,
    ):
        self.TargetPosition = Vector3.Lerp(
            self.IK.LastBone().GetPosition(), self.TargetPosition, contact
        )
        self.TargetRotation = self.IK.LastBone().GetRotation()
        self.IK.Solve(
            self.TargetPosition,
            self.TargetRotation,
            maxIterations,
            maxAccuracy,
        )
