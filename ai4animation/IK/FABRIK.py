# Copyright (c) Meta Platforms, Inc. and affiliates.
"""FABRIK (Forward And Backward Reaching Inverse Kinematics) solver."""

from typing import List, Optional

from ai4animation.Components.Actor import Actor
from ai4animation.Math import Tensor, Transform, Vector3


class FABRIK:
    def __init__(self, source: "Actor.Bone", target: "Actor.Bone"):
        self.Bones: List["Actor.Bone"] = Actor.GetChain(source, target)
        self.Root = None
        self.Positions = Tensor.Zeros(len(self.Bones), 3)
        self.Lengths = Tensor.Zeros(len(self.Bones), 1)

    def Solve(
        self,
        position,
        rotation=None,
        max_iterations: int = 10,
        threshold: float = 0.001,
        pole_target=None,
        pole_weight: float = 1.0,
    ):
        self._prepare()
        target = Vector3.PositionTo(position, self.Root)

        pole_local = None
        if pole_target is not None:
            pole_local = Vector3.PositionTo(pole_target, self.Root)

        for _iteration in range(max_iterations):
            self._backward_pass(target)
            self._forward_pass()
            if pole_local is not None and pole_weight > 0.0:
                self._apply_pole_constraint(target, pole_local, pole_weight)

            distance_sq = Vector3.Distance(self.Positions[-1], target) ** 2
            if distance_sq < (threshold * threshold):
                break

        self._assign(rotation)

    def _prepare(self):
        self.Root = self.Bones[0].GetTransform().copy()
        for i in range(len(self.Bones)):
            self.Positions[i] = Vector3.PositionTo(
                self.Bones[i].GetPosition(), self.Root
            )
            # This should be current length but has issues at the moment...
            # self.Lengths[i] = self.Bones[i].GetDefaultLength()
            self.Lengths[i] = self.Bones[i].GetCurrentLength()

    def _backward_pass(self, target):
        # Set end effector to target
        for i in range(len(self.Bones) - 1, 0, -1):
            if i == len(self.Bones) - 1:
                self.Positions[i] = target
            else:
                self.Positions[i] = self.Positions[i + 1] + self.Lengths[
                    i + 1
                ] * Vector3.Normalize(self.Positions[i] - self.Positions[i + 1])

    def _forward_pass(self):
        # Keep root position fixed
        for i in range(1, len(self.Bones)):
            self.Positions[i] = self.Positions[i - 1] + self.Lengths[
                i
            ] * Vector3.Normalize(self.Positions[i] - self.Positions[i - 1])

    def _apply_pole_constraint(self, target, pole, weight):
        root_pos = self.Positions[0]
        chain_axis = target - root_pos
        chain_len = Vector3.Length(chain_axis)
        if chain_len < 1e-6:
            return
        chain_dir = chain_axis / chain_len

        # Project pole target onto the plane perpendicular to the chain axis
        pole_rel = pole - root_pos
        pole_proj = pole_rel - Vector3.Dot(pole_rel, chain_dir) * chain_dir
        pole_proj_len = Vector3.Length(pole_proj)
        if pole_proj_len < 1e-6:
            return
        pole_proj = pole_proj / pole_proj_len

        for i in range(1, len(self.Bones) - 1):
            joint_rel = self.Positions[i] - root_pos
            # Project joint onto the plane perpendicular to the chain axis
            joint_proj = joint_rel - Vector3.Dot(joint_rel, chain_dir) * chain_dir
            joint_proj_len = Vector3.Length(joint_proj)
            if joint_proj_len < 1e-6:
                continue
            joint_proj_norm = joint_proj / joint_proj_len

            # Compute signed angle from current projection to pole projection
            angle_deg = Vector3.SignedAngle(joint_proj_norm, pole_proj, chain_dir)
            angle_rad = Tensor.Deg2Rad(angle_deg * weight)

            # Rodrigues' rotation of the joint around the chain axis
            cos_a = Tensor.Cos(angle_rad)
            sin_a = Tensor.Sin(angle_rad)
            dot = Vector3.Dot(joint_rel, chain_dir)
            rotated = (
                cos_a * joint_rel
                + sin_a * Vector3.Cross(chain_dir, joint_rel)
                + (1 - cos_a) * dot * chain_dir
            )
            self.Positions[i] = root_pos + rotated

    def _assign(self, target_rotation):
        if target_rotation is None:
            target_rotation = self.LastBone().GetRotation()

        for i, bone in enumerate(self.Bones[:-1]):
            pos = Vector3.PositionFrom(self.Positions[i], self.Root)
            rot = bone.GetRotation()
            space = Transform.TR(pos, rot)
            bone.SetPositionAndRotation(
                pos,
                bone.ComputeAlignment(
                    space,
                    Vector3.PositionFrom(
                        Transform.GetPosition(self.Bones[i + 1].ZeroTransform), space
                    ),
                    Vector3.PositionFrom(self.Positions[i + 1], self.Root),
                ),
                FK=True,
            )
        self.Bones[-1].SetPositionAndRotation(
            Vector3.PositionFrom(self.Positions[-1], self.Root),
            target_rotation,
            FK=True,
        )
        # for bone in self.Bones:
        #     bone.RestoreAlignment()

    def FirstBone(self) -> Optional["Actor.Bone"]:
        return self.Bones[0] if self.Bones else None

    def LastBone(self) -> Optional["Actor.Bone"]:
        return self.Bones[-1] if self.Bones else None
