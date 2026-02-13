# Copyright (c) Meta Platforms, Inc. and affiliates.
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
    ):
        self._prepare()
        target = Vector3.PositionTo(position, self.Root)

        for _iteration in range(max_iterations):
            self._backward_pass(target)
            self._forward_pass()

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
            self.Lengths[i] = self.Bones[i].GetDefaultLength()

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
