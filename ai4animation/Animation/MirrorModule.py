# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Math import Rotation, Tensor, Transform, Vector3


class MirrorModule(Module):
    def __init__(
        self,
        motion: Motion,
        axis,  # Vector3
        correction,  # Vector3
        overrides=None,  # Dictionary Name -> Vector3
    ) -> None:
        super().__init__(motion)

        self.MirrorAxis = axis

        self.Symmetry = self.DetectSymmetry(self.Motion.Hierarchy.BoneNames)

        self.Correction = Rotation.Euler(Vector3.Zero(len(motion.Hierarchy.BoneNames)))
        for i, sym_idx in enumerate(self.Symmetry):
            self.Correction[i : i + 1] = (
                Rotation.Euler(correction) if sym_idx != i else Rotation.Euler(0, 0, 0)
            )

        if overrides is not None:
            for k, v in overrides.items():
                self.Correction[self.Motion.Hierarchy.GetBoneIndex([k])] = (
                    Rotation.Euler(v)
                )

        # self.NeedsCorrection: bool = not Tensor.All(
        #     self.Correction == Rotation.Euler(0, 0, 0)
        # )

    def Initialize(self):
        pass

    def GetName(self):
        return "Mirror"

    def GetBoneTransformations(self, frame_indices, bone_indices):
        bone_indices = [self.Symmetry[x] for x in bone_indices]
        transforms = self.Motion.Frames[frame_indices][:, bone_indices]
        transforms = Transform.GetMirror(transforms, self.MirrorAxis)
        # if self.NeedsCorrection:
        local_update = Transform.R(
            self.Correction[bone_indices],
        ).reshape(1, len(bone_indices), 4, 4)
        transforms = Transform.Multiply(transforms, local_update)
        return transforms

    def GUI(self, editor):
        if Module.Visualize[MirrorModule]:
            pass

    def Draw(self, editor):
        if Module.Visualize[MirrorModule]:
            names = editor.Actor.GetBoneNames()
            positions = self.Motion.GetBonePositions(
                editor.Timestamp,
                self.Motion.GetBoneIndices(names),
                True,
            ).reshape(-1, 3)
            AI4Animation.Draw.Skeleton(
                None, positions, editor.Actor, bones=names, size=2.0
            )

            # if self.Actor is None:
            #     self.Actor = editor.Actor.CreateCopy()
            # self.Actor.SetTransforms(
            #     self.GetBoneTransformations(
            #         self.Motion.GetFrameIndices(editor.Timestamp),
            #         self.Motion.GetBoneIndices(
            #             self.Actor.GetBoneNames(),
            #         ),
            #     )
            # )
            # self.Actor.SyncToScene()

    # Substring pairs for detecting left/right symmetry in bone names.
    # Each (left, right) entry is tried as a substring replacement.
    _SYMMETRY_SUBSTRINGS = [
        ("_l_", "_r_"),
        ("_left_", "_right_"),
        ("Left", "Right"),
    ]

    # Prefix pairs for detecting left/right symmetry via bone name prefix.
    # Each (left_prefix, right_prefix) entry is tried against the start of the name.
    _SYMMETRY_PREFIXES = [
        ("l_", "r_"),
    ]

    def DetectSymmetry(self, joint_names):
        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        symmetry = list(range(len(joint_names)))
        for i, boneName in enumerate(joint_names):
            if boneName is None:
                continue
            if self._TryAssignSymmetricName(boneName, i, name_to_idx, symmetry):
                continue
            symmetry[i] = i
        return symmetry

    def _TryAssignSymmetricName(self, boneName, bone_idx, name_to_idx, symmetry):
        for left, right in self._SYMMETRY_SUBSTRINGS:
            for src, dst in [(left, right), (right, left)]:
                if src in boneName:
                    mirror = boneName.replace(src, dst)
                    if mirror in name_to_idx:
                        symmetry[bone_idx] = name_to_idx[mirror]
                        return True

        for left, right in self._SYMMETRY_PREFIXES:
            for src, dst in [(left, right), (right, left)]:
                if boneName.startswith(src):
                    mirror = dst + boneName[len(src) :]
                    if mirror in name_to_idx:
                        symmetry[bone_idx] = name_to_idx[mirror]
                        return True

        return False
