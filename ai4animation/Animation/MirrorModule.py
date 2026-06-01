# Copyright (c) Meta Platforms, Inc. and affiliates.
from enum import Enum

from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Math import Rotation, Tensor, Transform, Vector3

# Correction

class MirrorModule(Module):
    class Map(Enum):
        Symmetric = "symmetry"
        All = "all"

    def __init__(
        self,
        motion: Motion,
        axis,  # Vector3
        correction=Vector3.Create(0,0,0),  # Vector3 for correcting local joint rotation
        map=Map.Symmetric, # Map for applying the correction across joints
        overrides={},  # Dictionary {Name -> Vector3} to override correction values for specific joints
    ) -> None:
        super().__init__(motion)

        symmetry = self.DetectSymmetry(self.Motion.Hierarchy.BoneNames)

        correctives = Rotation.Euler(Vector3.Zero(len(motion.Hierarchy.BoneNames)))
        for i, sym_idx in enumerate(symmetry):
            if map is MirrorModule.Map.Symmetric:
                if i != sym_idx:
                    correctives[i] = Rotation.Euler(correction)
            if map is MirrorModule.Map.All:
                correctives[i] = Rotation.Euler(correction)

        for k, v in overrides.items():
            correctives[self.Motion.Hierarchy.GetBoneIndex([k])] = (
                Rotation.Euler(v)
                )

        transforms = Transform.GetMirror(self.Motion.Frames, axis)
        delta = Transform.R(correctives[symmetry]).reshape(1, -1, 4, 4)
        transforms = Transform.Multiply(transforms, delta)
        self.Transforms = transforms[:, symmetry]

    def Initialize(self):
        pass

    def GetName(self):
        return "Mirror"

    def GetBoneTransformations(self, frame_indices, bone_indices):
        return self.Transforms[frame_indices][:, bone_indices]

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
        # print("Could not find symmetry for bone: " + boneName)
        return False
