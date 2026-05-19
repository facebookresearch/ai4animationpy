# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Skeletal actor component with bone hierarchy, transforms, and visualization."""

from typing import List

import numpy as np
from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Components.Component import Component
from ai4animation.Math import Quaternion, Rotation, Tensor, Transform, Vector3


class Actor(Component):
    def Start(self, params):
        model_path = params[0]
        bone_names = params[1] if len(params) > 1 else None
        self.ShowGUI = params[2] if len(params) > 2 else True

        if model_path.lower().endswith(".fbx"):
            from ai4animation.Import.FBXImporter import FBX

            self.Model = FBX.Create(model_path)
        else:
            from ai4animation.Import.GLBImporter import GLB

            self.Model = GLB.Create(model_path)

        if bone_names is None:
            bone_names = self.Model.JointNames

        self.Entities = self.CreateEntities()

        if bone_names is None:
            bone_names = self.Model.JointNames

        # Save input params
        self.ModelPath = model_path
        self.BoneNames = bone_names

        # Create missing nodes on the fly
        self.Bones = []
        self.NameToBoneMap = {}
        for i, name in enumerate(bone_names):
            entity = self.NameToEntity.get(name)
            if entity is None:
                entity = AI4Animation.Scene.AddEntity(
                    name, position=None, rotation=None, parent=self.Entity
                )
                self.NameToEntity[name] = entity

            bone = self.Bone(self, i, entity)
            self.Bones.append(bone)
            self.NameToBoneMap[name] = bone

        # Determine Parent-Child Hierarchy
        for bone in self.Bones:
            parent_entity = bone.Entity.FindParent(bone_names)
            if parent_entity is not None:
                parent_bone = self.NameToBoneMap.get(parent_entity.Name)
                if parent_bone is not None:
                    bone.SetParent(parent_bone)

        # Initialize
        self.Root = self.Entity.GetTransform()
        self.Transforms = AI4Animation.Scene.GetTransforms(self.GetBoneEntityIndices())
        self.Velocities = Vector3.Zero(self.GetBoneCount())
        for bone in self.Bones:
            bone.ComputeZeroTransform()

    def Update(self):
        pass

    def CreateCopy(self, name=None):
        entity = AI4Animation.Scene.AddEntity(
            self.Entity.Name if name is None else name
        )
        actor = entity.AddComponent(Actor, self.ModelPath, self.BoneNames, True)
        return actor

    def GetBone(self, name):
        bone = self.NameToBoneMap.get(name)
        if bone is None:
            print(f"Bone with name '{name}' could not be found.")
        return bone

    def PrintSuccessors(self, bone=None, indent=""):
        if bone is None:
            self.PrintSuccessors(self.Bones[0])
            return
        print(
            indent,
            bone.Entity.Name,
            "->",
            [self.Bones[b].Entity.Name for b in bone.Successors],
        )
        for c in bone.Children:
            self.PrintSuccessors(c, indent + "  ")

    def GetBoneNames(self):
        return list(self.NameToBoneMap.keys())

    def GetParentNames(self):
        return [
            bone.Parent.Entity.Name if bone.Parent is not None else None
            for bone in self.Bones
        ]

    def HasBone(self, name):
        return name in self.NameToBoneMap

    def GetBoneCount(self):
        return len(self.Bones)

    def GenericEvaluator(self, args, fn_default, fn_names, fn_bones, fn_indices):
        if args is None:
            return fn_default()
        if isinstance(args, list):
            if isinstance(args[0], str):
                return fn_names()
            if isinstance(args[0], self.Bone):
                return fn_bones()
            if isinstance(args[0], int):
                return fn_indices()
        print("Invalid generic type:", type(args))
        return None

    def GetBones(self, names_or_indices=None):
        return self.GenericEvaluator(
            args=names_or_indices,
            fn_default=lambda: self.Bones,
            fn_names=lambda: [
                self.GetBone(name)
                for name in names_or_indices
                if name in self.NameToBoneMap
            ],
            fn_bones=lambda: names_or_indices,
            fn_indices=lambda: self.Bones[names_or_indices],
        )

    def GetBoneIndices(self, names_or_bones=None):
        return self.GenericEvaluator(
            args=names_or_bones,
            fn_default=lambda: [item.Index for item in self.Bones],
            fn_names=lambda: [self.GetBone(item).Index for item in names_or_bones],
            fn_bones=lambda: [item.Index for item in names_or_bones],
            fn_indices=lambda: names_or_bones,
        )

    def GetParentIndices(self, names_or_bones=None):
        return self.GenericEvaluator(
            args=names_or_bones,
            fn_default=lambda: [item.GetParentIndex() for item in self.Bones],
            fn_names=lambda: [
                self.GetBone(item).GetParentIndex() for item in names_or_bones
            ],
            fn_bones=lambda: [item.GetParentIndex() for item in names_or_bones],
            fn_indices=lambda: [item.GetParentIndex() for item in names_or_bones],
        )

    def GetBoneEntityIndices(self, names_or_bones=None):
        return self.GenericEvaluator(
            args=names_or_bones,
            fn_default=lambda: [item.Entity.Index for item in self.Bones],
            fn_names=lambda: [
                self.GetBone(item).Entity.Index for item in names_or_bones
            ],
            fn_bones=lambda: [item.Entity.Index for item in names_or_bones],
            fn_indices=lambda: names_or_bones,
        )

    def GenericTensorOperation(self, args, values, op):
        return self.GenericEvaluator(
            args=args,
            fn_default=lambda: op(values),
            fn_names=lambda: op(values[self.GetBoneIndices(args)]),
            fn_bones=lambda: op(values[self.GetBoneIndices(args)]),
            fn_indices=lambda: op(values[args]),
        )

    def GetSceneTransforms(self):
        return AI4Animation.Scene.GetTransforms(self.Entities)

    def GetSceneBoneNames(self):
        return [entity.Name for entity in self.Entities]

    def GetSceneParentNames(self):
        return [entity.Parent.Name if entity.Parent is not self.Entity else None for entity in self.Entities]

    def SetTransforms(self, values, bones=None):
        Transform.SetTransform(
            self.Transforms,
            values,
            None if bones is None else self.GetBoneIndices(bones),
        )

    def GetTransforms(self, names_or_bones_or_indices=None):
        return self.GenericTensorOperation(
            names_or_bones_or_indices, self.Transforms, Transform.GetTransform
        )

    def SetPositionsAndRotations(self, positions, rotations, bones=None):
        self.SetTransforms(Transform.TR(positions, rotations), bones)

    def SetPositions(self, values, bones=None):
        Transform.SetPosition(
            self.Transforms,
            values,
            None if bones is None else self.GetBoneIndices(bones),
        )

    def GetPositions(self, names_or_bones_or_indices=None):
        return self.GenericTensorOperation(
            names_or_bones_or_indices, self.Transforms, Transform.GetPosition
        )

    def SetRotations(self, values, bones=None):
        Transform.SetRotation(
            self.Transforms,
            values,
            None if bones is None else self.GetBoneIndices(bones),
        )

    def GetRotations(self, names_or_bones_or_indices=None):
        return self.GenericTensorOperation(
            names_or_bones_or_indices, self.Transforms, Transform.GetRotation
        )

    def SetVelocities(self, values, bones=None):
        Vector3.SetVector(
            self.Velocities,
            values,
            None if bones is None else self.GetBoneIndices(bones),
        )

    def GetVelocities(self, names_or_bones_or_indices=None):
        return self.GenericTensorOperation(
            names_or_bones_or_indices, self.Velocities, Vector3.GetVector
        )

    def GetAlignments(self, names_or_bones_or_indices=None):
        bones = self.GetPositions(self.GetBoneIndices(names_or_bones_or_indices))
        parents = self.GetPositions(self.GetParentIndices(names_or_bones_or_indices))
        return Vector3.Normalize(bones - parents)

    def GetRoot(self):
        return self.Root

    def SetRoot(self, root):
        self.Root = root

    def GetRootPosition(self):
        return Transform.GetPosition(self.Root)

    def GetRootRotation(self):
        return Transform.GetRotation(self.Root)

    def GetRootDirection(self):
        return Transform.GetAxisZ(self.Root)

    def SyncToScene(self, bones=None, root=True):
        if root:
            self.Entity.SetTransform(self.Root)
        for bone in self.GetBones(bones):
            bone.Entity.SetTransform(bone.GetTransform())

    def SyncFromScene(self, bones=None, root=True):
        if root:
            self.SetRoot(self.Entity.GetTransform())
        for bone in self.GetBones(bones):
            bone.SetTransform(bone.Entity.GetTransform())

    def SearchParent(self, names, parents, current, candidates, result):
        if len(result) > 0 or current not in names:
            return
        idx = names.index(current)
        if parents[idx] in candidates:
            result.append(parents[idx])
        else:
            self.SearchParent(names, parents, parents[idx], candidates, result)

    def GetDefaultBoneLengths(self, bones=None):
        return Tensor.Create([bone.GetDefaultLength() for bone in self.GetBones(bones)])

    def GetCurrentBoneLengths(self, bones=None):
        return Tensor.Create([bone.GetCurrentLength() for bone in self.GetBones(bones)])

    def RestoreBoneLengths(self, bones=None):
        bones = self.GetBones(bones)
        parents = self.GetParentIndices(bones)
        children = self.GetBoneIndices(bones)
        a = self.GetPositions(parents)
        b = self.GetPositions(children)
        c = self.GetDefaultBoneLengths()[children].reshape(-1, 1)
        d = a + c * Vector3.Normalize(b - a)
        self.SetPositions(d, children)

    def SetBoneLengths(self, values, bones=None):
        bones = self.GetBones(bones)
        parents = self.GetParentIndices(bones)
        children = self.GetBoneIndices(bones)

        a = self.GetPositions(parents)
        b = self.GetPositions(children)
        c = values.reshape(-1, 1)
        d = a + c * Vector3.Normalize(b - a)
        self.SetPositions(d, children)

    def RestoreBoneAlignments(self, bones=None):
        bones = self.GetBones(bones)
        for bone in bones:
            bone.RestoreAlignment()

        # Filter bones with single children
        # bone_indices = []
        # child_indices = []
        # single_child_bones = []
        # for bone in bones:
        #     if len(bone.Children) == 1:
        #         bone_indices.append(bone.Index)
        #         child_indices.append(bone.Children[0].Index)
        #         single_child_bones.append(bone)

        # if not single_child_bones:
        #     return
        # bone_Transforms = self.GetTransforms(bone_indices)
        # child_Transforms = self.GetTransforms(child_indices)
        # bone_positions = Transform.GetPosition(bone_Transforms)
        # bone_rotations = Transform.GetRotation(bone_Transforms)
        # child_positions = Transform.GetPosition(child_Transforms)

        # child_zero_positions = Vector3.Zero(len(single_child_bones))
        # for i, bone in enumerate(single_child_bones):
        #     child_zero_positions[i] = Transform.GetPosition(bone.Children[0].ZeroTransform)
        # zero_positions = bone_positions + Rotation.MultiplyVector(bone_rotations, child_zero_positions)

        # rotations = Rotation.Multiply(
        #         Rotation.RotationFromTo(
        #             zero_positions - bone_positions,
        #             child_positions - bone_positions
        #         ),
        #         bone_rotations
        #     )
        # self.SetRotations(rotations, bone_indices)

    def GetChain(source: "Actor.Bone", target: "Actor.Bone") -> List["Actor.Bone"]:
        chain = []
        pivot = target
        chain.append(pivot)

        while pivot != source:
            if pivot.Parent is None:
                print(
                    f"Chain from {source.Entity.Name} to {target.Entity.Name} could not be found."
                )
                return []
            else:
                pivot = pivot.Parent
                chain.append(pivot)
        chain.reverse()
        return chain

    def AssignZeroPose(self, names, Transforms, entity=None):
        if entity is None:
            entity = self.Entity

        name_to_index = {n: i for i, n in enumerate(names)}
        stack = [entity]
        while stack:
            ent = stack.pop()
            idx = name_to_index.get(ent.Name)
            if idx is not None:
                AI4Animation.Scene.Transforms[ent.Index] = Transforms[idx]
            stack.extend(ent.Children)

    def CreateEntities(self):
        names, parents, transforms = (
            self.Model.JointNames,
            self.Model.JointParents,
            self.Model.JointMatrices,
        )

        self.NameToEntity = {
            name: AI4Animation.Scene.AddEntity(
                name, position=None, rotation=None, parent=self.Entity
            )
            for name in names
        }
        entities = list(self.NameToEntity.values())

        for name, parent_name in zip(names, parents):
            parent = self.NameToEntity.get(parent_name)
            if parent is not None:
                self.NameToEntity[name].SetParent(parent)

        self.AssignZeroPose(names, transforms, self.Entity)
        return entities

    def DrawHandle(self):
        self.Entity.DrawHandle()

    def DrawHandles(self):
        for bone in self.Bones:
            bone.DrawHandle()

    def ShowMesh(self, value):
        if value:
            self.SkinnedMesh.Register()
        else:
            if AI4Animation.Standalone.RenderPipeline.HasModel(self.SkinnedMesh.Models):
                self.SkinnedMesh.Unregister()

    def ToggleMesh(self):
        if AI4Animation.Standalone.RenderPipeline.HasModel(self.SkinnedMesh.Models):
            self.SkinnedMesh.Unregister()
        else:
            self.SkinnedMesh.Register()

    def Standalone(self):
        self.SkinnedMesh = AI4Animation.Standalone.CreateSkinnedMesh(self, self.Model)

        if not self.ShowGUI:
            self.Canvas = None
            self.Button_Root = None
            self.Button_Skeleton = None
            self.Button_Velocities = None
            self.Button_Mesh = None
            self.Button_Labels = None
            self.Button_Hierarchy = None
            return

        self.Canvas = AI4Animation.GUI.Canvas("Actor", 0.01, 0.3, 0.125, 0.25)
        tuples = [
            ("Button_Root", "Show Root"),
            ("Button_Skeleton", "Show Skeleton"),
            ("Button_Velocities", "Show Velocities"),
            ("Button_Mesh", "Show Mesh"),
            ("Button_Labels", "Show Labels"),
            ("Button_Hierarchy", "Show Hierarchy"),
        ]
        item = 0
        for button, text in tuples:
            setattr(self, button, AI4Animation.GUI.Button(text, 0.05, 0.15 + item/(len(tuples)+1), 0.9, 0.65/len(tuples), False, True, self.Canvas))
            item += 1

    def Draw(self):
        if not self.ShowGUI:
            return

        boneSize = 0.0175
        if self.Button_Root.Active:
            AI4Animation.Draw.Transform(self.Root, 0.5)
        if self.Button_Skeleton.Active:
            AI4Animation.Draw.Transform(self.Transforms, 0.25)
            AI4Animation.Draw.Cylinder(
                self.GetPositions(self.GetParentIndices()),
                self.GetPositions(self.GetBoneIndices()),
                boneSize,
                0.0,
                10,
                Utility.Opacity(AI4Animation.Color.BLACK, 0.75),
            )
        if self.Button_Velocities.Active:
            AI4Animation.Draw.Vector(
                self.GetPositions(),
                self.Velocities,
                boneSize,
                Utility.Opacity(AI4Animation.Color.BLUE, 0.5),
            )
        if self.Button_Hierarchy.Active:
            self.Entity.DrawHierarchy()

    def GUI(self):
        if not self.ShowGUI:
            return

        self.Canvas.GUI()
        self.Button_Root.GUI()
        self.Button_Skeleton.GUI()
        self.Button_Velocities.GUI()
        self.Button_Mesh.GUI()
        self.Button_Labels.GUI()
        self.Button_Hierarchy.GUI()

        if self.Button_Labels.Active:
            AI4Animation.Draw.Text3D(
                self.GetBoneNames(), self.GetPositions(), 0.01, AI4Animation.Color.BLACK
            )

        if self.Button_Mesh.IsPressed():
            self.ToggleMesh()

    class Bone:
        def __init__(self, actor, index, entity):
            self.Actor = actor
            self.Index = index
            self.Entity = entity
            self.Parent = None
            self.Children = []
            self.Successors = []
            self.ZeroTransform = Transform.Identity()

        def SetTransform(self, value, FK=False):
            if FK:
                tmp = Transform.TransformationTo(
                    self.Actor.Transforms[self.Successors], self.GetTransform()
                )
                self.SetTransform(value)
                self.Actor.Transforms[self.Successors] = Transform.TransformationFrom(
                    tmp, self.GetTransform()
                )
            else:
                self.Actor.Transforms[self.Index] = value

        def GetTransform(self):
            return Transform.GetTransform(self.Actor.Transforms, self.Index)

        def SetPositionAndRotation(self, position, rotation, FK=False):
            if FK:
                tmp = Transform.TransformationTo(
                    self.Actor.Transforms[self.Successors], self.GetTransform()
                )
                self.SetPosition(position)
                self.SetRotation(rotation)
                self.Actor.Transforms[self.Successors] = Transform.TransformationFrom(
                    tmp, self.GetTransform()
                )
            else:
                self.SetPosition(position)
                self.SetRotation(rotation)

        def SetPosition(self, value, FK=False):
            if FK:
                tmp = Transform.TransformationTo(
                    self.Actor.Transforms[self.Successors], self.GetTransform()
                )
                self.SetPosition(value)
                self.Actor.Transforms[self.Successors] = Transform.TransformationFrom(
                    tmp, self.GetTransform()
                )
            else:
                Transform.SetPosition(self.Actor.Transforms, value, self.Index)

        def GetPosition(self):
            return Transform.GetPosition(self.Actor.Transforms, self.Index)

        def SetRotation(self, value, FK=False):
            if FK:
                tmp = Transform.TransformationTo(
                    self.Actor.Transforms[self.Successors], self.GetTransform()
                )
                self.SetRotation(value)
                self.Actor.Transforms[self.Successors] = Transform.TransformationFrom(
                    tmp, self.GetTransform()
                )
            else:
                self.Actor.Transforms[self.Index, :3, :3] = value
                # Transform.SetRotation(self.Actor.Transforms, value, self.Index)

        def GetRotation(self):
            return Transform.GetRotation(self.Actor.Transforms, self.Index)

        def SetLocalRotation(self, value, FK=False):
            if self.Parent is None:
                self.SetRotation(value, FK)
            else:
                global_rotation = Rotation.RotationFrom(
                    value, self.Parent.GetTransform()
                )
                self.SetRotation(global_rotation, FK)

        def GetLocalRotation(self):
            if self.Parent is None:
                return self.GetRotation()
            return Rotation.RotationTo(self.GetRotation(), self.Parent.GetTransform())

        def SetVelocity(self, value):
            Vector3.SetVector(self.Actor.Velocities, value, self.Index)

        def GetVelocity(self):
            return Vector3.GetVector(self.Actor.Velocities, self.Index)

        def ComputeZeroTransform(self):
            if self.Parent is None:
                self.ZeroTransform = Transform.Identity()
                return

            try:
                self.ZeroTransform = Transform.TransformationTo(
                    self.GetTransform(), self.Parent.GetTransform()
                )
            except np.linalg.LinAlgError as e:
                # Some models (especially GLB exports) may introduce singular or
                # degenerate parent transforms (e.g. zero scale), which makes the
                # inverse undefined. In that case we fall back to identity.
                print(
                    f"Warning: failed to compute zero pose for bone '{self.Entity.Name}' due to {type(e).__name__}: {e}"
                )
                self.ZeroTransform = Transform.Identity()

        def GetZeroLocalRotation(self):
            return Transform.GetRotation(self.ZeroTransform)

        def GetCurrentLength(self):
            return (
                0.0
                if self.Parent is None
                else Vector3.Distance(self.Parent.GetPosition(), self.GetPosition())[0]
            )

        def GetDefaultLength(self):
            return Vector3.Length(Transform.GetPosition(self.ZeroTransform))[0]

        def SetLength(self, value):
            if self.Parent is not None:
                current = self.GetPosition()
                parent = self.Parent.GetPosition()
                self.SetPosition(parent + value * Vector3.Normalize(current - parent))

        def RestoreLength(self):
            self.SetLength(self.GetDefaultLength())

        def RestoreAlignment(self):
            if len(self.Children) == 1:
                self.SetRotation(
                    self.ComputeAlignment(
                        self.GetTransform(),
                        Vector3.PositionFrom(
                            Transform.GetPosition(self.Children[0].ZeroTransform),
                            self.GetTransform(),
                        ),
                        self.Children[0].GetPosition(),
                    )
                )

        def ComputeAlignment(self, source, from_pos, to_pos):
            return Rotation.Multiply(
                Quaternion.ToMatrix(
                    Quaternion.FromTo(
                        from_pos - Transform.GetPosition(source),
                        to_pos - Transform.GetPosition(source),
                    )
                ),
                Transform.GetRotation(source),
            )

        def SetParent(self, parent):
            if self.Parent is not None:
                self.Parent.Children.remove(self)
            if parent is not None:
                parent.Children.append(self)
            self.Parent = parent
            if self.Parent is not None:
                self.Parent.AddSuccessor(self)

        def GetParentIndex(self):
            return self.Index if self.Parent is None else self.Parent.Index

        def AddSuccessor(self, bone):
            self.Successors.append(bone.Index)
            if self.Parent is not None:
                self.Parent.AddSuccessor(bone)

        def DrawHandle(self):
            self.Entity.DrawHandle()
