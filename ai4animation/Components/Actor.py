from typing import List

from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Components.Component import Component
from ai4animation.Import.GLBImporter import GLB
from ai4animation.Math import Quaternion, Rotation, Tensor, Transform, Vector3


class Actor(Component):
    def Start(self, params):
        self.GLB = GLB.Create(params[0])
        self.Entities = self.CreateEntities()

        # Create Hierarchy
        self.Bones = []
        self.NameToBoneMap = {}
        boneNames = params[1]
        for i in range(len(boneNames)):
            bone = self.Bone(self, i, self.Entity.FindChild(boneNames[i]))
            self.Bones.append(bone)
            self.NameToBoneMap[bone.Entity.Name] = bone
        for bone in self.Bones:
            parent = bone.Entity.FindParent(boneNames)
            if parent:
                bone.SetParent(self.GetBone(parent.Name))

        # Initialize Values
        self.Root = self.Entity.GetTransform()
        self.Transforms = AI4Animation.Scene.GetTransforms(self.GetBoneEntityIndices())
        self.Velocities = Vector3.Zero(self.GetBoneCount())
        for bone in self.Bones:
            bone.ComputeZeroTransform()

        # Precomputed Values
        self.DefaultLengths = self.GetDefaultBoneLengths()

    def Update(self):
        pass

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

        # bones = self.GetBones(bones)
        # entites = ([self.Entity] + [bone.Entity for bone in bones]) if root else [bone.Entity for bone in bones]
        # transforms = ([self.Root] + [bone.GetTransform() for bone in bones]) if root else [bone.GetTransform() for bone in bones]
        # AI4Animation.Scene.SetTransforms(entites, transforms)

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
        c = self.DefaultLengths[children].reshape(-1, 1)
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
        if entity.Name in names:
            index = names.index(entity.Name)
            entity.SetTransform(Transforms[index])
        for child in entity.Children:
            self.AssignZeroPose(names, Transforms, child)

    def CreateEntities(self):
        names = self.GLB.JointNames
        parents = self.GLB.JointParents
        Transforms = self.GLB.JointMatrices
        entities = []
        for i in range(len(names)):
            entities.append(
                AI4Animation.Scene.AddEntity(
                    names[i], position=None, rotation=None, parent=self.Entity
                )
            )
        for i in range(len(names)):
            self.Entity.FindChild(names[i]).SetParent(
                self.Entity if parents[i] is None else self.Entity.FindChild(parents[i])
            )
        self.AssignZeroPose(names, Transforms, self.Entity)
        return entities

    def DrawHandles(self):
        for bone in self.Bones:
            bone.DrawHandle()

    def Standalone(self):
        self.Canvas = AI4Animation.GUI.Canvas("Actor", 0.01, 0.3, 0.125, 0.25)
        self.Button_Root = AI4Animation.GUI.Button(
            "Show Root", 0.05, 0.15, 0.9, 0.125, False, True, self.Canvas
        )
        self.Button_Skeleton = AI4Animation.GUI.Button(
            "Show Skeleton", 0.05, 0.3, 0.9, 0.125, False, True, self.Canvas
        )
        self.Button_Velocities = AI4Animation.GUI.Button(
            "Show Velocities", 0.05, 0.45, 0.9, 0.125, False, True, self.Canvas
        )
        self.Button_Mesh = AI4Animation.GUI.Button(
            "Show Mesh", 0.05, 0.6, 0.9, 0.125, True, True, self.Canvas
        )
        self.Button_Labels = AI4Animation.GUI.Button(
            "Show Labels", 0.05, 0.75, 0.9, 0.125, False, True, self.Canvas
        )

        self.SkinnedMesh = AI4Animation.Standalone.CreateSkinnedMesh(self, self.GLB)

    def Draw(self):
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

    def GUI(self):
        self.Canvas.GUI()
        self.Button_Root.GUI()
        self.Button_Skeleton.GUI()
        self.Button_Velocities.GUI()
        self.Button_Mesh.GUI()
        self.Button_Labels.GUI()

        if self.Button_Labels.Active:
            AI4Animation.Draw.Text3D(
                self.GetBoneNames(), self.GetPositions(), 0.01, AI4Animation.Color.BLACK
            )

        if self.Button_Mesh.IsPressed():
            if AI4Animation.Standalone.RenderPipeline.HasModel(self.SkinnedMesh.Models):
                self.SkinnedMesh.Unregister()
            else:
                self.SkinnedMesh.Register()

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

        def SetVelocity(self, value):
            Vector3.SetVector(self.Actor.Velocities, value, self.Index)

        def GetVelocity(self):
            return Vector3.GetVector(self.Actor.Velocities, self.Index)

        def ComputeZeroTransform(self):
            self.ZeroTransform = (
                Transform.Identity()
                if self.Parent is None
                else Transform.TransformationTo(
                    self.GetTransform(), self.Parent.GetTransform()
                )
            )

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
