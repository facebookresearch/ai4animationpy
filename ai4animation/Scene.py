# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation import AI4Animation, Entity
from ai4animation.Math import Tensor, Transform, Vector3


class Scene:
    def __init__(self):
        self.Entities = []
        self.Transforms = Transform.Identity(0)
        self.Scales = Vector3.One(0)

    @staticmethod
    def SetTransforms(entities, transforms):
        def FK(entity):
            if entity in dictionary.keys():
                entity.SetTransform(dictionary[entity], fk=False)
                dictionary.pop(entity)
            else:
                entity.SyncGlobal()
            for child in entity.Children:
                FK(child)

        count = 0
        dictionary = dict(zip(entities, transforms))
        while len(dictionary) > 0:
            count += 1
            FK(list(dictionary)[0])
        if count > 1:
            print(
                "SetTransforms needed to iterate more than once. Consider rearranging the layout of your dictionary."
            )

    @staticmethod
    def GetTransforms(entities_or_indices):
        if isinstance(entities_or_indices, list):
            if len(entities_or_indices) == 0:
                return Transform.Identity(0)
            if isinstance(entities_or_indices[0], int):
                indices = entities_or_indices
            if isinstance(entities_or_indices[0], Entity.Entity):
                indices = [entity.Index for entity in entities_or_indices]
            return AI4Animation.AI4Animation.Scene.Transforms[indices]

    @staticmethod
    def GetSkinningTransforms(entities_or_indices):
        if isinstance(entities_or_indices, list):
            if len(entities_or_indices) == 0:
                return Transform.Identity(0)
            if isinstance(entities_or_indices[0], int):
                indices = entities_or_indices
            if isinstance(entities_or_indices[0], Entity.Entity):
                indices = [entity.Index for entity in entities_or_indices]
            transforms = AI4Animation.AI4Animation.Scene.Transforms[indices]
            scales = AI4Animation.AI4Animation.Scene.Scales[indices]
            return Transform.TRS(
                Transform.GetPosition(transforms),
                Transform.GetRotation(transforms),
                scales,
            )

    def Update(self):
        for entity in self.Entities:
            entity.Update()

    def Standalone(self):
        self.Canvas = AI4Animation.AI4Animation.GUI.Canvas(
            "Scene", 0.01, 0.6, 0.125, 0.15
        )
        self.Button_Hierarchy = AI4Animation.AI4Animation.GUI.Button(
            "Draw Hierarchy", 0.05, 0.25, 0.9, 0.2, False, True, self.Canvas
        )
        self.Button_Handles = AI4Animation.AI4Animation.GUI.Button(
            "Draw Handles", 0.05, 0.5, 0.9, 0.2, False, True, self.Canvas
        )

    def Draw(self):
        if self.Button_Hierarchy.Active:
            self.DrawHierarchy()
        for entity in self.Entities:
            entity.Draw()

    def GUI(self):
        self.Canvas.GUI()
        self.Button_Hierarchy.GUI()
        self.Button_Handles.GUI()
        if self.Button_Handles.Active:
            self.DrawHandles()
            self.DrawLabels()
        for entity in self.Entities:
            entity.GUI()

    def AddEntity(self, name, position=None, rotation=None, parent=None):
        self.Transforms = Tensor.Concat((self.Transforms, Transform.Identity(1)), 0)
        self.Scales = Tensor.Concat((self.Scales, Vector3.One(1)), 0)
        instance = Entity.Entity(len(self.Entities), name, position, rotation, parent)
        self.Entities.append(instance)
        return instance

    def PrintHierarchy(self):
        for entity in self.Entities:
            if entity.Parent is None:
                entity.PrintHierarchy()

    def DrawHierarchy(self):
        for entity in self.Entities:
            if entity.Name != "Camera":  # Buggy for now...
                if entity.Parent is None:
                    entity.DrawHierarchy()

    def DrawHandles(self):
        for entity in self.Entities:
            if entity.Name != "Camera":  # Buggy for now...
                entity.DrawHandle()

    def DrawLabels(self):
        for entity in self.Entities:
            if entity.Name != "Camera":  # Buggy for now...
                entity.DrawLabel()
