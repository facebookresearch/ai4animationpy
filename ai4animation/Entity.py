# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation import AI4Animation
from ai4animation.Math import Transform


class Entity:
    def __init__(self, index, name, position=None, rotation=None, parent=None):
        self.Index = index
        self.Name = name
        self.Parent = None
        self.Children = []
        self.Successors = []
        self.Components = {}
        self.SetParent(parent)
        if position is not None and rotation is not None:
            self.SetPositionAndRotation(position, rotation)
        else:
            if position is not None:
                self.SetPosition(position)
            if rotation is not None:
                self.SetRotation(rotation)
        if AI4Animation.AI4Animation.Standalone is not None:
            self.Standalone()

    def Update(self):
        for c in self.Components.values():
            c.Update()

    def Standalone(self):
        self.Handle = AI4Animation.AI4Animation.GUI.Handle(self)

    def Draw(self):
        for c in self.Components.values():
            c.Draw()

    def GUI(self):
        for c in self.Components.values():
            c.GUI()

    def SetTransform(self, value, fk=True):
        delta = Transform.TransformationTo(
            AI4Animation.AI4Animation.Scene.Transforms[self.Successors],
            self.GetTransform(),
        )
        AI4Animation.AI4Animation.Scene.Transforms[self.Index] = value
        AI4Animation.AI4Animation.Scene.Transforms[self.Successors] = (
            Transform.TransformationFrom(delta, self.GetTransform())
        )

    def SetPosition(self, value, fk=True):
        delta = Transform.TransformationTo(
            AI4Animation.AI4Animation.Scene.Transforms[self.Successors],
            self.GetTransform(),
        )
        AI4Animation.AI4Animation.Scene.Transforms[self.Index, :3, 3] = value
        AI4Animation.AI4Animation.Scene.Transforms[self.Successors] = (
            Transform.TransformationFrom(delta, self.GetTransform())
        )

    def SetRotation(self, value, fk=True):
        delta = Transform.TransformationTo(
            AI4Animation.AI4Animation.Scene.Transforms[self.Successors],
            self.GetTransform(),
        )
        AI4Animation.AI4Animation.Scene.Transforms[self.Index, :3, :3] = value
        AI4Animation.AI4Animation.Scene.Transforms[self.Successors] = (
            Transform.TransformationFrom(delta, self.GetTransform())
        )

    def SetPositionAndRotation(self, position, rotation, fk=True):
        delta = Transform.TransformationTo(
            AI4Animation.AI4Animation.Scene.Transforms[self.Successors],
            self.GetTransform(),
        )
        AI4Animation.AI4Animation.Scene.Transforms[self.Index, :3, 3] = position
        AI4Animation.AI4Animation.Scene.Transforms[self.Index, :3, :3] = rotation
        AI4Animation.AI4Animation.Scene.Transforms[self.Successors] = (
            Transform.TransformationFrom(delta, self.GetTransform())
        )

    def GetTransform(self):
        return AI4Animation.AI4Animation.Scene.Transforms[self.Index]

    def GetPosition(self):
        return AI4Animation.AI4Animation.Scene.Transforms[self.Index, :3, 3]

    def GetRotation(self):
        return AI4Animation.AI4Animation.Scene.Transforms[self.Index, :3, :3]

    def SetScale(self, value):
        delta = value / self.GetScale()
        AI4Animation.AI4Animation.Scene.Scales[self.Index] *= delta
        AI4Animation.AI4Animation.Scene.Scales[self.Successors] *= delta
        AI4Animation.AI4Animation.Scene.Transforms[self.Successors] = Transform.TR(
            self.GetPosition()
            + delta
            * (
                Transform.GetPosition(
                    AI4Animation.AI4Animation.Scene.Transforms[self.Successors]
                )
                - self.GetPosition()
            ),
            Transform.GetRotation(
                AI4Animation.AI4Animation.Scene.Transforms[self.Successors]
            ),
        )

    def GetScale(self):
        return AI4Animation.AI4Animation.Scene.Scales[self.Index]

    def SetParent(self, parent):
        if self.Parent is not None:
            self.Parent.Children.remove(self)
        if parent is not None:
            parent.Children.append(self)
        self.Parent = parent
        if self.Parent is not None:
            self.Parent.AddSuccessor(self)

    def AddSuccessor(self, entity):
        self.Successors.append(entity.Index)
        if self.Parent is not None:
            self.Parent.AddSuccessor(entity)

    def IsParentOf(self, entity):
        t = entity
        while t.Parent is not None:
            t = t.Parent
            if t is self:
                return True
        return False

    def FindParent(self, candidates):
        entity = self
        while entity.Parent is not None:
            entity = entity.Parent
            if entity.Name in candidates:
                return entity
        return None

    def FindChild(self, name):
        result = []

        def Recursion(entity, result):
            for child in entity.Children:
                if child.Name == name:
                    result.append(child)
                Recursion(child, result)

        Recursion(self, result)
        if len(result) == 0:
            print(
                "Entity with name " + self.Name + " has no child with name " + str(name)
            )
            return None
        if len(result) > 1:
            print(
                "Entity with name "
                + self.Name
                + " has multiple children with name "
                + str(name)
            )
            return None
        return result[0]

    def FindChilds(self, *names):
        return [self.FindChild(name) for name in names]

    def AddComponent(self, component, *params):
        if not component in self.Components.keys():
            c = component(self, params)
            self.Components[component] = c
            print("Added component with type", type(c).__name__, "to entity", self.Name)
            return c
        else:
            print(
                "Component with type "
                + str(component)
                + " already exists on entity "
                + self.Name
            )
            return None

    def GetComponent(self, component):
        try:
            return self.Components[component]
        except KeyError:
            print(
                "Component with type "
                + str(component)
                + " does not exist on entity "
                + self.Name
            )
            return None

    def GetHierarchy(root, entities):
        joints = []
        for entity in entities:
            chain = Entity.GetChain(root, entity)
            for joint in chain:
                if joint not in joints:
                    joints.append(joint)
        return joints

    def GetChain(source, target):
        chain = []
        entity = target
        chain.append(entity)
        while entity.Parent is not None:
            entity = entity.Parent
            chain.append(entity)
            if entity is source:
                break
        chain.reverse()
        return chain

    def GetIndices(entities):
        return [entity.Index for entity in entities]

    def PrintHierarchy(self, entity=None, indent=""):
        entity = self if entity is None else entity
        print(
            indent,
            entity.Index,
            entity.Name,
            "->",
            entity.Parent.Name if entity.Parent is not None else "None",
        )
        # print(indent, entity.Name, '->', self.Successors)
        for c in entity.Children:
            self.PrintHierarchy(c, indent + "  ")

    def DrawHierarchy(self):
        AI4Animation.AI4Animation.Draw.Sphere(self.GetPosition(), 0.025)
        for child in self.Children:
            AI4Animation.AI4Animation.Draw.Line(self.GetPosition(), child.GetPosition())
            child.DrawHierarchy()

    def DrawHandle(self):
        self.Handle.GUI()

    def DrawHandles(self):
        self.DrawHandle()
        for child in self.Children:
            child.DrawHandles()

    def DrawLabel(self):
        AI4Animation.AI4Animation.Draw.Text3D(
            self.Name, self.GetPosition(), 0.01, AI4Animation.AI4Animation.Color.BLACK
        )

    def DrawLabels(self):
        self.DrawLabel()
        for child in self.Children:
            child.DrawLabel()
