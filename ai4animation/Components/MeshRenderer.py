# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Components.Component import Component
from ai4animation.Math import Quaternion, Vector3


class MeshRenderer(Component):
    def Start(self, params):
        self.Model = AI4Animation.Standalone.RenderPipeline.RegisterModel(
            self.Entity.Name, params[0], None
        )

    def Update(self):
        self.Model.position = Vector3.ToRayLib(self.Entity.GetPosition())
        angle, axis = Quaternion.ToAngleAxis(
            Quaternion.FromMatrix(self.Entity.GetRotation())
        )
        self.Model.rotationAxis = Vector3.ToRayLib(axis)
        self.Model.rotationAngle = angle
        self.Model.scale = Vector3.ToRayLib(self.Entity.GetScale())
