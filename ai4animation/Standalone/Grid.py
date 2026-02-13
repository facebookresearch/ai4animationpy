# Copyright (c) Meta Platforms, Inc. and affiliates.
import pyray as pr
import raylib as rl
from ai4animation.Components.Component import Component
from ai4animation.Math import Quaternion, Vector3


class Grid(Component):
    def Start(self, params):
        size = params[0]
        resolution = params[1]
        light = params[2]

        model = rl.LoadModelFromMesh(
            rl.GenMeshPlane(float(size), float(size), resolution, resolution)
        )

        position = self.Entity.GetPosition()
        rotation = self.Entity.GetRotation()
        quaternion = Quaternion.FromMatrix(rotation)
        angle, axis = Quaternion.ToAngleAxis(quaternion)

        light.RegisterModel(
            name=self.Entity.Name,
            model=model,
            skinned_mesh=None,
            position=Vector3.ToRayLib(position),
            rotationAxis=Vector3.ToRayLib(axis),
            rotationAngle=angle,
            scale=None,
            color=pr.Color(190, 190, 190, 255),
        )
