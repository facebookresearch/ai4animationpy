# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation import AI4Animation, Component, MeshRenderer, Tensor, Time, Vector3


class Program:
    def Start(self):
        self.Root1 = AI4Animation.Scene.AddEntity(
            "Root1", position=Vector3.Create(0, 1.0, 0)
        )

        self.E1 = AI4Animation.Scene.AddEntity(
            "Entity1", position=Vector3.Create(0, 2.0, 1.0), parent=self.Root1
        )
        self.E2 = AI4Animation.Scene.AddEntity(
            "Entity2", position=Vector3.Create(0, 3.0, 1.0), parent=self.E1
        )

        self.E3 = AI4Animation.Scene.AddEntity(
            "Entity3", position=Vector3.Create(2, 2.0, 1.0), parent=self.Root1
        )
        self.E4 = AI4Animation.Scene.AddEntity(
            "Entity4", position=Vector3.Create(2, 3.0, 1.0), parent=self.E3
        )

        self.Root2 = AI4Animation.Scene.AddEntity(
            "Root2", position=Vector3.Create(-1, 1.0, 1.0)
        )

        self.E5 = AI4Animation.Scene.AddEntity(
            "Entity5", position=Vector3.Create(-1, 2.0, 3.0), parent=self.Root2
        )
        self.E6 = AI4Animation.Scene.AddEntity(
            "Entity6", position=Vector3.Create(-1, 3.0, 3.0), parent=self.E5
        )

        self.Cube1 = AI4Animation.Standalone.Primitives.CreateCube(
            "Cube1", position=Vector3.Create(-1, 2, -2.5)
        )
        self.Cube1.AddComponent(self.CustomBehavior1, 0.5, 1)

        self.Sphere = AI4Animation.Standalone.Primitives.CreateSphere(
            "Sphere", position=Vector3.Create(1, 2, -2.5)
        )

        self.Cube2 = AI4Animation.Standalone.Primitives.CreateCube(
            "Cube2", position=Vector3.Create(3, 2, -2.5)
        )
        self.Cube2.AddComponent(self.CustomBehavior2, 0.25, 1)

        AI4Animation.Scene.PrintHierarchy()

    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.Root1)

    def Draw(self):
        AI4Animation.Scene.DrawHierarchy()

    def GUI(self):
        AI4Animation.Scene.DrawHandles()

    class CustomBehavior1(Component):
        def Start(self, params):
            self.Amplitude = params[0]
            self.Frequency = params[1]
            self.Position = self.Entity.GetPosition().copy()

        def Update(self):
            self.Entity.SetPosition(
                self.Position
                + Vector3.Create(
                    0.0,
                    self.Amplitude
                    * Tensor.Sin(
                        self.Frequency * 360.0 * Time.TotalTime, inDegrees=True
                    ),
                    0.0,
                )
            )

    class CustomBehavior2(Component):
        def Start(self, params):
            self.Amplitude = params[0]
            self.Frequency = params[1]
            self.Scale = self.Entity.GetScale().copy()

        def Update(self):
            self.Entity.SetScale(
                self.Scale
                + self.Amplitude
                * Tensor.Sin(self.Frequency * 360.0 * Time.TotalTime, inDegrees=True)
                * Vector3.One()
            )


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
