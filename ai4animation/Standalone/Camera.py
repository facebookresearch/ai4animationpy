import pyray as pr
import raylib as rl
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Components.Component import Component
from ai4animation.Math import Rotation, Transform, Vector3


class Camera(Component):
    def Start(self, params):
        self.Camera = pr.Camera3D(
            [5.0, 5.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 45.0, 0
        )
        self.Camera.projection = rl.CAMERA_PERSPECTIVE

        self.Mode = 1  # 0=Free, 1=Fixed, 2=Third, 3=Orbit
        self.Distance = 5.0
        self.Target = None

    def SetTarget(self, value):
        self.Target = value

    def Update(self):
        # Better don't touch this code for now...
        oldPosition = Vector3.FromRayLib(self.Camera.position)
        oldTarget = Vector3.FromRayLib(self.Camera.target)
        position = oldPosition
        target = oldTarget
        blend = 1

        if self.Mode == 0:
            sensitivityP = 10
            sensitivityX = 60
            sensitivityY = 60
            transform = self.Entity.GetTransform()
            if rl.IsMouseButtonDown(rl.MOUSE_BUTTON_RIGHT):
                deltaPosition = (
                    sensitivityP
                    * rl.GetFrameTime()
                    * Vector3.Create(AI4Animation.Standalone.IO.GetWASDQE())
                )
                deltaPosition[1] = 0
                deltaPosition[2] *= -1
                mouseDelta = rl.GetMouseDelta()
                deltaRotation = [
                    -sensitivityY * mouseDelta.y * rl.GetFrameTime(),
                    -sensitivityX * mouseDelta.x * rl.GetFrameTime(),
                    0,
                ]
                transform = Transform.Multiply(
                    transform,
                    Transform.TR(deltaPosition, Rotation.Euler(deltaRotation)),
                )
            position = Transform.GetPosition(transform)
            target = position - Transform.GetAxisZ(transform)

        if self.Mode == 1:
            if self.Target is not None:
                blend = 10 * rl.GetFrameTime()
                position = self.Target.GetPosition() + Vector3.Create(
                    0.0, 2.0, self.Distance
                )
                target = self.Target.GetPosition() + Vector3.Create(0.0, 1.0, 0.0)

        if self.Mode == 2:
            if self.Target is not None:
                blend = 30 * rl.GetFrameTime()
                position = Vector3.PositionFrom(
                    Vector3.Create(0.0, 2.0, -self.Distance), self.Target.GetTransform()
                )
                target = Vector3.PositionFrom(
                    Vector3.Create(0.0, 1.0, self.Distance), self.Target.GetTransform()
                )

        if self.Mode == 3:
            if self.Target is not None:
                blend = 30 * rl.GetFrameTime()
                selfHeight = 2
                targetHeight = 1
                speed = 60
                position = self.Target.GetPosition() + Rotation.Multiply(
                    Rotation.Euler(0, speed * rl.GetTime(), 0),
                    Vector3.Create(0, selfHeight, self.Distance),
                )
                target = self.Target.GetPosition() + Vector3.Create(0, targetHeight, 0)

        position = (1 - blend) * oldPosition + blend * position
        target = (1 - blend) * oldTarget + blend * target

        self.Camera.position = Vector3.ToRayLib(position)
        self.Camera.target = Vector3.ToRayLib(target)
        self.Camera.up = (0, 1, 0)

        z = Vector3.Normalize(target - position)
        x = Rotation.Multiply(Rotation.Euler(0, 90, 0), z)
        x[1] = 0.0
        y = Vector3.Cross(x, z)
        transform = Transform.TXYZ(position, -x, -y, -z)
        self.Entity.SetTransform(transform)

    def CreateButtons(self):
        buttons = []
        modes = ["Free Camera", "Fixed View", "Third Person", "Orbit View"]
        for i in range(len(modes)):
            buttons.append(
                AI4Animation.GUI.Button(
                    modes[i],
                    0.05,
                    (i + 1) * 0.15,
                    0.9,
                    0.125,
                    False,
                    False,
                    self.Canvas,
                )
            )
        buttons[self.Mode].Active = True
        return buttons

    def Standalone(self):
        self.Canvas = AI4Animation.GUI.Canvas(self.Entity.Name, 0.01, 0.01, 0.125, 0.25)
        self.Buttons = []
        modes = ["Free Camera", "Fixed View", "Third Person", "Orbit View"]
        for i in range(len(modes)):
            self.Buttons.append(
                AI4Animation.GUI.Button(
                    modes[i],
                    0.05,
                    (i + 1) * 0.15,
                    0.9,
                    0.125,
                    False,
                    False,
                    self.Canvas,
                )
            )
        self.Buttons[self.Mode].Active = True

    def GUI(self):
        self.Canvas.GUI()
        for i, button in enumerate(self.Buttons):
            if button.IsPressed():
                self.Mode = i
        for i, button in enumerate(self.Buttons):
            button.GUI()
            button.Active = self.Mode == i
