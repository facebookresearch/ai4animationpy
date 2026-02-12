import raylib as rl
from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Transform as t
from raylib.colors import BLACK, BLUE, GREEN, RED, WHITE


def ScreenWidth():
    return rl.GetScreenWidth()


def ScreenHeight():
    return rl.GetScreenHeight()


def Cube(position, size=0.1, color=BLACK):
    if position.shape[0] == 0:
        return
    position_list = position.tolist()
    for pos in position_list:
        rl.DrawCube(pos, size, size, size, color)


# Default is 16 rings (first parameter) and 16 slices (second parameter)
def Sphere(position, size=0.1, resolution=6, color=BLACK):
    if position.shape[0] == 0:
        return
    position_list = position.reshape(-1, 3).tolist()
    for pos in position_list:
        rl.DrawSphereEx(pos, size, resolution, resolution, color)


def Line(start, end, color=BLACK):
    start, end = start.reshape(-1, 3), end.reshape(-1, 3)
    for start_pos, end_pos in zip(start.tolist(), end.tolist()):
        rl.DrawLine3D(start_pos, end_pos, color)


def LineStrip(positions, color=BLACK):
    if positions.shape[0] < 2:
        return
    positions_list = positions.tolist()
    for i in range(1, positions.shape[0]):
        rl.DrawLine3D(positions_list[i - 1], positions_list[i], color)


def Plane(position, size, color=BLACK):
    if position.shape[0] == 0:
        return
    position_list = position.tolist()
    size_list = size.tolist()
    for pos in position_list:
        rl.DrawPlane(pos, size_list, color)


def Cylinder(start, end, startSize, endSize, resolution=10, color=BLACK):
    if start.shape[0] == 0:
        return
    start = start.reshape(-1, 3)
    end = end.reshape(-1, 3)
    start_list = start.tolist()
    end_list = end.tolist()
    for start_pos, end_pos in zip(start_list, end_list):
        rl.DrawCylinderEx(start_pos, end_pos, startSize, endSize, resolution, color)


def Model(model, position, scale, color=WHITE):
    rl.DrawModel(model, position, scale, color)


def Transform(matrix, size=0.1, axisSize=0.25):
    p = t.GetPosition(matrix)
    x = t.GetAxisX(matrix)
    y = t.GetAxisY(matrix)
    z = t.GetAxisZ(matrix)
    Line(p, p + size * axisSize * x, RED)
    Line(p, p + size * axisSize * y, GREEN)
    Line(p, p + size * axisSize * z, BLUE)
    Sphere(p, size=0.05 * size, color=BLACK)
    Sphere(p + size * axisSize * x, size=0.05 * size * axisSize, color=RED)
    Sphere(p + size * axisSize * y, size=0.05 * size * axisSize, color=GREEN)
    Sphere(p + size * axisSize * z, size=0.05 * size * axisSize, color=BLUE)


def Vector(origin, direction, size=1.0, color=BLACK):
    Cylinder(origin, origin + direction, size, 0, color=color)


def Text(
    text, x, y, size=1.0, color=WHITE, pivot=0, canvas=None
):  # x and y are between 0 and 1
    text = Utility.ToBytes(text)

    if canvas is not None:
        x = (
            canvas.Rectangle.x + (canvas.Rectangle.width * x)
            if canvas.ScaleWidth
            else x
        )
        y = (
            canvas.Rectangle.y + (canvas.Rectangle.height * y)
            if canvas.ScaleHeight
            else y
        )

    size = int(ScreenHeight() * size)

    w = rl.MeasureText(text, size)
    h = size

    x = ScreenWidth() * x - w * pivot
    y = ScreenHeight() * y

    rl.DrawText(text, int(x), int(y), size, color)


def Text3D(text, position, size=1.0, color=WHITE):
    camera = AI4Animation.Standalone.Camera.Camera
    if not isinstance(text, list):
        text = [text]
    position_list = position.reshape(-1, 3).tolist()
    for i, pos in enumerate(position_list):
        point = rl.GetWorldToScreen(pos, camera)
        x = point.x / ScreenWidth()
        y = point.y / ScreenHeight()
        Text(text[i], x, y, size, color)


def Skeleton(root, positions, actor, bones=None, size: float = 1.0, color=None):
    parent_positions = positions[actor.GetParentIndices(bones)]
    bone_positions = positions[actor.GetBoneIndices(bones)]
    color = AI4Animation.Color.BLACK if color is None else color
    if root is not None:
        Transform(root, 0.5)
    Cylinder(
        parent_positions,
        bone_positions,
        0.02 * size,
        0.0,
        10,
        Utility.Opacity(color, 0.25),
    )
