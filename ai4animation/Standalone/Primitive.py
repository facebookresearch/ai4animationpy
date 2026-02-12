import raylib as rl
from ai4animation import AI4Animation, MeshRenderer

RESOLUTION = 30


def CreateCube(
    name, width=1.0, height=1.0, length=1.0, position=None, rotation=None, parent=None
):
    entity = AI4Animation.Scene.AddEntity(name, position, rotation, parent)
    model = rl.LoadModelFromMesh(rl.GenMeshCube(width, height, length))
    entity.AddComponent(MeshRenderer, model)
    return entity


def CreateSphere(name, radius=0.5, position=None, rotation=None, parent=None):
    entity = AI4Animation.Scene.AddEntity(name, position, rotation, parent)
    model = rl.LoadModelFromMesh(rl.GenMeshSphere(radius, RESOLUTION, RESOLUTION))
    entity.AddComponent(MeshRenderer, model)
    return entity
