# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation.Math import Quaternion, Tensor, Transform


def Identity(shape=None):
    tensor = Tensor.Eye(3)
    if shape is not None:
        if isinstance(shape, int):
            tensor = Tensor.Repeat(tensor.reshape(-1, 3, 3), shape, 0)
        else:
            capacity = Tensor.ShapeCapacity(shape)
            if capacity is not None:
                tensor = Tensor.Repeat(
                    tensor.reshape(-1, 3, 3), Tensor.ShapeCapacity(shape), 0
                ).reshape(list(shape) + [3, 3])
    return tensor


def Euler(*values):  # Euler angles in degrees
    if len(values) == 1:
        angles = Tensor.Create(values[0])
    if len(values) == 3:
        angles = Tensor.Transpose(Tensor.Create(values))
    rx = RotationX(angles[..., 0])
    ry = RotationY(angles[..., 1])
    rz = RotationZ(angles[..., 2])
    return Tensor.MatMul(ry, Tensor.MatMul(rx, rz))


def RotationX(angle):
    c = Tensor.Cos(Tensor.Deg2Rad(angle))
    s = Tensor.Sin(Tensor.Deg2Rad(angle))
    r = Tensor.Repeat(
        Tensor.Eye(3).reshape(-1, 3, 3), Tensor.TensorCapacity(angle), 0
    ).reshape(list(angle.shape) + [3, 3])
    r[..., 1, 1] = c
    r[..., 2, 2] = c
    r[..., 1, 2] = -s
    r[..., 2, 1] = s
    return r


def RotationY(angle):
    c = Tensor.Cos(Tensor.Deg2Rad(angle))
    s = Tensor.Sin(Tensor.Deg2Rad(angle))
    r = Tensor.Repeat(
        Tensor.Eye(3).reshape(-1, 3, 3), Tensor.TensorCapacity(angle), 0
    ).reshape(list(angle.shape) + [3, 3])
    r[..., 0, 0] = c
    r[..., 2, 2] = c
    r[..., 2, 0] = -s
    r[..., 0, 2] = s
    return r


def RotationZ(angle):
    c = Tensor.Cos(Tensor.Deg2Rad(angle))
    s = Tensor.Sin(Tensor.Deg2Rad(angle))
    r = Tensor.Repeat(
        Tensor.Eye(3).reshape(-1, 3, 3), Tensor.TensorCapacity(angle), 0
    ).reshape(list(angle.shape) + [3, 3])
    r[..., 0, 0] = c
    r[..., 1, 1] = c
    r[..., 0, 1] = -s
    r[..., 1, 0] = s
    return r


def GetAxisX(tensor):
    return tensor[..., :3, 0]


def GetAxisY(tensor):
    return tensor[..., :3, 1]


def GetAxisZ(tensor):
    return tensor[..., :3, 2]


def Inverse(tensor):
    # T = Tensor.Transpose(tensor)
    # T = Normalize(T)
    # return T
    return Tensor.Inverse(tensor)


def Interpolate(a, b, weight):
    R = Tensor.Interpolate(a, b, weight)
    R = Normalize(R)
    return R


def Look(z, y):  # Not yet differentiable
    z = Tensor.Normalize(z)
    y = Tensor.Normalize(y)
    x = Tensor.Cross(y, z)
    return Tensor.Stack((x, y, z), axis=-1)


def LookPlanar(z):
    z = Tensor.Normalize(z)
    y = Tensor.ZerosLike(z)
    y[..., 1] = 1.0
    x = Tensor.Cross(y, z)
    return Tensor.Stack((x, y, z), axis=-1)


def RotationFrom(tensor, space):
    return Multiply(Transform.GetRotation(space), tensor)


def RotationTo(tensor, space):
    return Multiply(Inverse(Transform.GetRotation(space)), tensor)


def RotationFromTo(u, v):
    return Quaternion.ToMatrix(Quaternion.FromTo(u, v))


def Multiply(a, b):
    return Tensor.MatMul(a, b)


def MultiplyVector(a, b):
    return Tensor.Squeeze(Tensor.MatMul(a, Tensor.Unsqueeze(b, -1)), -1)


def Normalize(tensor):
    # Option 1
    return Look(GetAxisZ(tensor), GetAxisY(tensor))

    # Option 2
    # u, s, vh = np.linalg.svd(tensor)
    # del s
    # return np.matmul(u, vh)

    # Option 3
    # q = Quaternion.FromMatrix(tensor)
    # q = Quaternion.Normalize(q)
    # q = Quaternion.ToMatrix(q)
    # return q
