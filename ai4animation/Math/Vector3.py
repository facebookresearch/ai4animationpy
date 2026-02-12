from enum import Enum

from ai4animation.Math import Tensor, Transform


class Axis(Enum):
    XPositive = 0
    YPositive = 1
    ZPositive = 2


def Create(*values):
    if len(values) == 0:
        return Create(0, 0, 0)
    if len(values) == 1:
        return Tensor.Create(values[0])
    if len(values) == 3:
        return Tensor.Transpose(Tensor.Create(values))


def Zero(shape=None):
    return Tensor.Shapify(Tensor.Zeros(3), shape)


def One(shape=None):
    return Tensor.Shapify(Tensor.Ones(3), shape)


def UnitX(shape=None):
    return Tensor.Shapify(X, shape)


def UnitY(shape=None):
    return Tensor.Shapify(Y, shape)


def UnitZ(shape=None):
    return Tensor.Shapify(Z, shape)


def Length(tensor):
    return Tensor.Norm(tensor)


def Normalize(tensor):
    return Tensor.Normalize(tensor)


def Cross(a, b):
    return Tensor.Cross(a, b)


def Distance(a, b):
    return Length(b - a)


def ClampMagnitude(v, max):
    if Length(v) > max:
        v = max * Normalize(v)
    return v


def Dot(a, b):
    return Tensor.Dot(a, b)


def Lerp(a, b, weight):
    return Tensor.Interpolate(a, b, weight)


def LerpDt(a, b, dt, rate, eps=0.01):
    if rate == 0:
        return a
    value = Lerp(a, b, 1.0 - Tensor.Exp(-dt * rate))
    if Distance(value, b) < eps:
        return b
    return value


def Slerp(a, b, weight):
    angle = SignedAngle(a, b, Y)
    if angle == 180:
        b = Normalize(b + 0.1 * Tensor.RandomUniform(3))
    startNorm = Normalize(a)
    endNorm = Normalize(b)
    dot = Dot(startNorm, endNorm)
    dot = Tensor.Clamp(dot, -1.0, 1.0)
    theta = Tensor.ArcCos(dot) * weight
    relativeVec = Normalize(endNorm - (startNorm * dot))
    return (startNorm * Tensor.Cos(theta)) + (relativeVec * Tensor.Sin(theta))


def SlerpDt(a, b, dt, rate, eps=0.01):
    if rate == 0:
        return a
    value = Slerp(a, b, 1.0 - Tensor.Exp(-dt * rate))
    if Distance(value, b) < eps:
        return b
    return value


def SignedAngle(v1, v2, axis):
    a = Dot(Cross(v1, v2), axis)
    b = Dot(v1, v2)
    angle = Tensor.ArcTan2(a, b)
    return Tensor.Rad2Deg(angle)


def SetVector(tensor, value, index=None):
    if index is None:
        tensor[..., :3] = value
    else:
        tensor[index, :3] = value


def GetVector(tensor, index=None):
    if index is None:
        return tensor.copy()
    else:
        return tensor[index].copy()


def PositionFrom(tensor, space):
    return Transform.GetPosition(space) + DirectionFrom(tensor, space)


def PositionTo(tensor, space):
    return PositionFrom(tensor, Transform.Inverse(space))


def PositionFromTo(tensor, fromSpace, toSpace):
    return PositionFrom(PositionTo(tensor, fromSpace), toSpace)


def DirectionFrom(tensor, space):
    return Tensor.Squeeze(
        Tensor.MatMul(Transform.GetRotation(space), Tensor.Unsqueeze(tensor, -1)), -1
    )


def DirectionTo(tensor, space):
    return Tensor.Squeeze(
        Tensor.MatMul(
            Tensor.Inverse(Transform.GetRotation(space)), Tensor.Unsqueeze(tensor, -1)
        ),
        -1,
    )


def DirectionFromTo(tensor, fromSpace, toSpace):
    return DirectionFrom(DirectionTo(tensor, fromSpace), toSpace)


def FromRayLib(values):
    return Create(values.x, values.y, values.z)


def ToRayLib(tensor):
    c = Tensor.TensorCapacity(tensor)
    if c != 3:
        print("Can only create RayLib vector for tensor of 3 but given", c)
        return None
    return tensor.flatten().tolist()


X = Create(1, 0, 0)
Y = Create(0, 1, 0)
Z = Create(0, 0, 1)
