from ai4animation.Math import Rotation, Tensor, Vector3


def Identity(shape=None):
    return Tensor.Shapify(Tensor.Eye(4), shape)


def TR(translation, rotation):
    tShape = translation.shape[:-1]
    # rShape = rotation.shape[:-2]
    # if tShape != rShape:
    #     print("Batch shapes for translation and rotation do not match:", tShape, rShape)
    tensor = Identity(tShape)
    tensor[..., :3, 3] = translation
    tensor[..., :3, :3] = rotation
    return tensor


def TRS(translation, rotation, scale):
    tShape = translation.shape[:-1]
    # rShape = rotation.shape[:-2]
    # if tShape != rShape:
    #     print("Batch shapes for translation and rotation do not match:", tShape, rShape)
    tensor = Identity(tShape)
    tensor[..., :3, 3] = translation
    tensor[..., :3, :3] = rotation

    tensor[..., :3, 0] *= scale[..., 0:1]
    tensor[..., :3, 1] *= scale[..., 1:2]
    tensor[..., :3, 2] *= scale[..., 2:3]

    return tensor


def T(translation):
    tensor = Identity(translation.shape[:-1])
    tensor[..., :3, 3] = translation
    return tensor


def R(rotation):
    tensor = Identity(rotation.shape[:-2])
    tensor[..., :3, :3] = rotation
    return tensor


def S(scale):
    tensor = Identity(scale.shape[:-1])
    tensor[..., 0, 0] = scale[..., 0]
    tensor[..., 1, 1] = scale[..., 1]
    tensor[..., 2, 2] = scale[..., 2]
    return tensor


def TXYZ(t, x, y, z):
    values = Identity(t.shape[:-1])
    values[..., :3, 3] = t
    values[..., :3, 0] = x
    values[..., :3, 1] = y
    values[..., :3, 2] = z
    return values


def DeltaXZ(delta):
    pos = Tensor.Copy(delta)
    pos[..., 1] = 0
    return TR(pos, Rotation.RotationY(delta[..., 1]))


def SetTransform(tensor, value, index=None):
    if index is None:
        tensor[..., :4, :4] = value
    else:
        tensor[index, :4, :4] = value


def GetTransform(tensor, index=None):
    if index is None:
        return tensor
    else:
        return tensor[index]


def SetPosition(tensor, value, index=None):
    if index is None:
        tensor[..., :3, 3] = value
    else:
        tensor[index, :3, 3] = value


def GetPosition(tensor, index=None):
    if index is None:
        return tensor[..., :3, 3]
    else:
        return tensor[index, :3, 3]


def SetRotation(tensor, value, index=None):
    if index is None:
        tensor[..., :3, :3] = value
    else:
        tensor[index, :3, :3] = value


def GetRotation(tensor, index=None):
    if index is None:
        return tensor[..., :3, :3]
    else:
        return tensor[index, :3, :3]


def GetAxisX(tensor, index=None):
    if index is None:
        return tensor[..., :3, 0]
    else:
        return tensor[index, :3, 0]


def GetAxisY(tensor, index=None):
    if index is None:
        return tensor[..., :3, 1]
    else:
        return tensor[index, :3, 1]


def GetAxisZ(tensor, index=None):
    if index is None:
        return tensor[..., :3, 2]
    else:
        return tensor[index, :3, 2]


def Inverse(tensor):
    # pos = GetPosition(tensor)
    # rot = GetRotation(tensor)
    # rot = Rotation.Inverse(rot)
    # pos = -Rotation.MultiplyVector(rot, pos)
    # return TR(pos, rot)

    # pos = GetPosition(tensor)
    # rot = GetRotation(tensor)
    # quat = Quaternion.FromMatrix(rot)
    # quat = Quaternion.Inverse(quat)
    # rot = Quaternion.ToMatrix(quat)
    # pos = -Rotation.MultiplyVector(rot, pos)
    # return TR(pos, rot)

    return Tensor.Inverse(tensor)


def Multiply(a, b):
    return Tensor.MatMul(a, b)


def Interpolate(a, b, weight):
    m = Tensor.Interpolate(a, b, weight)
    SetRotation(m, Rotation.Normalize(GetRotation(m)))
    return m


def GetMirror(tensor, axis):
    tensor = tensor.copy()
    if axis == Vector3.Axis.XPositive:
        tensor[..., 0, 3] *= -1
        tensor[..., 0, 1] *= -1
        tensor[..., 0, 2] *= -1
        tensor[..., 1, 0] *= -1
        tensor[..., 2, 0] *= -1
    elif axis == Vector3.Axis.YPositive:
        tensor[..., 1, 3] *= -1
        tensor[..., 1, 0] *= -1
        tensor[..., 1, 2] *= -1
        tensor[..., 0, 1] *= -1
        tensor[..., 2, 1] *= -1
    elif axis == Vector3.Axis.ZPositive:
        tensor[..., 2, 3] *= -1
        tensor[..., 2, 0] *= -1
        tensor[..., 2, 1] *= -1
        tensor[..., 0, 2] *= -1
        tensor[..., 1, 2] *= -1
    return tensor


def TransformationFrom(tensor, space):
    return Multiply(space, tensor)


def TransformationTo(tensor, space):
    return Multiply(Inverse(space), tensor)


def TransformationFromTo(tensor, fromSpace, toSpace):
    return TransformationFrom(TransformationTo(tensor, fromSpace), toSpace)
