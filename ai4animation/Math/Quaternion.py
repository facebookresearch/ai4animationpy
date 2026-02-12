from ai4animation.Math import Tensor, Vector3


def Create(*values):
    if len(values) == 0:
        return Create(0, 0, 0, 1)
    if len(values) == 1:
        return Tensor.Create(values[0])
    if len(values) == 4:
        return Tensor.Transpose(Tensor.Create(values))


def Euler(*values):
    if len(values) == 0:
        print("Did not provide any values")
    if len(values) == 1:
        angles = Tensor.Create(values[0])
    if len(values) == 3:
        angles = Tensor.Create(values)
    x = RotationX(angles[..., 0])
    y = RotationY(angles[..., 1])
    z = RotationZ(angles[..., 2])
    return Multiply(y, Multiply(x, z))


def RotationX(angle):
    return AngleAxis(angle, Vector3.X)


def RotationY(angle):
    return AngleAxis(angle, Vector3.Y)


def RotationZ(angle):
    return AngleAxis(angle, Vector3.Z)


def AngleAxis(angle, axis):
    axis = Tensor.Normalize(axis)
    angle = Tensor.Deg2Rad(Tensor.Div(angle, 2))
    c = Tensor.Cos(angle)
    s = Tensor.Sin(angle)
    x = axis[..., 0] * s
    y = axis[..., 1] * s
    z = axis[..., 2] * s
    w = c
    return Tensor.Stack((x, y, z, w), -1)


def ToAngleAxis(q):
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]  # [x, y, z, w]
    angle = Tensor.Rad2Deg(2.0 * Tensor.ArcCos(qw))
    if angle == 0.0:
        return angle, Vector3.Create(0, 0, 0)  # This may be buggy
    else:
        x = qx / Tensor.Sqrt(1 - qw * qw)
        y = qy / Tensor.Sqrt(1 - qw * qw)
        z = qz / Tensor.Sqrt(1 - qw * qw)
        return angle, Vector3.Create(x, y, z)


def Multiply(a, b):
    if b.shape[-1] == 3:  # Quaternion-Vector
        shape = list(b.shape)
        shape[-1] = 4
        tmp = Tensor.Zeros(shape)
        tmp[..., :3] = b
        tmp = Multiply(a, Multiply(tmp, Conjugate(a)))
        tmp = tmp[..., :3]
        return tmp
    if b.shape[-1] == 4:  # Quaternion-Quaternion
        q1x, q1y, q1z, q1w = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        q2x, q2y, q2z, q2w = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        w = q1w * q2w - (q1x * q2x + q1y * q2y + q1z * q2z)
        x = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
        y = q1w * q2y + q1y * q2w + q1z * q2x - q1x * q2z
        z = q1w * q2z + q1z * q2w + q1x * q2y - q1y * q2x
        return Tensor.Stack((x, y, z, w), -1)


def Conjugate(tensor):
    values = tensor.copy()
    values[..., :3] *= -1
    return values


def Inverse(tensor):
    values = Conjugate(tensor)
    sqr = Tensor.Sum(values**2, -1)
    return values / sqr


def Normalize(tensor):
    return tensor / Tensor.Norm(tensor)


def ToMatrix(q):
    q0, q1, q2, q3 = q[..., 3], q[..., 0], q[..., 1], q[..., 2]  # [w, x, y, z]

    R = Tensor.Zeros(list(q.shape)[:-1] + [3, 3])

    # First row
    R[..., 0, 0] = 2 * (q0**2 + q1**2) - 1
    R[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)

    # Second row
    R[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[..., 1, 1] = 2 * (q0**2 + q2**2) - 1
    R[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)

    # Third row
    R[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[..., 2, 2] = 2 * (q0**2 + q3**2) - 1

    return R


def FromMatrix(R):
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    r11, r12, r13 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r21, r22, r23 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r31, r32, r33 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    q0 = 0.5 * Tensor.Sqrt(Tensor.Maximum(1 + r11 + r22 + r33, 0))  # w
    q1 = (
        0.5
        * Tensor.Sqrt(Tensor.Maximum(1 + r11 - r22 - r33, 0))
        * Tensor.Sign(r32 - r23)
    )  # x
    q2 = (
        0.5
        * Tensor.Sqrt(Tensor.Maximum(1 - r11 + r22 - r33, 0))
        * Tensor.Sign(r13 - r31)
    )  # y
    q3 = (
        0.5
        * Tensor.Sqrt(Tensor.Maximum(1 - r11 - r22 + r33, 0))
        * Tensor.Sign(r21 - r12)
    )  # z
    M = Tensor.Stack((q1, q2, q3, q0), -1)
    M = M.reshape(list(shape) + [4])
    return M


def FromTo(u, v):
    u = u / Vector3.Length(u)
    v = v / Vector3.Length(v)

    dot_product = Vector3.Dot(u, v)
    cross_product = Vector3.Cross(u, v)

    # Handle the case of parallel or anti-parallel vectors
    import numpy as np

    if dot_product == -1:  # 180-degree rotation (anti-parallel)
        print("UNHANDLED CASE")
        # Find an arbitrary orthogonal vector to u for the axis of rotation
        arbitrary_axis = Vector3.Create(1, 0, 0)
        if np.allclose(u, arbitrary_axis):
            arbitrary_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(u, arbitrary_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        quaternion = np.array([0, rotation_axis[0], rotation_axis[1], rotation_axis[2]])
    else:
        w = Tensor.Sqrt(Tensor.Div(Tensor.Add(dot_product, 1), 2))
        xyz = cross_product / (2 * w)
        quaternion = Create(xyz[0], xyz[1], xyz[2], w)

    return Normalize(quaternion)  # Ensure unit quaternion
