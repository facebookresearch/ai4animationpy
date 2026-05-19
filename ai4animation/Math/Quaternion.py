# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Quaternion creation, multiplication, and conversion utilities."""

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
    q = Normalize(q)

    # Storage layout is [x, y, z, w]; unpack as (w, x, y, z).
    q0, q1, q2, q3 = q[..., 3], q[..., 0], q[..., 1], q[..., 2]

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
    """Convert rotation matrix to quaternion (x, y, z, w) using Shepperd's method.

    Picks the largest of the four candidate magnitudes
    (1 + r11 + r22 + r33), (1 + r11 - r22 - r33),
    (1 - r11 + r22 - r33), (1 - r11 - r22 + r33)
    as the pivot, computes that component from sqrt, and derives the remaining
    three from off-diagonal sums/differences. This avoids the sign(0) failure
    of the naive formulation (e.g. for 180-degree rotations) and is numerically
    stable across the full range of valid rotation matrices.
    """
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    r11, r12, r13 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r21, r22, r23 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r31, r32, r33 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # Four candidate values (each equals 4 * component^2 for w, x, y, z resp).
    # Their sum is always 4, so the maximum is always >= 1 for a valid rotation.
    t0 = 1.0 + r11 + r22 + r33
    t1 = 1.0 + r11 - r22 - r33
    t2 = 1.0 - r11 + r22 - r33
    t3 = 1.0 - r11 - r22 + r33

    # Pivot scale s_i = 4 * pivot_component for each case. Clamp to EPS so the
    # divisions in non-selected branches don't produce NaN/Inf.
    s0 = Tensor.Maximum(2.0 * Tensor.Sqrt(Tensor.Maximum(t0, 0.0)), Tensor.EPS)
    s1 = Tensor.Maximum(2.0 * Tensor.Sqrt(Tensor.Maximum(t1, 0.0)), Tensor.EPS)
    s2 = Tensor.Maximum(2.0 * Tensor.Sqrt(Tensor.Maximum(t2, 0.0)), Tensor.EPS)
    s3 = Tensor.Maximum(2.0 * Tensor.Sqrt(Tensor.Maximum(t3, 0.0)), Tensor.EPS)

    # Case 0: w is the pivot.
    w0 = 0.25 * s0
    x0 = (r32 - r23) / s0
    y0 = (r13 - r31) / s0
    z0 = (r21 - r12) / s0

    # Case 1: x is the pivot.
    w1 = (r32 - r23) / s1
    x1 = 0.25 * s1
    y1 = (r12 + r21) / s1
    z1 = (r13 + r31) / s1

    # Case 2: y is the pivot.
    w2 = (r13 - r31) / s2
    x2 = (r12 + r21) / s2
    y2 = 0.25 * s2
    z2 = (r23 + r32) / s2

    # Case 3: z is the pivot.
    w3 = (r21 - r12) / s3
    x3 = (r13 + r31) / s3
    y3 = (r23 + r32) / s3
    z3 = 0.25 * s3

    # Select branch with maximum t (most numerically stable pivot).
    cond01 = t0 >= t1
    w01 = Tensor.Where(cond01, w0, w1)
    x01 = Tensor.Where(cond01, x0, x1)
    y01 = Tensor.Where(cond01, y0, y1)
    z01 = Tensor.Where(cond01, z0, z1)
    t01 = Tensor.Where(cond01, t0, t1)

    cond23 = t2 >= t3
    w23 = Tensor.Where(cond23, w2, w3)
    x23 = Tensor.Where(cond23, x2, x3)
    y23 = Tensor.Where(cond23, y2, y3)
    z23 = Tensor.Where(cond23, z2, z3)
    t23 = Tensor.Where(cond23, t2, t3)

    cond = t01 >= t23
    w = Tensor.Where(cond, w01, w23)
    x = Tensor.Where(cond, x01, x23)
    y = Tensor.Where(cond, y01, y23)
    z = Tensor.Where(cond, z01, z23)

    # Storage layout is [x, y, z, w].
    M = Tensor.Stack((x, y, z, w), -1)
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
