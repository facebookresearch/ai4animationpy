# v1 = Vector3.Create(1,2,3)
# v2 = Vector3.Zero()
# v3 = Vector3.Zero(5)
# print(v1, v2, v3)


# q1 = Quaternion.Create(1,0,0,1)
# q2 = Quaternion.Euler(10,20,30)
# q3 = Quaternion.Euler(-10,-20,-30)
# q4 = Quaternion.Euler(Tensor.Uniform(8,5,3))
# print(q1, q2, q3, q4.shape)


# r1 = Rotation.Euler(10,20,30)
# r2 = Rotation.Euler(Tensor.Uniform(8,5,3))
# print(r1.shape, r2.shape)


# U = Tensor.Uniform(8,5,3)
# r1 = Rotation.Euler(U)
# r2 = Quaternion.ToMatrix(Quaternion.Euler(U))
# print(r1[0])
# print(r2[0])


# U = Tensor.Uniform(8,5,3)
# R = Rotation.Euler(U)
# q = Quaternion.FromMatrix(R)
# r = Quaternion.ToMatrix(q)
# print(R[0])
# print(r[0])


# U = Tensor.Uniform(8,5,3)
# Q = Quaternion.Euler(U)
# R = Quaternion.ToMatrix(Q)
# q = Quaternion.FromMatrix(R)
# print(Q[0])
# print(q[0])


# m1 = Transform.Identity()
# m2 = Transform.Identity(2)
# p1 = Transform.GetPosition(m1)
# p2 = Transform.GetPosition(m2)
# r1 = Transform.GetRotation(m1)
# r2 = Transform.GetRotation(m2)
# print(m1.shape, m2.shape, p1, p2, r1, r2)


# A = Transform.TR(Vector3.Create(1,2,3), Rotation.Euler(10,20,30))
# B = Transform.TR(Vector3.Create(2,4,6), Rotation.Euler(-10,-20,-30))
# T = Transform.Identity((5,7))
# print(A.shape, B.shape, T.shape)


# Uv = Tensor.Uniform(9,5,3)
# Uq = Tensor.Uniform(9,5,3,3)
# C = Transform.TR(Uv, Uq)
# print(C.shape)


# A = Transform.TR(Vector3.Create(1,2,3), Rotation.Euler(10,20,30))
# B = Transform.TR(Vector3.Create(2,4,6), Rotation.Euler(-10,-20,-30))
# M1 = Transform.TransformationFrom(B, A)
# M2 = Transform.TransformationTo(M1, A)
# print(A)
# print(B)
# print(M1)
# print(M2)


# T = Transform.Identity(5)
# v = Vector3.One()
# Transform.SetPosition(T, v, [1,2,3])
# print(T)


# U = Tensor.Uniform(8,5,3)
# R = Rotation.Euler(U)

# start = time.time()
# for i in range(10000):
#     R1 = Rotation.Normalize(R)
# end = time.time()
# print(end - start)

# start = time.time()
# for i in range(10000):
#     R2 = Rotation.NormalizeFast(R)
# end = time.time()
# print(end - start)


# M = Transform.Identity(8)
# p = Vector3.Zero(8)
# print(M.shape, p.shape)
# v = Vector3.PositionFrom(p, M)
# print(v.shape)


# T = Transform.Identity()
# start = time.time()
# for i in range(10000):
#     I = Transform.Inverse(T)
# end = time.time()
# print(end - start)
# start = time.time()
# for i in range(10000):
#     I = Tensor.Inverse(T)
# end = time.time()
# print(end - start)
