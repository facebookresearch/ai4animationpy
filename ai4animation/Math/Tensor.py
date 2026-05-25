# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tensor utilities for creating, reshaping, and operating on PyTorch and NumPy tensors."""

from enum import Enum

import numpy as np
import torch

EPS = 1e-5


class Backend(Enum):
    NumPy = 0
    PyTorch = 1


class Device(Enum):
    CPU = -1
    GPU1 = 0
    GPU2 = 1
    GPU3 = 2
    GPU4 = 3


TensorDevice = Device.GPU1 if torch.cuda.is_available() else Device.CPU
TensorBackend = Backend.NumPy


def SetDevice(device):
    global TensorDevice
    TensorDevice = device


def SetBackend(backend):
    global TensorBackend
    TensorBackend = backend


def GetBackend(tensor):
    if isinstance(tensor, np.ndarray):
        return Backend.NumPy
    if isinstance(tensor, torch.Tensor):
        return Backend.PyTorch
    if isinstance(tensor, tuple):
        if all(isinstance(item, np.ndarray) for item in tensor):
            return Backend.NumPy
        if all(isinstance(item, torch.Tensor) for item in tensor):
            return Backend.PyTorch
    # else:
    # raise TypeError(f"Unsupported tensor type: {type(tensor)}")
    return Backend.NumPy


def ToDevice(tensor):
    if TensorDevice == Device.CPU or isinstance(tensor, np.ndarray):
        return tensor
    assert TensorDevice.value < torch.cuda.device_count(), (
        f"{TensorDevice} not available"
    )
    return tensor.to(torch.device("cuda", TensorDevice.value))


def ToNumPy(tensor):
    return tensor.detach().cpu().numpy()


def ToPyTorch(tensor):
    return ToDevice(torch.from_numpy(tensor))


def ToInt(values):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return values.astype(np.int32)
    if backend == Backend.PyTorch:
        return values.to(torch.int32)


def TensorCapacity(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return tensor.size
    if backend == Backend.PyTorch:
        return tensor.numel()


def ShapeCapacity(shape):
    return None if shape == () else np.prod(shape)


def Shapify(tensor, shape):
    if shape is not None:
        dims = list(tensor.shape)
        if isinstance(shape, int):
            tensor = Repeat(Unsqueeze(tensor, 0), shape, 0)
        else:
            capacity = ShapeCapacity(shape)
            if capacity is not None:
                tensor = Repeat(Unsqueeze(tensor, 0), ShapeCapacity(shape), 0).reshape(
                    list(shape) + dims
                )
    return tensor


def Create(*values):
    if len(values) == 1:
        values = values[0]
    if isinstance(values, float) or isinstance(values, float):
        values = [values]
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.array(values, dtype=np.float32)
    if backend == Backend.PyTorch:
        return ToDevice(torch.tensor(values, dtype=torch.float32))


def Eye(dim, backend=None):
    backend = TensorBackend if backend is None else backend
    if backend == Backend.NumPy:
        return np.eye(dim, dtype=np.float32)
    if backend == Backend.PyTorch:
        return ToDevice(torch.eye(dim, dtype=torch.float32))


def Empty(*shape, backend=None):
    backend = TensorBackend if backend is None else backend
    if len(shape) == 1:
        shape = shape[0]
    if backend == Backend.NumPy:
        return np.empty(shape, dtype=np.float32)
    if backend == Backend.PyTorch:
        return ToDevice(torch.empty(shape, dtype=torch.float32))


def Ones(*shape, backend=None):
    backend = TensorBackend if backend is None else backend
    if len(shape) == 1:
        shape = shape[0]
    if backend == Backend.NumPy:
        return np.ones(shape, dtype=np.float32)
    if backend == Backend.PyTorch:
        return ToDevice(torch.ones(shape, dtype=torch.float32))


def OnesLike(tensor, backend=None):
    backend = TensorBackend if backend is None else backend
    if backend == Backend.NumPy:
        return np.ones_like(tensor)
    if backend == Backend.PyTorch:
        return torch.ones_like(tensor)


def Zeros(*shape, backend=None):
    backend = TensorBackend if backend is None else backend
    if len(shape) == 1:
        shape = shape[0]
    if backend == Backend.NumPy:
        return np.zeros(shape, dtype=np.float32)
    if backend == Backend.PyTorch:
        return ToDevice(torch.zeros(shape, dtype=torch.float32))


def ZerosLike(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.zeros_like(tensor)
    if backend == Backend.PyTorch:
        return torch.zeros_like(tensor)


def LinSpace(a, b, c, axis=-1):
    backend = GetBackend(a)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.linspace(a, b, c, axis=axis)
    if backend == Backend.PyTorch:
        return ToPyTorch(np.linspace(a, b, c, axis=axis))
        # return ToDevice(torch.linspace(a, b, c, axis=axis))


def Arange(start, end, step, backend=None):
    backend = TensorBackend if backend is None else backend
    if type(step) is int:
        if backend == Backend.NumPy:
            return np.arange(start, end, step, dtype=np.int32)
        if backend == Backend.PyTorch:
            return ToDevice(torch.arange(start, end, step, dtype=torch.int32))
    if type(step) is float:
        if backend == Backend.NumPy:
            return np.arange(start, end, step, dtype=np.float32)
        if backend == Backend.PyTorch:
            return ToDevice(torch.arange(start, end, step, dtype=torch.float32))


def RandomUniform(shape=None, min=0.0, max=1.0, backend=None):
    backend = TensorBackend if backend is None else backend
    if backend == Backend.NumPy:
        if shape is None:
            return np.random.rand() * (max - min) + min
        else:
            return np.random.uniform(0.0, 1.0, shape) * (max - min) + min
    if backend == Backend.PyTorch:
        print("PyTorch backend for RandomUniform not yet implemented.")
        return None


def RandomBool(shape=None, backend=None):
    backend = TensorBackend if backend is None else backend
    return RandomUniform(shape, 0.0, 1.0, backend=TensorBackend) > 0.5


def Transpose(tensor, axis1=-1, axis2=-2):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        if len(tensor.shape) > 1:
            return np.swapaxes(tensor, axis1, axis2)
        else:
            return np.transpose(tensor)
    if backend == Backend.PyTorch:
        if len(tensor.shape) > 1:
            return torch.transpose(tensor, axis1, axis2)
        else:
            return torch.transpose(tensor)


def Normalize(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        n = Norm(tensor)
        idx = np.where(n == 0.0)
        n[idx] = 1.0
        tensor = tensor / n
        return tensor
    if backend == Backend.PyTorch:
        n = Norm(tensor)
        # idx = torch.where(n == 0.0)
        # n[idx] = 1.0
        tensor = tensor / n
        return tensor


def Norm(tensor, axis=-1, keepDim=True):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        if isinstance(tensor, float):
            return np.abs(tensor)
        else:
            return np.linalg.norm(tensor, axis=axis, keepdims=keepDim)
    if backend == Backend.PyTorch:
        if isinstance(tensor, float):
            return torch.abs(tensor)
        else:
            return torch.norm(tensor, dim=axis, keepdim=keepDim)


def Distance(a, b):
    return Norm(b - a)


def Cross(a, b):
    backend = GetBackend(a)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.cross(a, b, axis=-1)
    if backend == Backend.PyTorch:
        return torch.cross(a, b, dim=-1)


def Dot(a, b):
    backend = GetBackend(a)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.sum(a * b, axis=-1)
    if backend == Backend.PyTorch:
        return torch.sum(a * b, dim=-1)


def Sqrt(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.sqrt(tensor)
    if backend == Backend.PyTorch:
        return torch.sqrt(tensor)


def Deg2Rad(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.deg2rad(tensor, dtype=np.float32)
    if backend == Backend.PyTorch:
        return torch.deg2rad(tensor)


def Rad2Deg(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.rad2deg(tensor, dtype=np.float32)
    if backend == Backend.PyTorch:
        return torch.rad2deg(tensor)


def Sin(tensor, inDegrees=False):
    if inDegrees:
        tensor = Deg2Rad(tensor)
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.sin(tensor)
    if backend == Backend.PyTorch:
        return torch.sin(tensor)


def Cos(tensor, inDegrees=False):
    if inDegrees:
        tensor = Deg2Rad(tensor)
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.cos(tensor)
    if backend == Backend.PyTorch:
        return torch.cos(tensor)


def ArcCos(tensor, inDegrees=False):
    if inDegrees:
        tensor = Deg2Rad(tensor)
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.arccos(tensor)
    if backend == Backend.PyTorch:
        return torch.arccos(tensor)


def ArcTan2(v1, v2):
    backend = GetBackend(v1)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.arctan2(v1, v2)
    if backend == Backend.PyTorch:
        return torch.arctan2(v1, v2)


def Add(a, b):
    backend = GetBackend(a)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.add(a, b, dtype=a.dtype)
    if backend == Backend.PyTorch:
        return a + b


def Div(a, b):
    backend = GetBackend(a)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.divide(a, b, dtype=a.dtype)
    if backend == Backend.PyTorch:
        return a / b


def Abs(values):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.abs(values)
    if backend == Backend.PyTorch:
        return torch.abs(values)


def Sum(values, axis=-1, keepDim=True):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.sum(values, axis=axis, keepdims=keepDim)
    if backend == Backend.PyTorch:
        return torch.sum(values, dim=axis, keepDim=keepDim)


def CumulativeSum(values, axis, backend=None):
    backend: Backend = GetBackend(values) if backend is None else backend
    if backend == Backend.NumPy:
        return np.cumsum(values, axis=axis)
    if backend == Backend.PyTorch:
        return torch.cumsum(values, dim=axis)


def Pow(values, power):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.power(values, power)
    if backend == Backend.PyTorch:
        return torch.pow(values, power)


def Stack(values, axis, backend=None):
    backend: Backend = GetBackend(values) if backend is None else backend
    if backend == Backend.NumPy:
        return np.stack(values, axis=axis)
    if backend == Backend.PyTorch:
        return torch.stack(values, dim=axis)


def Concat(values, axis, backend=None):
    backend: Backend = GetBackend(values) if backend is None else backend
    if backend == Backend.NumPy:
        return np.concatenate(values, axis=axis)
    if backend == Backend.PyTorch:
        return torch.cat(values, dim=axis)


def Repeat(values, num, axis):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return values.repeat(num, axis=axis)
    if backend == Backend.PyTorch:
        return values.repeat_interleave(num, dim=axis)


def Min(tensor, axis=-1, keepDim=True):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.min(tensor, axis=axis, keepdims=keepDim)
    if backend == Backend.PyTorch:
        return torch.min(tensor, axis, keepDim)


def Max(tensor, axis=-1, keepDim=True):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.max(tensor, axis=axis, keepdims=keepDim)
    if backend == Backend.PyTorch:
        return torch.max(tensor, axis, keepDim)


def Minimum(a, b):
    backend = GetBackend(a)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.minimum(a, b)
    if backend == Backend.PyTorch:
        return torch.minimum(a, b)


def Maximum(a, b):
    backend = GetBackend(a)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.maximum(a, b)
    if backend == Backend.PyTorch:
        return torch.maximum(a, b)


def Sign(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.sign(tensor)
    if backend == Backend.PyTorch:
        return torch.sign(tensor)


def Where(condition, a, b):
    backend = GetBackend(a)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.where(condition, a, b)
    if backend == Backend.PyTorch:
        return torch.where(condition, a, b)


def MatMul(m1, m2):
    backend = GetBackend(m1)  # TODO: check matching types
    if backend == Backend.NumPy:
        return np.matmul(m1, m2)
    if backend == Backend.PyTorch:
        return torch.matmul(m1, m2)


def Inverse(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.linalg.inv(tensor)
    if backend == Backend.PyTorch:
        return torch.inverse(tensor)


def All(expression, backend=None):
    backend = TensorBackend if backend is None else backend
    if backend == Backend.NumPy:
        return np.all(expression)
    if backend == Backend.PyTorch:
        return torch.all(expression)


def Clamp(values, min_val, max_val):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.clip(values, min_val, max_val)
    if backend == Backend.PyTorch:
        return torch.clamp(values, min_val, max_val)


def Round(values):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.round(values)
    if backend == Backend.PyTorch:
        return torch.round(values)


def Flatten(tensor, start_dim=0):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        shape = tensor.shape[:start_dim] + (-1,)
        return tensor.reshape(shape)
    if backend == Backend.PyTorch:
        return tensor.flatten(start_dim=start_dim, end_dim=-1)


def Mean(tensor, axis=-1, keepDim=True):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.mean(tensor, axis=axis, keepdims=keepDim)
    if backend == Backend.PyTorch:
        return torch.mean(tensor, dim=axis, keepdim=keepDim)


def Gaussian(values, power=1.0, axis=-1, keepDim=True):
    if values.shape[axis] == 1:
        return values

    values = SwapAxes(values, axis, -1)

    padding = (values.shape[-1] - 1) / 2.0
    idx = Arange(0, values.shape[-1], 1)
    weight = Ones(values.shape) * Exp(-((idx - padding) ** 2) / (0.5 * padding) ** 2)

    weight = SwapAxes(weight, axis, -1)
    values = SwapAxes(values, axis, -1)

    weight = Pow(weight, power)
    values *= weight
    values = Sum(values, axis, keepDim=True) / Maximum(
        Sum(weight, axis, keepDim=True), EPS
    )

    if not keepDim:
        values = Squeeze(values, axis)

    return values


def Copy(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return tensor.copy()
    if backend == Backend.PyTorch:
        return tensor.clone()


def SwapAxes(values, axis1, axis2):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.swapaxes(values, axis1, axis2)
    if backend == Backend.PyTorch:
        return torch.swapaxes(values, axis1, axis2)


def Log(values):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.log(values)
    if backend == Backend.PyTorch:
        return torch.log(values)


def Exp(values):
    backend = GetBackend(values)
    if backend == Backend.NumPy:
        return np.exp(values)
    if backend == Backend.PyTorch:
        return torch.exp(values)


def Interpolate(a, b, weight):
    return (1 - weight) * a + weight * b


def Unsqueeze(tensor, dim):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.expand_dims(tensor, dim)
    if backend == Backend.PyTorch:
        return torch.unsqueeze(tensor, dim)


def Squeeze(tensor, dim):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.squeeze(tensor, dim)
    if backend == Backend.PyTorch:
        return torch.squeeze(tensor, dim)


def Determinant(tensor):
    backend = GetBackend(tensor)
    if backend == Backend.NumPy:
        return np.linalg.det(tensor)
    if backend == Backend.PyTorch:
        return torch.linalg.det(tensor)


def InterpolateDt(a, b, dt, rate, eps=0.01):
    if rate == 0:
        return a
    value = Interpolate(a, b, 1.0 - Exp(-dt * rate))
    if Distance(value, b) < eps:
        return b
    return value
