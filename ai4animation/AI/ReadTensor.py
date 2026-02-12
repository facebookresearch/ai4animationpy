import torch
from ai4animation.Math import Rotation, Tensor, Transform, Vector3


class ReadTensor:
    def __init__(self, name, data):
        self.Name = name
        self.Pivot = 0
        self.Shape = data.shape
        self.Fixed = list(data.shape[:-1])
        self.Dims = data.shape[-1]
        self.Flatten = len(self.Shape) - 1
        self.Data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def Verify(self, count):
        if self.Pivot + count > self.Dims:
            print(
                f"Attempting to read more values than outputs available for tensor: {self.Name} ({self.Pivot + count} / {self.Dims})"
            )
            return False
        return True

    def DetermineShape(self, shape):
        if isinstance(shape, int):
            return list(self.Fixed) + [shape]
        else:
            return list(self.Fixed) + list(shape)

    def Read(self, shape):
        size = Tensor.ShapeCapacity(shape)
        if not self.Verify(size):
            return
        values = self.Data[..., self.Pivot : self.Pivot + size]
        self.Pivot += size
        return values.reshape(self.DetermineShape(shape))

    def ReadRootDelta(self):
        if not self.Verify(3):
            return
        x, y, z = self.Data[self.Pivot : self.Pivot + 3]
        self.Pivot += 3
        return Transform.TR(Vector3.Create((x, 0, z)), Vector3.Create((0, y, 0)))

    def ReadVector3(self, shape=None, x=True, y=True, z=True):
        dim = x + y + z
        if shape is None:
            shape = [dim]
        elif isinstance(shape, int):
            shape = [shape, dim]
        else:
            shape = list(shape) + [dim]

        if x and y and z:
            values = self.Read(shape)
        else:
            mask = []
            if x:
                mask.append(0)
            if y:
                mask.append(1)
            if z:
                mask.append(2)
            values = Tensor.Shapify(Vector3.Zero(), self.DetermineShape(shape)[:-1])
            values[..., mask] = self.Read(shape)

        return values

    def ReadRotation3D(self, shape):
        z = self.ReadVector3(shape)
        y = self.ReadVector3(shape)
        return Rotation.Look(z, y)

    def GetTensor(self):
        if self.Pivot < self.Capacity:
            print(
                f"Did not feed all inputs for tensor: {self.Name} ({self.Pivot} / {self.Capacity})"
            )
        return Tensor.ToDevice(
            torch.tensor(self.Data.reshape(self.Shape), dtype=torch.float32)
        )
