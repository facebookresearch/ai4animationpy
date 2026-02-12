import torch
from ai4animation.Math import Tensor


class FeedTensor:
    def __init__(self, name, shape):
        self.Name = name
        self.Pivot = 0
        self.Shape = [shape] if type(shape) is int else shape
        self.Fixed = list(self.Shape[:-1])
        self.Dims = self.Shape[-1]  # Dim should be last dimension of shape
        self.Flatten = len(self.Shape) - 1
        self.Data = Tensor.Zeros(shape)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def Verify(self, count):
        if self.Pivot + count > self.Dims:
            print(
                f"Attempting to feed more values than inputs available for tensor: {self.Name} ({self.Pivot + count} / {self.Dims})"
            )
            return False
        return True

    def Feed(self, values):
        values = Tensor.Flatten(values, start_dim=self.Flatten)
        count = values.shape[-1]
        if not self.Verify(count):
            return

        self.Data[..., self.Pivot : self.Pivot + count] = values
        self.Pivot += count

    def FeedVector3(self, values, x=True, y=True, z=True):
        mask = []
        if x:
            mask.append(0)
        if y:
            mask.append(1)
        if z:
            mask.append(2)
        values = values[..., mask]
        self.Feed(values)

    def GetTensor(self):
        if self.Pivot < self.Dims:
            print(
                f"Did not feed all inputs for tensor: {self.Name} ({self.Pivot} / {self.Dims})"
            )
        return Tensor.ToDevice(torch.tensor(self.Data, dtype=torch.float32))
