import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class RunningStats(nn.Module):
    """Computes running mean and standard deviation
    Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
        * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>
        
    Adapted for usage with PyTorch by converting the original implementation from NumPy to PyTorch.
    """

    def __init__(self, dim, n=0.0, m=None, s=None):
        super(RunningStats, self).__init__()

        self.n = n
        self.m = m
        self.s = s

        self.Dim = dim

        self.Norm = Parameter(
            torch.from_numpy(
                np.vstack(
                    (np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))
                )
            ),
            requires_grad=False,
        )

    def clear(self):
        self.n = 0.0

    def Update(self, data):
        values = data.reshape(-1, self.Dim)
        if data.is_cuda:
            values = values.cpu()
        values = values.detach().numpy()
        for k in range(values.shape[0]):
            self.UpdateParams(values[k])
        mean = self.mean
        std = self.std
        std[std < 0.001] = 1.0
        self.Norm[0] = torch.from_numpy(mean)
        self.Norm[1] = torch.from_numpy(std)

    def UpdateAndNormalize(self, tensor):
        self.Update(tensor)
        return self.Normalize(tensor)

    def Normalize(self, tensor):
        mean = self.Norm[0]
        std = self.Norm[1]
        return (tensor - mean) / std

    def Denormalize(self, tensor):
        mean = self.Norm[0]
        std = self.Norm[1]
        return (tensor * std) + mean

    def UpdateParams(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = np.zeros_like(x)
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    @property
    def mean(self):
        return self.m

    def variance(self):
        return self.s / (self.n - 1) if self.n > 1 else np.zeros_like(self.s)

    @property
    def std(self):
        return np.sqrt(self.variance())


# import numpy as np
# import torch
# import torch.nn as nn

# from torch.nn.parameter import Parameter

# class RunningStats(nn.Module):
#     """Computes running mean and standard deviation
#     Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
#     Adapted from:
#         *
#         <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
# calculate-a-running-standard-deviation>
#         * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
#         * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>

#     Adapted for usage with PyTorch by converting the original implementation from NumPy to PyTorch.
#     """

#     def __init__(self, dim, n=0., m=None, s=None):
#         super(RunningStats, self).__init__()

#         self.n = n
#         self.m = m
#         self.s = s

#         self.Dim = dim

#         self.Norm = Parameter(torch.from_numpy(np.vstack((
#             np.zeros(dim, dtype=np.float32),
#             np.ones(dim, dtype=np.float32)
#         ))), requires_grad=False)

#     def clear(self):
#         self.n = 0.

#     def Update(self, data):
#         values = data.reshape(-1, self.Dim)
#         for k in range(values.shape[0]):
#             self.UpdateParams(values[k])
#         mean = self.mean
#         std = self.std
#         std[std < 0.001] = 1.0
#         self.Norm[0] = mean
#         self.Norm[1] = std

#     def UpdateAndNormalize(self, tensor):
#         self.Update(tensor)
#         return self.Normalize(tensor)

#     def Normalize(self, tensor):
#         mean = self.Norm[0]
#         std = self.Norm[1]
#         return (tensor - mean) / std

#     def Denormalize(self, tensor):
#         mean = self.Norm[0]
#         std = self.Norm[1]
#         return (tensor * std) + mean

#     def UpdateParams(self, x):
#         self.n += 1
#         if self.n == 1:
#             self.m = x
#             self.s = torch.zeros_like(x)
#         else:
#             prev_m = self.m.clone()
#             self.m += (x - self.m) / self.n
#             self.s += (x - prev_m) * (x - self.m)

#     @property
#     def mean(self):
#         return self.m

#     def variance(self):
#         return self.s / (self.n - 1) if self.n > 1 else torch.zeros_like(self.s)

#     @property
#     def std(self):
#         return torch.sqrt(self.variance())
