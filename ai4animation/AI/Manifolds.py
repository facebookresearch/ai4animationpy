# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


def softmax(z, dim):
    return F.softmax(z.reshape(-1, dim), dim=-1).reshape(z.shape)


def hypersphere(z):
    shape = z.shape
    z = z.reshape(-1, 2)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    z = z.reshape(shape)
    return z


def atan2(y, x):
    # y = torch.where(y == 0.0, 1e-5, y)
    return 2.0 * torch.atan((torch.sqrt(x**2 + y**2) - x) / y)


def spherical(z):
    shape = z.shape
    z = z.reshape(-1, 2)
    x = z[..., 0]
    y = z[..., 1]

    p = atan2(x, y)
    # p2 = torch.atan2(x, y)

    sin = torch.sin(p)
    cos = torch.cos(p)
    manifold = torch.stack((sin, cos), -1)
    manifold = manifold.reshape(shape)
    return manifold


def quantize(z, levels):
    projection = torch.round(z * levels) / levels
    return z + (projection - z.detach())


def gumbel_sample(logits, noise, temperature, eps=1e-10):
    U = torch.rand_like(logits) if noise is None else noise
    N = -torch.log(-torch.log(U + eps) + eps)
    y = logits + N
    return F.softmax(y / temperature, dim=-1)


def gumbel(logits, dim, noise=None, temperature=1.0, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    shape = logits.shape
    logits = logits.reshape(logits.shape[0], -1, dim)
    noise = noise if noise is None else noise.reshape(-1, dim)

    y = gumbel_sample(logits, noise, temperature)

    if not hard:
        return y.reshape(shape)

    size = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, size[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*size)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.reshape(shape)


def gumbel_soft(z, dim):
    return F.gumbel_softmax(
        z.reshape(-1, dim), tau=1, hard=False, eps=1e-10, dim=-1
    ).reshape(z.shape)


def gumbel_hard(z, dim):
    return F.gumbel_softmax(
        z.reshape(-1, dim), tau=1, hard=True, eps=1e-10, dim=-1
    ).reshape(z.shape)


# z is of shape [B x C x D]
# y will be of shape [B x C*D]
# z are probs
def categorical_discretization(z, c, d):
    shape = z.shape
    z = z.reshape(-1, c, d)
    z = z.repeat(1, d, 1)
    # z = F.gumbel_softmax(z.reshape(-1, d), tau=1, hard=True, eps=1e-10, dim=-1).reshape(z.shape)
    z = (
        D.one_hot_categorical.OneHotCategoricalStraightThrough(probs=z.reshape(-1, d))
        .rsample()
        .reshape(z.shape)
    )
    steps = torch.linspace(0.0, 1.0, d).to(z.device)
    z = torch.sum(z * steps, -1)
    z = z.reshape(shape)
    return z


# z is of shape [B x D]
# y will be of shape [B x D]
# z are probs
def categorical(z, d):
    return (
        D.one_hot_categorical.OneHotCategoricalStraightThrough(probs=z.reshape(-1, d))
        .rsample()
        .reshape(z.shape)
    )


# z are probs in shape [B x D]
# y are one-hot in shape [B x D]
def argmax(z, d):
    return F.one_hot(z.reshape(z.shape[0], -1, d).argmax(dim=-1), d).reshape(z.shape)

    # encoding_indices = torch.argmax(z.reshape(-1, d), dim=-1).unsqueeze(1)
    # encoding_one_hot = torch.zeros(encoding_indices.size(0), d, device=z.device).scatter_(1, encoding_indices, 1)
    # encoding_one_hot = z + (encoding_one_hot.reshape(z.shape) - z).detach()
    # return encoding_one_hot
