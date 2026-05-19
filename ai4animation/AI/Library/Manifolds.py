# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F


def Softmax(logits, classes):
    return F.softmax(logits.reshape(-1, classes), dim=-1).reshape(logits.shape)


def Gumbel(logits, classes, noise=None, temperature=1.0, hard=False, eps=1e-10):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    U = torch.rand_like(logits) if noise is None else noise
    N = -torch.log(-torch.log(U + eps) + eps)
    y = logits + N
    y = y.reshape(-1, classes)
    y = F.softmax(y / temperature, dim=-1)

    if not hard:
        return y.reshape(logits.shape)

    size = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, size[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*size)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.reshape(logits.shape)
